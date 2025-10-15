from rag_opt.eval.evaluator import RAGEvaluator
from rag_opt._manager import RAGPipelineManager
from rag_opt.dataset import EvaluationDataset
from fastmobo.problems import FastMoboProblem
from typing_extensions import Annotated, Doc
from rag_opt._sampler import SamplerType
from rag_opt.dataset import TrainDataset
from rag_opt._config import RAGConfig
from functools import cached_property
from langchain.schema import Document
from rag_opt.llm import RAGLLM,RAGEmbedding
from loguru import logger
import torch

class RAGOptimizationProblem:
    """ Multi-objective optimization problem for RAG hyperparameter tuning """
    
    def __init__(
        self,
        train_dataset: Annotated[TrainDataset, Doc("the dataset to be used as ground truth and for querying in the optimization process")],
        rag_pipeline_manager: Annotated[RAGPipelineManager, Doc("This service will be used to sample the search space and create RAG instances during optimization")], 
        evaluator_llm: Annotated[RAGLLM | str, Doc("the llm to be used in the metric evaluation process in the objective function")]=None,
        *, 
        evaluator_embedding: Annotated[RAGEmbedding | str, Doc("the embedding to be used in the metric evaluation process in the objective function")]=None,
        evaluator: Annotated[RAGEvaluator, Doc("Evaluator used to evaluate the current generated response based on specific metrics")]=None 
    ):
        if not evaluator_llm and not evaluator:
            logger.error(f"You have to provide either evaluator or evaluator_llm")
            raise ValueError(f"You have to provide either evaluator or evaluator_llm")
        
        self.evaluator = evaluator or RAGEvaluator(evaluator_llm=evaluator_llm, evaluator_embedding=evaluator_embedding)
        self.rag_pipeline_manager = rag_pipeline_manager
        self.train_dataset = train_dataset

        self.problem: FastMoboProblem = None # fastmobo problem instance 
    
    def generate_initial_data(self,n_samples: int=20, **kwargs) -> tuple[list[RAGConfig], list[EvaluationDataset]]:
        """ prepare train_x, train_y for the optimization process"""
        return self.rag_pipeline_manager.generate_initial_data(
            train_data=self.train_dataset,
            n_samples=n_samples,
            sampler_type=SamplerType.SOBOL,
            **kwargs
        )

    @property
    def documents(self) -> list[Document]:
        """ list of dataset documents it has to be used in indexing process"""
        return self.train_dataset.to_langchain_docs()
    
    @property
    def bounds(self) -> torch.Tensor:
        """Get bounds tensor from search space"""
        return self.rag_pipeline_manager.get_problem_bounds()
    
    @cached_property
    def ref_point(self) -> torch.Tensor:
        """Define reference point for hypervolume calculation"""
        ref_values = self.evaluator.ref_point
        return ref_values.detach().clone().double()

    
    @cached_property
    def objectives(self) -> list[str]:
        return self.evaluator.metric_names

    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-objective function for hyperparameter optimization
        Args:
            x: [batch_size, num_hyperparams] 
        Returns: 
            [batch_size, num_objectives]
        """
        # Handle both single sample and batched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        
        batch_size = x.shape[0]
        objectives_list = []
        
        for i in range(batch_size):
            sample = x[i]
            rag_config = self.rag_pipeline_manager.decode_sample_to_rag_config(sample=sample)
            eval_dataset = self.rag_pipeline_manager.generate_evaluation_data(
                config=rag_config,
                train_data=self.train_dataset
            )
            
            objectives = self.evaluator.evaluate(eval_dataset, return_tensor=True)
            objectives_list.append(objectives)
        
        return torch.stack(objectives_list)
        
    def create_fastmobo_problem(self) -> FastMoboProblem:
        """Create FastMoBo problem instance"""
        noise_std = torch.tensor([0.1] * len(self.objectives), dtype=torch.float)
        self.problem = FastMoboProblem(
            objective_func=self.objective_function,
            bounds=self.bounds,
            ref_point=self.ref_point,
            num_objectives=len(self.objectives),
            noise_std=noise_std,
            # dim=len(self.bounds),
            negate=False  # NOTE:: we do negation internally per metric like cost , latency
        )
        return self.problem

