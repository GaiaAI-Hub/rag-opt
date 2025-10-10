from typing import Optional
from typing_extensions import Annotated, Doc

from loguru import logger
from fastmobo import FastMobo
from fastmobo.mobo import OptimizationResult

from rag_opt.eval._problem import RAGOptimizationProblem
from rag_opt.search_space import RAGSearchSpace
from rag_opt._manager import RAGPipelineManager
from rag_opt.dataset import TrainDataset
from rag_opt._config import RAGConfig
from rag_opt.llm import RAGLLM
import torch

class Optimizer:
    """Multi-Objective Bayesian Optimization for RAG pipeline"""
    
    def __init__(
        self, 
        train_dataset: Annotated[TrainDataset, Doc("Dataset for ground truth in optimization")],
        config_path: Annotated[str, Doc("Path to RAG configuration YAML file")],
        *,
        optimizer: Annotated[Optional[FastMobo], Doc("Custom FastMobo optimizer")] = None,
        problem: Annotated[Optional[RAGOptimizationProblem], Doc("Optimization problem")] = None,
        search_space: Annotated[Optional[RAGSearchSpace], Doc("RAG search space")] = None,
        verbose: Annotated[bool, Doc("Enable optimization logging")] = True,
        evaluator_llm: Annotated[Optional[RAGLLM | str], Doc("LLM for metric evaluation")] = None, 
    ):
        """Initialize optimizer with configuration and optional custom components"""
        self.verbose = verbose
        self.train_dataset = train_dataset
        
        self.search_space = search_space or RAGSearchSpace.from_yaml(config_path)
        self.rag_pipeline_manager = RAGPipelineManager(search_space=self.search_space,verbose=verbose)
        self.evaluator_llm = self._get_evaluator_llm(evaluator_llm)
        self.optimization_problem = problem or RAGOptimizationProblem(
            train_dataset=train_dataset,
            rag_pipeline_manager=self.rag_pipeline_manager,
            evaluator_llm=self.evaluator_llm,
        )
        self.mobo_optimizer = optimizer or self._initialize_optimizer()
    
    def _get_evaluator_llm(self, evaluator_llm: Optional[RAGLLM | str]) -> RAGLLM:
        """Get or initialize evaluator LLM"""
        if evaluator_llm is None:
            return self.rag_pipeline_manager.initiate_llm()
        if isinstance(evaluator_llm, RAGLLM):
            return evaluator_llm
        return self.rag_pipeline_manager.initiate_llm(llm_name=evaluator_llm)
    
    def _initialize_optimizer(self, **kwargs) -> FastMobo:
        """Initialize FastMobo optimizer with initial training data"""
        # Generate initial data
        train_configs, evaluation_datasets = self.optimization_problem.generate_initial_data(
            n_samples=1
        )
        
        # Convert to tensors
        # TODO:: recheck matching between train_x and train_y
        train_x = self.search_space.configs_to_tensor(train_configs)
        train_y = self.optimization_problem.evaluator.evaluate_batch(evaluation_datasets,return_tensor=True)

        return FastMobo(
            problem=self.optimization_problem.create_fastmobo_problem(),
            acquisition_functions=['qEHVI', 'Random'], 
            batch_size=2,
            train_x=train_x, 
            train_y=train_y,
            n_initial=10,
            ref_point=self.optimization_problem.ref_point,
            bounds=self.optimization_problem.bounds,
            **kwargs
        )
    
    def optimize(self, n_trials: int = 50,best_one:bool=False, **kwargs) -> dict[str, RAGConfig] | RAGConfig:
        """
        Run Bayesian optimization to find best RAG configuration
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best configuration per acquisition function
        """
        logger.info(f"Running {n_trials} optimization trials...")
        
        result: OptimizationResult = self.mobo_optimizer.optimize(
            n_iterations=n_trials, 
            verbose=self.verbose
        )
        
        logger.info(f"Optimization complete. Hypervolumes: {result.hypervolumes}")
        
        # Extract best configs for each acquisition function
        best_configs = {}
        
        for acq_func in result.train_x.keys():
            X = result.train_x[acq_func]  
            Y = result.train_obj_true[acq_func] 
            
            avg_performance = Y.mean(dim=1)
            best_idx = avg_performance.argmax().item()
            
            best_config_tensor = X[best_idx]
            
            logger.info(f"  Best config index: {best_idx}")
            logger.info(f"  Best average objective: {avg_performance[best_idx]:.4f}")
            logger.info(f"  Best objectives: {Y[best_idx]}")
            
            try:
                best_configs[acq_func] = self.search_space.tensor_to_config(best_config_tensor)
                logger.info(f"  Successfully decoded to RAGConfig")
            except Exception as e:
                logger.error(f"  Failed to decode config for {acq_func}: {e}")
                logger.error(f"  Tensor shape: {best_config_tensor.shape}")
                logger.error(f"  Tensor values: {best_config_tensor}")
        
        # TODO:: return best 
        if best_one:
            best_configs = list(best_configs.values())[0] # get first value
        return best_configs