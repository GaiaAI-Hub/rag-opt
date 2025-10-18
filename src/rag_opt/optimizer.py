from typing_extensions import Annotated, Doc, Literal, Optional

from loguru import logger
from fastmobo import FastMobo
from fastmobo.mobo import OptimizationResult

from rag_opt.eval._problem import RAGOptimizationProblem
from rag_opt.search_space import RAGSearchSpace
from rag_opt._manager import RAGPipelineManager
from rag_opt.dataset import TrainDataset
from rag_opt._config import RAGConfig
from rag_opt.llm import RAGLLM, RAGEmbedding, Embeddings, BaseChatModel
from rag_opt.eval.evaluator import RAGEvaluator


AllowedAcq = Literal['qEHVI', 'qNEHVI', 'qNParEGO', 'Random']
AcqType = AllowedAcq | list[AllowedAcq]

class Optimizer:
    """Multi-Objective Bayesian Optimization for RAG pipeline"""
    
    def __init__(
        self, 
        train_dataset: Annotated[TrainDataset, Doc("Dataset for ground truth in optimization")],
        config_path: Annotated[str, Doc("Path to RAG configuration YAML file")],
        *,
        acquisition_functions: Annotated[Optional[AcqType], Doc("List of acquisition functions")] = None,
        optimizer: Annotated[Optional[FastMobo], Doc("Custom FastMobo optimizer")] = None,
        problem: Annotated[Optional[RAGOptimizationProblem], Doc("Optimization problem")] = None,
        search_space: Annotated[Optional[RAGSearchSpace], Doc("RAG search space")] = None,
        verbose: Annotated[bool, Doc("Enable optimization logging")] = True,
        evaluator_llm: Annotated[Optional[RAGLLM | str], Doc("LLM for metric evaluation")] = None, 
        evaluator_embedding: Annotated[Optional[RAGEmbedding | Embeddings | str], Doc("Embedding for metric evaluation")] = None,
        custom_evaluator: Annotated[Optional[RAGEvaluator], Doc("Custom evaluator")] = None,
        custom_rag_pipeline_manager: Annotated[Optional[RAGPipelineManager], Doc("Custom RAG pipeline manager")] = None
    ):
        """Initialize optimizer with configuration and optional custom components"""
        self.verbose = verbose
        self.train_dataset = train_dataset
        
        logger.debug(f"Loading RAG Search Space from {config_path}")
        self.search_space = search_space or RAGSearchSpace.from_yaml(config_path)

        logger.debug("Initializing RAG Pipeline Manager")
        self.rag_pipeline_manager = custom_rag_pipeline_manager or RAGPipelineManager(search_space=self.search_space,verbose=verbose)

        if custom_evaluator:
            self.evaluator_llm = custom_evaluator.evaluator_llm or self._get_evaluator_llm(evaluator_llm)
        else:
            self.evaluator_llm = self._get_evaluator_llm(evaluator_llm)
            
        self.evaluator_embedding = self._get_evaluator_embedding(evaluator_embedding)

        self.optimization_problem = problem or RAGOptimizationProblem(
            train_dataset=train_dataset,
            rag_pipeline_manager=self.rag_pipeline_manager,
            evaluator_llm=self.evaluator_llm,
            evaluator_embedding=self.evaluator_embedding
        )
        if custom_evaluator:
            self.optimization_problem.evaluator = custom_evaluator
        self.mobo_optimizer = optimizer or self._initialize_optimizer(acquisition_functions=acquisition_functions)
    
    def _get_evaluator_llm(self, evaluator_llm: Optional[RAGLLM | str]) -> RAGLLM:
        """Get or initialize evaluator LLM"""
        if evaluator_llm is None or isinstance(evaluator_llm, str):
            return self.rag_pipeline_manager.initiate_llm(evaluator_llm)
        elif isinstance(evaluator_llm, RAGLLM | BaseChatModel):
            return evaluator_llm
        else:
            logger.error(f"Invalid evaluator_llm type: {type(evaluator_llm)}")
            raise ValueError(f"Invalid evaluator_llm type: {type(evaluator_llm)}")

    def _get_evaluator_embedding(self, evaluator_embedding: Optional[RAGEmbedding | str]) -> RAGEmbedding:
        """Get or initialize evaluator embedding"""
        if evaluator_embedding is None or isinstance(evaluator_embedding, str):
            return self.rag_pipeline_manager.initiate_embedding(evaluator_embedding)
        elif isinstance(evaluator_embedding, RAGEmbedding | Embeddings):
            return evaluator_embedding
        else:
            logger.error(f"Invalid evaluator_embedding type: {type(evaluator_embedding)}")
            raise ValueError(f"Invalid evaluator_embedding type: {type(evaluator_embedding)}")
    
    def _initialize_optimizer(self, **kwargs) -> FastMobo:
        """Initialize FastMobo optimizer with initial evaluation data"""
        # Generate initial data
        logger.debug("Generating initial evaluation data")
        train_configs, evaluation_datasets = self.optimization_problem.generate_initial_data(
            n_samples=1
        )
        
        # Convert to tensors
        train_x = self.search_space.configs_to_tensor(train_configs)
        train_y = self.optimization_problem.evaluator.evaluate_batch(evaluation_datasets,
                                                                     return_tensor=True, 
                                                                     normalize=False)
        
        return FastMobo(
            problem=self.optimization_problem.create_fastmobo_problem(),
            acquisition_functions=kwargs.pop("acquisition_functions", ['qEHVI','Random']),
            batch_size=2,
            train_x=train_x, 
            train_y=train_y,
            n_initial=10,
            ref_point=self.optimization_problem.ref_point,
            bounds=self.optimization_problem.bounds,
            **kwargs
        )
    
    def optimize(self, 
                    n_trials: int = 50, 
                    best_one:bool=False,
                    *, 
                    plot_hypervolume:bool=False,
                    plot_hypervolume_path:Optional[str]=None,
                    plot_pareto:bool=False,
                    plot_pareto_path:Optional[str]=None,
                    **kwargs) -> dict[str, RAGConfig] | RAGConfig:
        """
        Run Bayesian optimization to find best RAG configuration
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best configuration per acquisition function
        """
        logger.debug(f"Running {n_trials} optimization trials...")
        
        result: OptimizationResult = self.mobo_optimizer.optimize(
            n_iterations=n_trials, 
            verbose=False, # self.verbose
        )
        logger.warning(f"Optimization complete. Hypervolumes: {result.hypervolumes}")
        
        # Extract best configs for each acquisition function
        best_configs = {}
        
        for acq_func in result.train_x.keys():
            X = result.train_x[acq_func]  
            Y = result.train_obj_true[acq_func] 
            
            avg_performance = Y.mean(dim=1)
            best_idx = avg_performance.argmax().item()
            
            best_config_tensor = X[best_idx]
            
             
            try:
                best_configs[acq_func] = self.search_space.tensor_to_config(best_config_tensor)
                logger.debug(f"  Successfully decoded Best RAG config for {acq_func} acquisition function")
            except Exception as e:
                logger.error(f"  Failed to decode config for {acq_func}: {e}")
                logger.error(f"  Tensor shape: {best_config_tensor.shape}")
                logger.error(f"  Tensor values: {best_config_tensor}")
        
        if best_one:
            best_configs = list(best_configs.values())[0]  if best_configs else None

        logger.success(f"Optimization complete. Hypervolumes: {result.hypervolumes}")

        if plot_hypervolume:
            problem = self.optimization_problem.problem or self.optimization_problem.create_fastmobo_problem()
            result.plot_convergence(problem=problem, save_path=plot_hypervolume_path)
        
        if plot_pareto:
            result.plot_objectives(save_path=plot_pareto_path)
        
        if not best_configs:
            logger.warning("optimization failed. No best configs found.")
        return best_configs
    