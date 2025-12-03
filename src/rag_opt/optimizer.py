from typing import Optional, Literal, Callable, Any
from typing_extensions import TypedDict
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
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import datetime
import textwrap
import pickle
import gzip



AcquisitionFunc = Literal['qEHVI', 'qNEHVI', 'qNParEGO', 'Random']


@dataclass
class SavedOptimizationResult:
    """Minimal stand-in for OptimizationResult used for plotting only."""
    train_x: dict[str, Any]
    train_obj_true: dict[str, Any]
    hypervolumes: dict[str, Any]
    meta: dict[str, Any]


class OptimizationProgress(TypedDict):
    """Real-time optimization progress report"""
    iteration: int
    total_iterations_completed: int
    hypervolumes: dict[str, list[float]]
    best_config: dict[str, dict]
    best_performance: dict[str, float]
    evaluations_per_acq: dict[str, int]
    total_evaluations: int
    acquisition_functions: list[str]
    completed: bool
    message: str


class Optimizer:
    """
    Multi-Objective Bayesian Optimization for RAG pipelines.
    """

    def __init__(
        self,
        train_dataset: TrainDataset,
        config_path: str,
        *,
        acquisition_functions: Optional[list[AcquisitionFunc] | AcquisitionFunc] = None,
        optimizer: Optional[FastMobo] = None,
        problem: Optional[RAGOptimizationProblem] = None,
        search_space: Optional[RAGSearchSpace] = None,
        verbose: bool = True,
        evaluator_llm: Optional[RAGLLM | str] = None,
        evaluator_embedding: Optional[RAGEmbedding | Embeddings | str] = None,
        custom_evaluator: Optional[RAGEvaluator] = None,
        custom_rag_pipeline_manager: Optional[RAGPipelineManager] = None,
        use_gateway: bool = True,
        gateway_api_key: Optional[str] = None,
        max_workers: int = 5,
        eager_load: bool = True,
    ):
        self.verbose = verbose
        self.train_dataset = train_dataset

        # Initialize search space
        logger.debug(f"Loading RAG Search Space from {config_path}")
        self.search_space = search_space or RAGSearchSpace.from_yaml(config_path)

        # Initialize pipeline manager with optimization settings
        logger.debug("Initializing RAG Pipeline Manager")
        self.rag_pipeline_manager = custom_rag_pipeline_manager or RAGPipelineManager(
            search_space=self.search_space, 
            verbose=verbose,
            use_gateway=use_gateway,
            gateway_api_key=gateway_api_key,
            max_workers=max_workers,
            eager_load=eager_load  # Load all components upfront for faster optimization
        )

        # Initialize evaluator components
        self.evaluator_llm = self._resolve_evaluator_llm(evaluator_llm)
        self.evaluator_embedding = self._resolve_evaluator_embedding(evaluator_embedding)

        # Initialize optimization problem
        self.optimization_problem = problem or RAGOptimizationProblem(
            train_dataset=train_dataset,
            rag_pipeline_manager=self.rag_pipeline_manager,
            evaluator_llm=self.evaluator_llm,
            evaluator_embedding=self.evaluator_embedding,
        )

        if custom_evaluator:
            self.optimization_problem.evaluator = custom_evaluator

        # Normalize acquisition functions
        self.acquisition_functions = self._normalize_acquisition_functions(acquisition_functions)

        # Initialize FastMobo optimizer
        self.mobo_optimizer = optimizer or self._initialize_optimizer()
        self._last_result: Optional[OptimizationResult] = None
        self._ensure_initial_state()

    def _resolve_evaluator_llm(self, evaluator_llm: Optional[RAGLLM | str]) -> RAGLLM:
        """Resolve evaluator LLM from string or instance."""
        if evaluator_llm is None or isinstance(evaluator_llm, str):
            return self.rag_pipeline_manager.initiate_llm(evaluator_llm)
        if isinstance(evaluator_llm, (RAGLLM, BaseChatModel)):
            return evaluator_llm
        raise ValueError(f"Invalid evaluator_llm type: {type(evaluator_llm)}")

    def _resolve_evaluator_embedding(
        self, 
        evaluator_embedding: Optional[RAGEmbedding | str]
    ) -> RAGEmbedding:
        """Resolve evaluator embedding from string or instance."""
        if evaluator_embedding is None or isinstance(evaluator_embedding, str):
            return self.rag_pipeline_manager.initiate_embedding(evaluator_embedding)
        if isinstance(evaluator_embedding, (RAGEmbedding, Embeddings)):
            return evaluator_embedding
        raise ValueError(f"Invalid evaluator_embedding type: {type(evaluator_embedding)}")

    def _normalize_acquisition_functions(
        self, 
        acquisition_functions: Optional[list[AcquisitionFunc] | AcquisitionFunc]
    ) -> list[str]:
        """Normalize acquisition functions to list format."""
        if acquisition_functions is None:
            return ['qEHVI', 'Random']
        if isinstance(acquisition_functions, str):
            return [acquisition_functions]
        return list(acquisition_functions)

    def _initialize_optimizer(self) -> FastMobo:
        """Initialize FastMobo optimizer with initial samples."""
        logger.debug("Generating initial evaluation data (this may take a moment)...")
        train_configs, evaluation_datasets = self.optimization_problem.generate_initial_data(
            n_samples=1
        )

        train_x = self.search_space.configs_to_tensor(train_configs)
        train_y = self.optimization_problem.evaluator.evaluate_batch(
            evaluation_datasets, 
            return_tensor=True, 
            normalize=False
        )
        
        logger.debug(f"Initial data: {len(train_configs)} configs evaluated")
 
        return FastMobo(
            problem=self.optimization_problem.create_fastmobo_problem(),
            acquisition_functions=self.acquisition_functions,
            batch_size=2,
            train_x=train_x,
            train_y=train_y,
            batch_limit=5,
            maxiter=100,
            mc_samples=64,
            raw_samples=512,
            num_restarts=10
        )

    def _ensure_initial_state(self) -> None:
        """Ensure optimizer has valid initial state."""
        try:
            self._last_result = self.mobo_optimizer.optimize(n_iterations=0, verbose=False)
            logger.debug("Optimizer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize with n_iterations=0: {e}")
            self._last_result = None

    def _extract_best_configs(self, result: OptimizationResult) -> dict[str, RAGConfig]:
        """Extract best configuration for each acquisition function."""
        best_configs: dict[str, RAGConfig] = {}
        
        for acq_func in result.train_x.keys():
            try:
                X = result.train_x[acq_func]
                Y = result.train_obj_true[acq_func]

                # Find best performing configuration
                avg_performance = Y.mean(dim=1)
                best_idx = avg_performance.argmax().item()
                best_tensor = X[best_idx]

                config = self.search_space.tensor_to_config(best_tensor)
                best_configs[acq_func] = config
                
                if self.verbose:
                    logger.debug(f"Best config extracted for {acq_func}")
            except Exception as e:
                logger.error(f"Failed to decode best config for {acq_func}: {e}")
                best_configs[acq_func] = None
                
        return best_configs

    def _compute_current_iteration(self, result: OptimizationResult) -> int:
        """Compute current iteration number from hypervolume history."""
        if isinstance(result.hypervolumes, dict):
            iterations = [len(v) for v in result.hypervolumes.values() if v]
            return max(iterations, default=1) - 1
        return len(result.hypervolumes) - 1 if result.hypervolumes else 0

    def _build_progress_report(self, result: OptimizationResult, message: str = "") -> OptimizationProgress:
        """Build progress report from optimization result."""
        best_configs = self._extract_best_configs(result)
        current_iteration = self._compute_current_iteration(result)

        return {
            "iteration": current_iteration,
            "total_iterations_completed": current_iteration,
            "hypervolumes": dict(result.hypervolumes or {}),
            "best_config": {
                acq: (config.to_dict() if hasattr(config, "to_dict") else config.__dict__)
                for acq, config in best_configs.items()
                if config is not None
            },
            "best_performance": {
                acq: float(Y.mean(dim=1).max().item())
                for acq, Y in result.train_obj_true.items()
            },
            "evaluations_per_acq": {
                acq: int(X.shape[0]) 
                for acq, X in result.train_x.items()
            },
            "total_evaluations": sum(int(X.shape[0]) for X in result.train_x.values()),
            "acquisition_functions": list(result.train_x.keys()),
            "completed": False,
            "message": message or f"Iteration {current_iteration}",
        }

    def run_step(self, n_iterations: int = 1) -> OptimizationProgress:
        """
        Run one or more BO iterations and return progress report.
        Call repeatedly for real-time progress updates.
        
        Args:
            n_iterations: Number of iterations to run (minimum 1)
            
        Returns:
            OptimizationProgress with current state
        """
        n_iterations = max(1, n_iterations)

        logger.info(f"Running {n_iterations} optimization iteration(s)...")

        result: OptimizationResult = self.mobo_optimizer.optimize(
            n_iterations=n_iterations,
            verbose=self.verbose,
        )
        self._last_result = result

        report = self._build_progress_report(
            result, 
            f"Completed iteration {self._compute_current_iteration(result)}"
        )

        logger.success(
            f"Step complete — Iteration {report['iteration']} — "
            f"Total evals: {report['total_evaluations']}"
        )
        return report

    def get_current_progress(self) -> OptimizationProgress:
        """Get latest state without advancing (for polling)."""
        if self._last_result is None:
            return {
                "iteration": 0,
                "total_iterations_completed": 0,
                "hypervolumes": {},
                "best_config": {},
                "best_performance": {},
                "evaluations_per_acq": {},
                "total_evaluations": 0,
                "acquisition_functions": self.acquisition_functions,
                "completed": False,
                "message": "Optimization not started yet",
            }

        report = self._build_progress_report(self._last_result, "Current state (no new iteration)")
        return report

    # def optimize(
    #     self,
    #     n_trials: int = 5,
    #     best_one: bool = False,
    #     *,
    #     progress_callback: Optional[Callable[[OptimizationProgress], None]] = None,
    #     step_size: int = 5,
    #     plot_hypervolume: bool = False,
    #     plot_hypervolume_path: Optional[str] = None,
    #     plot_pareto: bool = False,
    #     plot_pareto_path: Optional[str] = None,
    # ) -> dict[str, RAGConfig] | RAGConfig:
    #     """
    #     Run full optimization with optional live progress callback.
        
    #     Args:
    #         n_trials: Total number of optimization iterations
    #         best_one: If True, return single best config; else return all
    #         progress_callback: Optional callback for progress updates
    #         step_size: Iterations per step (for progress updates)
    #         plot_hypervolume: Whether to plot convergence
    #         plot_hypervolume_path: Path to save hypervolume plot
    #         plot_pareto: Whether to plot Pareto front
    #         plot_pareto_path: Path to save Pareto plot
            
    #     Returns:
    #         Best configuration(s) - dict if best_one=False, single config if best_one=True
    #     """
    #     logger.info(f"Starting optimization: {n_trials} trials with step size {step_size}")
        
    #     remaining = n_trials
        
    #     while remaining > 0:
    #         step = min(step_size, remaining)
    #         report = self.run_step(n_iterations=step)
            
    #         if progress_callback is not None:
    #             try:
    #                 progress_callback(report)
    #             except Exception as e:
    #                 logger.warning(f"Progress callback failed: {e}")
                
    #         remaining -= step

    #     # Final report
    #     final_report = self.get_current_progress()
    #     final_report["completed"] = True
    #     final_report["message"] = "Optimization completed successfully!"
        
    #     if progress_callback:
    #         try:
    #             progress_callback(final_report)
    #         except Exception as e:
    #             logger.warning(f"Final progress callback failed: {e}")

    #     # Extract best configurations
    #     best_configs = {
    #         acq: self.search_space.tensor_to_config(
    #             self._last_result.train_x[acq][
    #                 self._last_result.train_obj_true[acq].mean(dim=1).argmax()
    #             ]
    #         )
    #         for acq in self._last_result.train_x.keys()
    #     }

    #     # Generate plots if requested
    #     if plot_hypervolume:
    #         try:
    #             problem = (self.optimization_problem.problem or 
    #                       self.optimization_problem.create_fastmobo_problem())
    #             self._last_result.plot_convergence(problem=problem, save_path=plot_hypervolume_path)
    #             logger.info(f"Hypervolume plot saved to {plot_hypervolume_path or 'display'}")
    #         except Exception as e:
    #             logger.error(f"Failed to plot hypervolume: {e}")

    #     if plot_pareto:
    #         try:
    #             self._last_result.plot_objectives(save_path=plot_pareto_path)
    #             logger.info(f"Pareto plot saved to {plot_pareto_path or 'display'}")
    #         except Exception as e:
    #             logger.error(f"Failed to plot Pareto front: {e}")

    #     logger.success(f"Optimization complete! Evaluated {final_report['total_evaluations']} configurations")
        
    #     # Return single best or all best configs
    #     if best_one and best_configs:
    #         return list(best_configs.values())[0]
    #     return best_configs


    def optimize_with_details(
        self,
        n_trials: int = 50,
        *,
        step_size: int = 5,
        progress_callback: Optional[Callable[[OptimizationProgress], None]] = None,
        plot_hypervolume: bool = False,
        plot_hypervolume_path: Optional[str] = None,
        plot_pareto: bool = False,
        plot_pareto_path: Optional[str] = None,
    ) -> dict[str, RAGConfig]:
        """
        Run optimization with detailed progress tracking and return all configs.
        
        This method runs optimization step-by-step to enable real-time progress updates
        and comprehensive result tracking.
        
        Args:
            n_trials: Total number of optimization iterations
            step_size: Iterations per step (for progress updates)
            progress_callback: Optional callback for progress updates
            plot_hypervolume: Whether to plot convergence
            plot_hypervolume_path: Path to save hypervolume plot
            plot_pareto: Whether to plot Pareto front
            plot_pareto_path: Path to save Pareto plot
            
        Returns:
            Best configuration per acquisition function
        """
        logger.info(f"Starting optimization: {n_trials} trials with step size {step_size}")
        
        remaining = n_trials
        total_completed = 0
        
        while remaining > 0:
            step = min(step_size, remaining)
            
            # Run optimization step
            report = self.run_step(n_iterations=step)
            total_completed += step
            
            # Update report with correct total
            report["iteration"] = total_completed
            report["total_iterations_completed"] = total_completed
            
            # Call progress callback
            if progress_callback is not None:
                try:
                    progress_callback(report)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            
            remaining -= step
        
        # Final report
        final_report = self.get_current_progress()
        final_report["completed"] = True
        final_report["iteration"] = total_completed
        final_report["total_iterations_completed"] = total_completed
        final_report["message"] = "Optimization completed successfully!"
        
        if progress_callback:
            try:
                progress_callback(final_report)
            except Exception as e:
                logger.warning(f"Final progress callback failed: {e}")
        
        best_configs = self._extract_best_configs(self._last_result)
        if plot_hypervolume and self._last_result:
            try:
                problem = (self.optimization_problem.problem or 
                        self.optimization_problem.create_fastmobo_problem())
                self._last_result.plot_convergence(problem=problem, save_path=plot_hypervolume_path)
                logger.info(f"Hypervolume plot saved to {plot_hypervolume_path or 'display'}")
            except Exception as e:
                logger.error(f"Failed to plot hypervolume: {e}")
        
        if plot_pareto and self._last_result:
            try:
                objective_names = list(self.optimization_problem.objectives)
                self._last_result.plot_objectives(
                    save_path=plot_pareto_path,
                    objective_names=objective_names
                )
                logger.info(f"Pareto plot saved to {plot_pareto_path or 'display'}")
            except Exception as e:
                logger.error(f"Failed to plot Pareto front: {e}")
        
        logger.success(f"Optimization complete! Evaluated {final_report['total_evaluations']} configurations")
        return best_configs


    def optimize(self, 
                n_trials: int = 50, 
                best_one: bool = False,
                *, 
                plot_hypervolume: bool = False,
                plot_hypervolume_path: Optional[str] = None,
                plot_radar: bool = False,
                plot_pareto: bool = False,
                plot_pareto_path: Optional[str] = None,
                objective_names: Optional[list[str]] = None,
                save_result: Optional[bool] = None,
                **kwargs) -> dict[str, RAGConfig] | RAGConfig:
        """
        Run Bayesian optimization to find best RAG configuration
        
        Args:
            n_trials: Number of optimization trials
            best_one: Return single best config if True
            plot_hypervolume: Whether to plot convergence
            plot_hypervolume_path: Path to save hypervolume plot
            plot_pareto: Whether to plot Pareto front
            plot_pareto_path: Path to save Pareto plot
            
        Returns:
            Best configuration per acquisition function
        """
        logger.debug(f"Running {n_trials} optimization trials...")
        
        result: OptimizationResult = self.mobo_optimizer.optimize(
            n_iterations=n_trials, 
            verbose=False,
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
            best_configs = list(best_configs.values())[0] if best_configs else None

        logger.success(f"Optimization complete. Hypervolumes: {result.hypervolumes}")

        # Get objective names from evaluator
        objective_names = objective_names or list(self.optimization_problem.objectives) 
        
        if plot_hypervolume:
            problem = self.optimization_problem.problem or self.optimization_problem.create_fastmobo_problem()
            result.plot_convergence(problem=problem, save_path=plot_hypervolume_path)
        
        if plot_pareto:
            result.plot_objectives(
                save_path=plot_pareto_path, 
                objective_names=objective_names
            )
        
        if plot_radar:
            self.plot_radar_objectives(result, objective_names=objective_names)
        
        if not best_configs:
            logger.warning("optimization failed. No best configs found.")

        # save
        if save_result:
            self.save_result(result, "rag_opt_result.pkl.gz")
        return best_configs


    def plot_radar_objectives(
    self, 
        result: OptimizationResult,
        objective_names: Optional[list[str]] = None,
        save_path: Optional[str] = None
    ):
        """Radar plot of multi-objective performance (mean across samples),
        with automatic label wrapping + spacing to prevent text overlap.
        """

        if objective_names is None:
            if self.optimization_problem:
                objective_names = list(self.optimization_problem.objectives)
            else:
                first_obj = next(iter(result.train_obj_true.values()))
                objective_names = [f"Objective {i+1}" for i in range(first_obj.shape[1])]

        n_objectives = len(objective_names)
        n_methods = len(result.train_obj_true)

        wrapped_names = ["\n".join(textwrap.wrap(name, width=12)) for name in objective_names]

        angles = np.linspace(0, 2 * np.pi, n_objectives, endpoint=False).tolist()
        angles += angles[:1]  # close loop

        fig, axes = plt.subplots(
            1, n_methods,
            figsize=(7 * n_methods, 7),
            subplot_kw=dict(polar=True)
        )

        if n_methods == 1:
            axes = [axes]

        for ax, (method, train_obj) in zip(axes, result.train_obj_true.items()):
            data = train_obj.cpu().numpy()

            # ensure correct shape
            if data.shape[1] != n_objectives:
                logger.warning(
                    f"[Radar Plot] {method}: Expected {n_objectives} objectives, "
                    f"but got shape {data.shape}. Truncating extra metrics."
                )
                data = data[:, :n_objectives]

            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            norm_data = (data - min_vals) / (max_vals - min_vals + 1e-8)

            mean_vals = norm_data.mean(axis=0).tolist()
            mean_vals += mean_vals[:1]

            ax.plot(angles, mean_vals, linewidth=2)
            ax.fill(angles, mean_vals, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(
                wrapped_names,
                fontsize=11,
                linespacing=1.0
            )

            for label, angle in zip(ax.get_xticklabels(), angles):
                label.set_horizontalalignment(
                    "right" if angle > np.pi/2 and angle < 3*np.pi/2 else "left"
                )

            ax.set_rlabel_position(30)
            ax.set_title(method, fontsize=15, fontweight="bold", pad=20)

        # plt.suptitle(
        #     "Multi-Objective Radar Chart (Mean Normalized Performance)",
        #     fontsize=18,
        #     fontweight="bold",
        #     y=1.05
        # )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def save_result(self, result: OptimizationResult, path: str):
        """
        Save optimization result (train_x, train_obj_true, hypervolumes, metadata).
        The result can later be reloaded and used to generate all plots.
        """

        payload = {
            "train_x": {k: v.cpu() for k, v in result.train_x.items()},
            "train_obj_true": {k: v.cpu() for k, v in result.train_obj_true.items()},
            "hypervolumes": result.hypervolumes,
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "objective_names": list(self.optimization_problem.objectives)
                    if self.optimization_problem else None,
                "acquisition_functions": list(result.train_x.keys()),
                "search_space": getattr(self.search_space, "bounds", None),
                "version": "1.0",
            }
        }

        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.success(f"Saved optimization result to {path}")

    def load_result(self, path: str) -> SavedOptimizationResult:
        """
        Load a saved optimization result and rebuild a lightweight result object 
        that can be used for plotting and analysis.
        """

        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)

        result = SavedOptimizationResult(
            train_x=payload["train_x"],
            train_obj_true=payload["train_obj_true"],
            hypervolumes=payload["hypervolumes"],
            meta=payload["meta"],
        )

        logger.success(f"Loaded optimization result from {path}")
        return result
