from rag_opt.eval.metrics.base import MetricResult, MetricCategory, MetricScope, BaseMetric
from rag_opt.eval.metrics.retriever import (RetrievalPrecision, 
                                            RetrievalRecall,
                                            RetrievalF1,
                                             MRR, MAP, NDCG,
                                             ContextPrecision,
                                             ContextEntitiesRecall,
                                             TokenPrecision,
                                             TokenRecall,
                                             TokenF1,
                                             )
from rag_opt.eval.metrics.generation import (Faithfulness, 
                                             ResponseRelevancy,
                                             NoiseSensitivity, 
                                             FaithfulnessHHEM, 
                                             BleuScore, 
                                             RougeScore, 
                                             SemanticScore,
                                             MeteorScore,
                                             BertScore,
                                             MetricScope,
                                             GEval
                                            )
from rag_opt.eval.metrics.full import CostMetric, LatencyMetric, SafetyMetric, AlignmentMetric
from rag_opt.dataset import RAGDataset
from rag_opt.llm import RAGLLM
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from fastmobo.problems import FastMoboProblem
from fastmobo.mobo import FastMobo
import itertools
from rag_opt.rag.rag import GAIARAG
from rag_opt.rag.config import RAGConfig
import torch
import time

# TODO:: move optimizer to optimizer.py
class HyperparameterSpace:
    """
    Defines the hyperparameter search space with both continuous and categorical variables
    """
    
    def __init__(self):
        # Define categorical choices
        self.categorical_choices = {
            'vector_store_provider': ['faiss', 'chroma', 'pinecone', 'weaviate'],
            'search_type': ['similarity', 'mmr', 'similarity_score_threshold'],
            'llm_provider': ['openai', 'anthropic', 'huggingface', 'azure'],
            'llm_model': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'claude-3-sonnet'],
            'embedding_provider': ['openai', 'huggingface', 'sentence-transformers'],
            'reranker_type': ['cross_encoder', 'colbert', 'bge'],
        }
        
        # Define continuous parameter bounds [min, max]
        self.continuous_bounds = {
            'chunk_size': [200, 2000],
            'chunk_overlap': [0, 500],
            'temperature': [0.0, 2.0],
            'k': [1, 20]
        }
        
        # Boolean parameters
        self.boolean_params = ['use_reranker']
        
        # Create encoding mappings
        self._create_encodings()
        
    def _create_encodings(self):
        """Create encodings for categorical variables"""
        self.categorical_to_index = {}
        self.index_to_categorical = {}
        
        for param, choices in self.categorical_choices.items():
            self.categorical_to_index[param] = {choice: i for i, choice in enumerate(choices)}
            self.index_to_categorical[param] = {i: choice for i, choice in enumerate(choices)}
    
    def get_bounds_tensor(self) -> torch.Tensor:
        """Get bounds tensor for all parameters in order"""
        bounds = []
        
        # Continuous parameters
        for param in ['chunk_size', 'chunk_overlap', 'temperature', 'k']:
            bounds.append(self.continuous_bounds[param])
        
        # Categorical parameters (as indices)
        for param in ['vector_store_provider', 'search_type', 'llm_provider', 
                     'llm_model', 'embedding_provider', 'reranker_type']:
            max_index = len(self.categorical_choices[param]) - 1
            bounds.append([0, max_index])
        
        # Boolean parameters (0 or 1)
        for param in self.boolean_params:
            bounds.append([0, 1])
        
        bounds_array = torch.tensor(bounds, dtype=torch.double).T  # Shape: [2, num_params]
        return bounds_array
    
    def decode_parameters(self, x: torch.Tensor) -> RAGConfig:
        """Convert optimization tensor to RAGConfig"""
        if x.dim() > 1:
            x = x[0]  # Take first row if batch
        
        params = x.tolist()
        idx = 0
        
        # Continuous parameters
        chunk_size = int(params[idx]); idx += 1
        chunk_overlap = int(params[idx]); idx += 1
        temperature = params[idx]; idx += 1
        k = int(params[idx]); idx += 1
        
        # Categorical parameters
        vector_store_provider = self.index_to_categorical['vector_store_provider'][int(params[idx])]; idx += 1
        search_type = self.index_to_categorical['search_type'][int(params[idx])]; idx += 1
        llm_provider = self.index_to_categorical['llm_provider'][int(params[idx])]; idx += 1
        llm_model = self.index_to_categorical['llm_model'][int(params[idx])]; idx += 1
        embedding_provider = self.index_to_categorical['embedding_provider'][int(params[idx])]; idx += 1
        reranker_type = self.index_to_categorical['reranker_type'][int(params[idx])]; idx += 1
        
        # Boolean parameters
        use_reranker = bool(int(params[idx]) > 0.5); idx += 1
        
        return RAGConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_store_provider=vector_store_provider,
            search_type=search_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            k=k,
            embedding_provider=embedding_provider,
            use_reranker=use_reranker,
            reranker_type=reranker_type
        )
    
    def encode_config(self, config: RAGConfig) -> torch.Tensor:
        """Convert RAGConfig to optimization tensor"""
        params = []
        
        # Continuous parameters
        params.extend([
            float(config.chunk_size),
            float(config.chunk_overlap),
            config.temperature,
            float(config.k)
        ])
        
        # Categorical parameters
        params.extend([
            float(self.categorical_to_index['vector_store_provider'][config.vector_store_provider]),
            float(self.categorical_to_index['search_type'][config.search_type]),
            float(self.categorical_to_index['llm_provider'][config.llm_provider]),
            float(self.categorical_to_index['llm_model'][config.llm_model]),
            float(self.categorical_to_index['embedding_provider'][config.embedding_provider]),
            float(self.categorical_to_index['reranker_type'][config.reranker_type])
        ])
        
        # Boolean parameters
        params.append(float(config.use_reranker))
        
        return torch.tensor(params, dtype=torch.double)
    
    def get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names"""
        return [
            'chunk_size', 'chunk_overlap', 'temperature', 'k',
            'vector_store_provider', 'search_type', 'llm_provider', 
            'llm_model', 'embedding_provider', 'reranker_type',
            'use_reranker'
        ]

class RAGEvaluationPipeline:
    """
    Pipeline that combines GAIARAG with RAGEvaluator for comprehensive evaluation
    """
    
    def __init__(self, 
                 base_embeddings_dict: Dict[str, Any],  # Different embedding providers
                 base_vector_stores_dict: Dict[str, Any],  # Different vector stores
                 base_llms_dict: Dict[str, Any],  # Different LLMs
                 base_rerankers_dict: Dict[str, Any],  # Different rerankers
                 evaluator: Optional['RAGEvaluator'] = None):
        
        self.base_embeddings_dict = base_embeddings_dict
        self.base_vector_stores_dict = base_vector_stores_dict
        self.base_llms_dict = base_llms_dict
        self.base_rerankers_dict = base_rerankers_dict
        self.evaluator = evaluator or RAGEvaluator()
        
        # Current RAG instance
        self.current_rag = None
        self.current_config = None
    
    def setup_rag(self, config: RAGConfig) -> 'GAIARAG':
        """Setup RAG with specific configuration"""
        try:
            # Get components based on config
            embeddings = self._get_embeddings(config)
            vector_store = self._get_vector_store(config)
            llm = self._get_llm(config)
            reranker = self._get_reranker(config) if config.use_reranker else None
            
            retriever_config = config.to_dict()
            
            rag = GAIARAG(
                embeddings=embeddings,
                vector_store=vector_store,
                llm=llm,
                reranker=reranker,
                retriever_config=retriever_config
            )
            
            self.current_rag = rag
            self.current_config = config
            return rag
            
        except Exception as e:
            print(f"Failed to setup RAG with config {config}: {e}")
            raise
    
    def _get_embeddings(self, config: RAGConfig):
        """Get embeddings based on provider"""
        if config.embedding_provider in self.base_embeddings_dict:
            return self.base_embeddings_dict[config.embedding_provider]
        else:
            # Fallback to first available
            return list(self.base_embeddings_dict.values())[0]
    
    def _get_vector_store(self, config: RAGConfig):
        """Get vector store based on provider"""
        if config.vector_store_provider in self.base_vector_stores_dict:
            return self.base_vector_stores_dict[config.vector_store_provider]
        else:
            return list(self.base_vector_stores_dict.values())[0]
    
    def _get_llm(self, config: RAGConfig):
        """Get LLM based on provider and model"""
        # You might want to have a more sophisticated mapping here
        key = f"{config.llm_provider}_{config.llm_model}"
        if key in self.base_llms_dict:
            llm = self.base_llms_dict[key]
            # Set temperature if the LLM supports it
            if hasattr(llm, 'temperature'):
                llm.temperature = config.temperature
            return llm
        elif config.llm_provider in self.base_llms_dict:
            return self.base_llms_dict[config.llm_provider]
        else:
            return list(self.base_llms_dict.values())[0]
    
    def _get_reranker(self, config: RAGConfig):
        """Get reranker based on type"""
        if config.reranker_type in self.base_rerankers_dict:
            return self.base_rerankers_dict[config.reranker_type]
        else:
            return list(self.base_rerankers_dict.values())[0] if self.base_rerankers_dict else None
    
    def evaluate_rag_performance(self, 
                                queries: List[str],
                                ground_truths: Optional[List[str]] = None,
                                config: Optional[RAGConfig] = None) -> Dict[str, Any]:
        """
        Evaluate RAG performance on a set of queries with comprehensive metrics
        """
        if config:
            try:
                self.setup_rag(config)
            except Exception as e:
                # Return penalty scores for failed configurations
                return self._get_penalty_scores(str(e))
        
        if not self.current_rag:
            return self._get_penalty_scores("RAG not setup")
        
        all_results = []
        total_cost = 0.0
        total_latency = 0.0
        total_setup_cost = self._estimate_setup_cost(self.current_config)
        
        for i, query in enumerate(queries):
            try:
                # Time the RAG execution
                start_time = time.time()
                
                # Get documents for evaluation
                retrieved_docs = self.current_rag.retriever.retrieve(query)
                retrieved_contexts = [doc.page_content for doc in retrieved_docs]
                
                # Get RAG response
                response = self.current_rag.get_answer(query, use_reranker=self.current_config.use_reranker)
                
                end_time = time.time()
                query_latency = end_time - start_time
                total_latency += query_latency
                
                # Calculate query-specific cost
                query_cost = self._estimate_query_cost(self.current_config, query, response)
                total_cost += query_cost
                
                # Prepare evaluation parameters
                eval_params = {
                    'query': query,
                    'contexts': retrieved_contexts,
                    'response': response,
                    'ground_truth': ground_truths[i] if ground_truths else None,
                    'retrieved_docs': retrieved_docs,
                    'config': self.current_config,
                    'latency': query_latency,
                    'cost': query_cost
                }
                
                # Evaluate metrics for this query
                query_results = self.evaluator.evaluate_all(**eval_params)
                all_results.append(query_results)
                
            except Exception as e:
                print(f"Query evaluation failed: {e}")
                # Add penalty for failed queries
                total_cost += 10.0  # High penalty cost
                total_latency += 30.0  # High penalty latency
                all_results.append(self._get_penalty_query_results())
        
        # Aggregate results
        aggregated_results = self._aggregate_query_results(all_results)
        
        # Add overall metrics including configuration costs
        aggregated_results['overall'] = {
            'total_cost': total_cost + total_setup_cost,
            'avg_latency': total_latency / len(queries) if queries else 30.0,
            'total_latency': total_latency,
            'setup_cost': total_setup_cost,
            'num_queries': len(queries),
            'config_complexity': self._calculate_config_complexity(self.current_config)
        }
        
        return aggregated_results
    
    def _estimate_setup_cost(self, config: RAGConfig) -> float:
        """Estimate setup cost based on configuration complexity"""
        setup_cost = 0.0
        
        # Model loading costs (higher for larger models)
        model_costs = {
            'gpt-3.5-turbo': 0.1,
            'gpt-4': 0.5,
            'gpt-4-turbo': 0.3,
            'claude-3-sonnet': 0.4
        }
        setup_cost += model_costs.get(config.llm_model, 0.2)
        
        # Vector store setup costs
        vs_costs = {'faiss': 0.0, 'chroma': 0.1, 'pinecone': 0.2, 'weaviate': 0.15}
        setup_cost += vs_costs.get(config.vector_store_provider, 0.1)
        
        # Reranker costs
        if config.use_reranker:
            setup_cost += 0.1
        
        return setup_cost
    
    def _estimate_query_cost(self, config: RAGConfig, query: str, response: str) -> float:
        """Estimate per-query cost based on configuration and usage"""
        cost = 0.0
        
        # Token-based costs (simplified)
        input_tokens = len(query.split()) + (config.k * 100)  # Assuming 100 tokens per context
        output_tokens = len(response.split())
        
        # Model pricing (per 1K tokens)
        pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'claude-3-sonnet': {'input': 0.015, 'output': 0.075}
        }
        
        model_pricing = pricing.get(config.llm_model, pricing['gpt-3.5-turbo'])
        cost += (input_tokens / 1000) * model_pricing['input']
        cost += (output_tokens / 1000) * model_pricing['output']
        
        # Embedding costs
        embedding_cost = (len(query.split()) / 1000) * 0.0001  # Approximate
        cost += embedding_cost
        
        # Reranker costs
        if config.use_reranker:
            cost += config.k * 0.001  # Per document reranking cost
        
        return cost
    
    def _calculate_config_complexity(self, config: RAGConfig) -> float:
        """Calculate configuration complexity score"""
        complexity = 0.0
        
        # Model complexity
        model_complexity = {
            'gpt-3.5-turbo': 1.0,
            'gpt-4': 3.0,
            'gpt-4-turbo': 2.0,
            'claude-3-sonnet': 2.5
        }
        complexity += model_complexity.get(config.llm_model, 1.5)
        
        # Vector store complexity
        vs_complexity = {'faiss': 1.0, 'chroma': 1.5, 'pinecone': 2.0, 'weaviate': 1.8}
        complexity += vs_complexity.get(config.vector_store_provider, 1.5)
        
        # Reranker adds complexity
        if config.use_reranker:
            complexity += 1.0
        
        # High k values add complexity
        complexity += config.k * 0.1
        
        return complexity
    
    def _get_penalty_scores(self, error_msg: str) -> Dict[str, float]:
        """Return penalty scores for failed configurations"""
        return {
            'cost': 1000.0,  # High penalty
            'avg_latency': 100.0,  # High penalty
            'faithfulness': 0.0,
            'response_relevancy': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'overall': {
                'total_cost': 1000.0,
                'avg_latency': 100.0,
                'setup_cost': 100.0,
                'config_complexity': 10.0,
                'error': error_msg
            }
        }
    
    def _get_penalty_query_results(self) -> Dict[str, 'MetricResult']:
        """Return penalty results for failed query evaluation"""
        # This would return MetricResult objects with penalty values
        # Implementation depends on your MetricResult structure
        return {}
    
    def _aggregate_query_results(self, all_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across multiple queries"""
        if not all_results:
            return {}
        
        # Get all metric names
        metric_names = set()
        for results in all_results:
            if isinstance(results, dict):
                metric_names.update(results.keys())
        
        aggregated = {}
        
        for metric_name in metric_names:
            values = []
            for results in all_results:
                if isinstance(results, dict) and metric_name in results:
                    if hasattr(results[metric_name], 'value') and not hasattr(results[metric_name], 'error'):
                        # MetricResult object
                        metric_result = results[metric_name]
                        if isinstance(metric_result.value, dict):
                            values.append(metric_result.value.get('total', 0.0))
                        else:
                            values.append(metric_result.value)
                    elif isinstance(results[metric_name], (int, float)):
                        # Direct numeric value
                        values.append(results[metric_name])
            
            if values:
                aggregated[metric_name] = sum(values) / len(values)
            else:
                aggregated[metric_name] = 0.0
        
        return aggregated

class RAGOptimizationProblem:
    """
    Multi-objective optimization problem for RAG hyperparameter tuning
    Handles both continuous and categorical variables
    """
    
    def __init__(self,
                 rag_pipeline: RAGEvaluationPipeline,
                 evaluation_queries: List[str],
                 ground_truths: Optional[List[str]] = None,
                 optimization_objectives: List[str] = None):
        
        self.rag_pipeline = rag_pipeline
        self.evaluation_queries = evaluation_queries
        self.ground_truths = ground_truths
        self.hyperparameter_space = HyperparameterSpace()
        
        # Define objectives to optimize
        self.objectives = optimization_objectives or [
            'total_cost',         # minimize (will be negated)
            'avg_latency',        # minimize (will be negated) 
            'config_complexity',  # minimize (will be negated)
            'faithfulness',       # maximize
            'response_relevancy', # maximize
            'context_precision'   # maximize
        ]
        
        # Get parameter bounds
        self.bounds = self.hyperparameter_space.get_bounds_tensor()
        
        # Reference point for hypervolume calculation
        self.ref_point = self._define_reference_point()
    
    def _define_reference_point(self) -> torch.Tensor:
        """Define reference point for hypervolume calculation"""
        # For objectives we want to minimize (cost, latency, complexity): negative values
        # For objectives we want to maximize (faithfulness, relevancy, precision): low positive values
        ref_values = []
        for obj in self.objectives:
            if obj in ['total_cost', 'avg_latency', 'config_complexity']:
                ref_values.append(-1000.0)  # Minimize objectives
            else:
                ref_values.append(0.0)      # Maximize objectives
        
        return torch.tensor(ref_values, dtype=torch.double)
    
    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-objective function for hyperparameter optimization
        x: [batch_size, num_hyperparams] 
        Returns: [batch_size, num_objectives]
        """
        batch_size = x.shape[0]
        results = torch.zeros(batch_size, len(self.objectives), dtype=torch.double)
        
        for i in range(batch_size):
            try:
                # Decode hyperparameters to RAGConfig
                config = self.hyperparameter_space.decode_parameters(x[i])
                
                # Evaluate RAG performance with this configuration
                evaluation_results = self.rag_pipeline.evaluate_rag_performance(
                    queries=self.evaluation_queries,
                    ground_truths=self.ground_truths,
                    config=config
                )
                
                # Extract objective values
                for j, objective in enumerate(self.objectives):
                    if objective in evaluation_results:
                        results[i, j] = float(evaluation_results[objective])
                    elif objective in evaluation_results.get('overall', {}):
                        results[i, j] = float(evaluation_results['overall'][objective])
                    else:
                        # Default penalty for missing metrics
                        if objective in ['total_cost', 'avg_latency', 'config_complexity']:
                            results[i, j] = 1000.0  # High penalty for minimize objectives
                        else:
                            results[i, j] = 0.0     # Low score for maximize objectives
                
            except Exception as e:
                print(f"Evaluation failed for configuration {i}: {e}")
                # Assign penalty values
                for j, objective in enumerate(self.objectives):
                    if objective in ['total_cost', 'avg_latency', 'config_complexity']:
                        results[i, j] = 1000.0  # High penalty
                    else:
                        results[i, j] = 0.0     # Low score
        
        return results
    
    def create_fastmobo_problem(self) -> FastMoboProblem:
        """Create FastMoBo problem instance"""
        # Noise standard deviation for each objective
        noise_std = torch.tensor([0.1] * len(self.objectives), dtype=torch.double)
        
        problem = FastMoboProblem(
            objective_func=self.objective_function,
            bounds=self.bounds,
            ref_point=self.ref_point,
            num_objectives=len(self.objectives),
            noise_std=noise_std,
            negate=True  # Important: We negate to turn minimization into maximization
        )
        
        return problem

class RAGHyperparameterOptimizer:
    """
    Complete RAG hyperparameter optimization workflow with categorical support
    """
    
    def __init__(self,
                 rag_pipeline: RAGEvaluationPipeline,
                 evaluation_queries: List[str],
                 ground_truths: Optional[List[str]] = None,
                 objectives: Optional[List[str]] = None):
        
        self.rag_pipeline = rag_pipeline
        self.optimization_problem = RAGOptimizationProblem(
            rag_pipeline=rag_pipeline,
            evaluation_queries=evaluation_queries,
            ground_truths=ground_truths,
            optimization_objectives=objectives
        )
    
    def optimize(self,
                 n_iterations: int = 25,
                 batch_size: int = 4,
                 acquisition_functions: List[str] = None) -> Dict[str, Any]:
        """
        Run multi-objective optimization for RAG hyperparameters
        """
        acquisition_functions = acquisition_functions or ['qEHVI', 'Random']
        
        # Create FastMoBo problem
        problem = self.optimization_problem.create_fastmobo_problem()
        
        # Initialize optimizer
        optimizer = FastMobo(
            problem=problem,
            acquisition_functions=acquisition_functions,
            batch_size=batch_size
        )
        
        print("="*80)
        print("🚀 RAG Hyperparameter Optimization Started")
        print("="*80)
        print(f"📊 Objectives: {self.optimization_problem.objectives}")
        print(f"🔧 Parameters: {self.optimization_problem.hyperparameter_space.get_parameter_names()}")
        print(f"⚙️  Iterations: {n_iterations}, Batch size: {batch_size}")
        print(f"📝 Evaluation queries: {len(self.optimization_problem.evaluation_queries)}")
        print("="*80)
        
        # Run optimization
        result = optimizer.optimize(n_iterations=n_iterations, verbose=True)
        
        # Process results
        processed_results = self._process_optimization_results(result)
        
        print("\n" + "="*80)
        print("✅ Optimization Complete!")
        print(f"🏆 Found {len(processed_results['best_configurations'])} Pareto-optimal configurations")
        print("="*80)
        
        return processed_results
    
    def _process_optimization_results(self, result) -> Dict[str, Any]:
        """Process and interpret optimization results"""
        processed = {
            'optimization_result': result,
            'best_configurations': [],
            'pareto_front': {},
            'optimization_summary': {}
        }
        
        # Extract Pareto front if available
        if hasattr(result, 'pareto_Y') and hasattr(result, 'pareto_X'):
            pareto_X = result.pareto_X
            pareto_Y = result.pareto_Y
            
            # Process top configurations
            num_configs = min(10, len(pareto_X))  # Top 10 configurations
            
            for i in range(num_configs):
                config = self.optimization_problem.hyperparameter_space.decode_parameters(pareto_X[i])
                objective_values = {
                    obj: pareto_Y[i][j].item()
                    for j, obj in enumerate(self.optimization_problem.objectives)
                }
                
                processed['best_configurations'].append({
                    'rank': i + 1,
                    'config': config,
                    'objectives': objective_values,
                    'config_dict': self._config_to_dict(config),
                    'summary': self._summarize_config(config, objective_values)
                })
            
            # Store Pareto front data
            processed['pareto_front'] = {
                'pareto_X': pareto_X,
                'pareto_Y': pareto_Y,
                'hypervolume': getattr(result, 'hypervolume', None)
            }
            
            # Create optimization summary
            processed['optimization_summary'] = self._create_optimization_summary(pareto_Y)
        
        return processed
    
    def _config_to_dict(self, config: RAGConfig) -> Dict[str, Any]:
        """Convert RAGConfig to dictionary for easy viewing"""
        return {
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'vector_store_provider': config.vector_store_provider,
            'search_type': config.search_type,
            'llm_provider': config.llm_provider,
            'llm_model': config.llm_model,
            'temperature': round(config.temperature, 3),
            'k': config.k,
            'embedding_provider': config.embedding_provider,
            'use_reranker': config.use_reranker,
            'reranker_type': config.reranker_type
        }
    
    def _summarize_config(self, config: RAGConfig, objectives: Dict[str, float]) -> str:
        """Create human-readable summary of configuration"""
        cost = objectives.get('total_cost', 0)
        latency = objectives.get('avg_latency', 0)
        faithfulness = objectives.get('faithfulness', 0)
        relevancy = objectives.get('response_relevancy', 0)
        
        return f"Cost: ${cost:.3f}, Latency: {latency:.2f}s, Quality: {(faithfulness + relevancy) / 2:.3f}, Model: {config.llm_model}"
    
    def _create_optimization_summary(self, pareto_Y: torch.Tensor) -> Dict[str, Any]:
        """Create summary statistics of the optimization"""
        pareto_array = pareto_Y.numpy()
        
        summary = {}
        for i, obj in enumerate(self.optimization_problem.objectives):
            obj_values = pareto_array[:, i]
            summary[obj] = {
                'best': float(obj_values.max()),
                'worst': float(obj_values.min()),
                'mean': float(obj_values.mean()),
                'std': float(obj_values.std())
            }
        
        return summary

# ============== ADVANCED OPTIMIZATION STRATEGIES ==============

class GridSearchOptimizer:
    """
    Grid search optimizer for comparison with Bayesian optimization
    Useful for smaller hyperparameter spaces or validation
    """
    
    def __init__(self, rag_pipeline: RAGEvaluationPipeline):
        self.rag_pipeline = rag_pipeline
    
    def create_grid_search_space(self, 
                                 reduced_space: bool = True) -> List[RAGConfig]:
        """Create grid search configurations"""
        if reduced_space:
            # Reduced space for faster experimentation
            grid = {
                'chunk_size': [500, 1000, 1500],
                'chunk_overlap': [50, 100, 200],
                'temperature': [0.3, 0.7, 1.0],
                'k': [3, 5, 8],
                'search_type': ['similarity', 'mmr'],
                'llm_model': ['gpt-3.5-turbo', 'gpt-4'],
                'use_reranker': [False, True]
            }
        else:
            # Full space (warning: can be very large)
            grid = {
                'chunk_size': [200, 500, 1000, 1500, 2000],
                'chunk_overlap': [0, 50, 100, 200, 300],
                'temperature': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
                'k': [2, 3, 5, 8, 12, 15],
                'vector_store_provider': ['faiss', 'chroma'],
                'search_type': ['similarity', 'mmr', 'similarity_score_threshold'],
                'llm_model': ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet'],
                'embedding_provider': ['openai', 'huggingface'],
                'use_reranker': [False, True],
                'reranker_type': ['cross_encoder', 'colbert']
            }
        
        # Generate all combinations
        keys = list(grid.keys())
        values = list(grid.values())
        
        configs = []
        for combination in itertools.product(*values):
            config_dict = dict(zip(keys, combination))
            # Fill in defaults for missing parameters
            config = RAGConfig(
                chunk_size=config_dict.get('chunk_size', 1000),
                chunk_overlap=config_dict.get('chunk_overlap', 50),
                vector_store_provider=config_dict.get('vector_store_provider', 'faiss'),
                search_type=config_dict.get('search_type', 'similarity'),
                llm_provider=config_dict.get('llm_provider', 'openai'),
                llm_model=config_dict.get('llm_model', 'gpt-3.5-turbo'),
                temperature=config_dict.get('temperature', 0.7),
                k=config_dict.get('k', 4),
                embedding_provider=config_dict.get('embedding_provider', 'openai'),
                use_reranker=config_dict.get('use_reranker', False),
                reranker_type=config_dict.get('reranker_type', 'cross_encoder')
            )
            configs.append(config)
        
        return configs
    
    def grid_search(self,
                    evaluation_queries: List[str],
                    ground_truths: Optional[List[str]] = None,
                    reduced_space: bool = True) -> Dict[str, Any]:
        """Perform grid search optimization"""
        
        configs = self.create_grid_search_space(reduced_space)
        print(f"🔍 Grid Search: Evaluating {len(configs)} configurations...")
        
        results = []
        for i, config in enumerate(configs):
            print(f"📊 Evaluating configuration {i+1}/{len(configs)}")
            
            try:
                eval_result = self.rag_pipeline.evaluate_rag_performance(
                    queries=evaluation_queries,
                    ground_truths=ground_truths,
                    config=config
                )
                
                results.append({
                    'config': config,
                    'config_dict': self._config_to_dict(config),
                    'results': eval_result
                })
                
            except Exception as e:
                print(f"❌ Configuration {i+1} failed: {e}")
                continue
        
        # Sort by composite score
        sorted_results = self._rank_grid_results(results)
        
        return {
            'best_configurations': sorted_results[:10],
            'all_results': sorted_results,
            'grid_size': len(configs),
            'successful_evaluations': len(results)
        }
    
    def _config_to_dict(self, config: RAGConfig) -> Dict[str, Any]:
        """Convert config to dict for display"""
        return {
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'temperature': config.temperature,
            'k': config.k,
            'llm_model': config.llm_model,
            'use_reranker': config.use_reranker
        }
    
    def _rank_grid_results(self, results: List[Dict]) -> List[Dict]:
        """Rank results by composite score"""
        def calculate_score(result):
            overall = result['results'].get('overall', {})
            # Normalize and combine metrics (lower cost/latency is better, higher quality is better)
            cost_score = 1.0 / (1.0 + overall.get('total_cost', 1000))
            latency_score = 1.0 / (1.0 + overall.get('avg_latency', 100))
            quality_score = (
                result['results'].get('faithfulness', 0) + 
                result['results'].get('response_relevancy', 0)
            ) / 2.0
            
            # Weighted composite score
            return 0.3 * cost_score + 0.3 * latency_score + 0.4 * quality_score
        
        # Add scores and sort
        for result in results:
            result['composite_score'] = calculate_score(result)
        
        return sorted(results, key=lambda x: x['composite_score'], reverse=True)

# ============== USAGE EXAMPLE AND UTILITIES ==============

class RAGOptimizationWorkflow:
    """
    Complete workflow for RAG optimization including data preparation,
    optimization, evaluation, and deployment
    """
    
    def __init__(self,
                 embeddings_dict: Dict[str, Any],
                 vector_stores_dict: Dict[str, Any], 
                 llms_dict: Dict[str, Any],
                 rerankers_dict: Dict[str, Any]):
        
        self.rag_pipeline = RAGEvaluationPipeline(
            base_embeddings_dict=embeddings_dict,
            base_vector_stores_dict=vector_stores_dict,
            base_llms_dict=llms_dict,
            base_rerankers_dict=rerankers_dict
        )
        
        self.bayesian_optimizer = None
        self.grid_optimizer = GridSearchOptimizer(self.rag_pipeline)
    
    def prepare_evaluation_dataset(self, 
                                   queries_file: Optional[str] = None,
                                   ground_truths_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Prepare evaluation dataset from files or generate synthetic data"""
        
        if queries_file and ground_truths_file:
            # Load from files
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            with open(ground_truths_file, 'r') as f:
                ground_truths = [line.strip() for line in f if line.strip()]
        else:
            # Generate synthetic evaluation data
            queries = self._generate_synthetic_queries()
            ground_truths = self._generate_synthetic_ground_truths(queries)
        
        return queries, ground_truths
    
    def _generate_synthetic_queries(self) -> List[str]:
        """Generate synthetic queries for evaluation"""
        return [
            "What is the definition of artificial intelligence?",
            "How does machine learning differ from traditional programming?",
            "What are the main types of neural networks?",
            "Explain the concept of natural language processing.",
            "What is the difference between supervised and unsupervised learning?",
            "How do recommendation systems work?",
            "What is deep learning and how does it relate to AI?",
            "Explain the importance of data preprocessing in ML.",
            "What are the ethical considerations in AI development?",
            "How do large language models like GPT work?"
        ]
    
    def _generate_synthetic_ground_truths(self, queries: List[str]) -> List[str]:
        """Generate corresponding ground truths"""
        # In practice, these would be human-annotated or from a curated dataset
        return [
            "AI is the simulation of human intelligence in machines.",
            "ML learns patterns from data, traditional programming uses explicit rules.",
            "Main types include feedforward, recurrent, and convolutional networks.",
            "NLP enables computers to understand and process human language.",
            "Supervised learning uses labeled data, unsupervised finds hidden patterns.",
            "Recommendation systems analyze user behavior to suggest relevant items.",
            "Deep learning uses multi-layer neural networks for complex pattern recognition.",
            "Data preprocessing cleans and prepares data for better model performance.",
            "AI ethics involves fairness, transparency, privacy, and avoiding harm.",
            "LLMs use transformer architecture to predict and generate text sequences."
        ]
    
    def run_comprehensive_optimization(self,
                                      evaluation_queries: List[str],
                                      ground_truths: List[str],
                                      optimization_method: str = 'bayesian',
                                      n_iterations: int = 25) -> Dict[str, Any]:
        """Run comprehensive optimization workflow"""
        
        print("🚀 Starting Comprehensive RAG Optimization")
        print("=" * 60)
        
        results = {}
        
        if optimization_method == 'bayesian':
            print("🧠 Running Bayesian Optimization...")
            self.bayesian_optimizer = RAGHyperparameterOptimizer(
                rag_pipeline=self.rag_pipeline,
                evaluation_queries=evaluation_queries,
                ground_truths=ground_truths
            )
            results['bayesian'] = self.bayesian_optimizer.optimize(n_iterations=n_iterations)
            
        elif optimization_method == 'grid':
            print("🔍 Running Grid Search...")
            results['grid'] = self.grid_optimizer.grid_search(
                evaluation_queries=evaluation_queries,
                ground_truths=ground_truths,
                reduced_space=True
            )
            
        elif optimization_method == 'both':
            print("🔄 Running Both Optimization Methods...")
            
            # Grid search first for baseline
            print("\n1️⃣ Grid Search (Baseline)")
            results['grid'] = self.grid_optimizer.grid_search(
                evaluation_queries=evaluation_queries,
                ground_truths=ground_truths,
                reduced_space=True
            )
            
            # Bayesian optimization
            print("\n2️⃣ Bayesian Optimization")
            self.bayesian_optimizer = RAGHyperparameterOptimizer(
                rag_pipeline=self.rag_pipeline,
                evaluation_queries=evaluation_queries,
                ground_truths=ground_truths
            )
            results['bayesian'] = self.bayesian_optimizer.optimize(n_iterations=n_iterations)
            
            # Compare results
            results['comparison'] = self._compare_optimization_methods(
                results['grid'], results['bayesian']
            )
        
        return results
    
    def _compare_optimization_methods(self, 
                                     grid_results: Dict, 
                                     bayesian_results: Dict) -> Dict[str, Any]:
        """Compare results from different optimization methods"""
        
        # Get best config from each method
        grid_best = grid_results['best_configurations'][0] if grid_results['best_configurations'] else None
        bayesian_best = bayesian_results['best_configurations'][0] if bayesian_results['best_configurations'] else None
        
        comparison = {
            'grid_search': {
                'best_score': grid_best['composite_score'] if grid_best else 0,
                'configurations_evaluated': grid_results.get('successful_evaluations', 0),
                'total_time_estimate': grid_results.get('grid_size', 0) * 2  # Rough estimate
            },
            'bayesian_optimization': {
                'best_hypervolume': bayesian_results.get('pareto_front', {}).get('hypervolume', 0),
                'pareto_solutions': len(bayesian_results.get('best_configurations', [])),
                'iterations': 25  # From n_iterations
            }
        }
        
        return comparison
    
    def deploy_best_configuration(self, 
                                  optimization_results: Dict[str, Any],
                                  method: str = 'bayesian') -> 'GAIARAG':
        """Deploy the best configuration found during optimization"""
        
        if method == 'bayesian' and 'bayesian' in optimization_results:
            best_config = optimization_results['bayesian']['best_configurations'][0]['config']
        elif method == 'grid' and 'grid' in optimization_results:
            best_config = optimization_results['grid']['best_configurations'][0]['config']
        else:
            raise ValueError(f"No results found for method: {method}")
        
        print(f"🚀 Deploying best {method} configuration:")
        print(f"   Model: {best_config.llm_model}")
        print(f"   Chunk Size: {best_config.chunk_size}")
        print(f"   K: {best_config.k}")
        print(f"   Temperature: {best_config.temperature}")
        print(f"   Reranker: {best_config.use_reranker}")
        
        # Setup and return optimized RAG
        optimized_rag = self.rag_pipeline.setup_rag(best_config)
        return optimized_rag

# ============== COMPLETE USAGE EXAMPLE ==============

class RAGEvaluator:
    """
    Main RAG evaluation class that merges and organizes different types of metrics
    """
    llm: RAGLLM = None
    def __init__(self, 
                 llm: RAGLLM=None,
                 metrics: Optional[Dict[str, BaseMetric]] = None, # TODO:: recheck this again 
                 dataset: Optional[RAGDataset] = None
                 ):
        
        
        self.metrics = {
            MetricCategory.PERFORMANCE: {},
            MetricCategory.SAFETY: {},
            MetricCategory.RETRIEVAL: {},
            MetricCategory.GENERATION: {},
        }
        self._initialize_metrics(metrics)
        self.llm = llm

    def _initialize_metrics(self, metrics: Optional[Dict[str, BaseMetric]] = None):
        """Initialize all available metrics"""
        # TODO:: load from metrics
        # Performance Metrics (System-level)
        self.metrics[MetricCategory.PERFORMANCE].update({
            'cost': CostMetric(),
            'latency': LatencyMetric()
        })
        
        # Safety Metrics (System-level)
        self.metrics[MetricCategory.SAFETY].update({
            'safety': SafetyMetric(),
            'alignment': AlignmentMetric()
        })
        
        # Retrieval Metrics (Component-level)
        self.metrics[MetricCategory.RETRIEVAL].update({
            # RAGAS Context Metrics
            'context_precision': ContextPrecision(),
            'context_entities_recall': ContextEntitiesRecall(),
            'noise_sensitivity': NoiseSensitivity(),
            
            # AutoRAG Retrieval Metrics
            'precision': RetrievalPrecision(),
            'recall': RetrievalRecall(),
            'f1_score': RetrievalF1(),
            'mrr': MRR(),
            'map': MAP(),
            'ndcg': NDCG(),
            
            # Token-level Retrieval Metrics
            'token_precision': TokenPrecision(),
            'token_recall': TokenRecall(),
            'token_f1': TokenF1()
        })
        
        # Generation Metrics (Content-level)
        self.metrics[MetricCategory.GENERATION].update({
            # RAGAS Generation Metrics
            'response_relevancy': ResponseRelevancy(),
            'faithfulness': Faithfulness(),
            'faithfulness_hhem': FaithfulnessHHEM(),
            
            # AutoRAG Generation Metrics
            'bleu': BleuScore(),
            'rouge': RougeScore(),
            'meteor': MeteorScore(),
            'sem_score': SemanticScore(),
            'g_eval': GEval(),
            'bert_score': BertScore()
        })
    

    @staticmethod
    def evaluate_metric(metric: BaseMetric, **kwargs) -> MetricResult:
        # TODO:: how to pass / use the llm here to evaluate the metric
        return metric.evaluate(**kwargs)
        
    # ============== EVALUATION METHODS ==============
    
    def evaluate_all(self, query: str, retrieved_contexts: List[str], 
                    generated_response: str, ground_truth: Optional[str] = None,
                    **kwargs) -> Dict[str, MetricResult]:
        """Evaluate all metrics and return comprehensive results"""
        results = {}
        
        for category in MetricCategory:
            category_results = self.evaluate_category(
                category, query, retrieved_contexts, generated_response, ground_truth, **kwargs
            )
            results.update(category_results)
        
        return results
    
    def evaluate_category(self, category: MetricCategory, query: str, 
                         retrieved_contexts: List[str], generated_response: str,
                         ground_truth: Optional[str] = None, **kwargs) -> Dict[str, MetricResult]:
        """Evaluate all metrics in a specific category"""
        results = {}
        
        for metric_name, metric in self.metrics[category].items():
            try:
                result = metric.evaluate(
                    query=query,
                    contexts=retrieved_contexts,
                    response=generated_response,
                    ground_truth=ground_truth,
                    **kwargs
                )
                results[metric_name] = result
            except Exception as e:
                results[metric_name] = MetricResult(
                    name=metric_name,
                    value=0.0,
                    category=category,
                    scope=metric.scope,
                    error=str(e)
                )
        
        return results
    
    def evaluate_specific_metrics(self, metric_names: List[str], **kwargs) -> Dict[str, MetricResult]:
        """Evaluate only specific metrics by name"""
        results = {}
        
        for category_metrics in self.metrics.values():
            for metric_name, metric in category_metrics.items():
                if metric_name in metric_names:
                    results[metric_name] = metric.evaluate(**kwargs)
        
        return results
    
    # ============== COST EVALUATION (DETAILED) ==============
    
    def evaluate_cost_detailed(self, **kwargs) -> Dict[str, float]:
        """
        Detailed cost evaluation - this is where you'd handle the complexity
        Cost can include: API costs, compute costs, storage costs, etc.
        """
        cost_breakdown = {
            'api_cost': self._calculate_api_cost(**kwargs),
            'compute_cost': self._calculate_compute_cost(**kwargs),
            'storage_cost': self._calculate_storage_cost(**kwargs),
            'total_cost': 0.0
        }
        cost_breakdown['total_cost'] = sum(cost_breakdown.values())
        return cost_breakdown
    
    def _calculate_api_cost(self, **kwargs) -> float:
        """Calculate API costs (LLM calls, embedding calls, etc.)"""
        # Implementation would calculate based on:
        # - Number of tokens used
        # - Model pricing
        # - API provider rates
        pass
    
    def _calculate_compute_cost(self, **kwargs) -> float:
        """Calculate compute costs (processing time, resources)"""
        # Implementation would calculate based on:
        # - Processing time
        # - Resource utilization
        # - Infrastructure costs
        pass
    
    def _calculate_storage_cost(self, **kwargs) -> float:
        """Calculate storage costs (vector DB, caching, etc.)"""
        # Implementation would calculate based on:
        # - Vector database storage
        # - Cache storage
        # - Document storage
        pass
    
    # ============== AGGREGATION METHODS ==============
    
    def aggregate_results(self, results: Dict[str, MetricResult], 
                         aggregation_method: str = 'weighted_average') -> float:
        """Aggregate multiple metric results into a single score"""
        if aggregation_method == 'weighted_average':
            return self._weighted_average(results)
        elif aggregation_method == 'harmonic_mean':
            return self._harmonic_mean(results)
        elif aggregation_method == 'geometric_mean':
            return self._geometric_mean(results)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _weighted_average(self, results: Dict[str, MetricResult]) -> float:
        """Calculate weighted average of results"""
        # Implementation would apply different weights based on metric importance
        pass
    
    def _harmonic_mean(self, results: Dict[str, MetricResult]) -> float:
        """Calculate harmonic mean of results"""
        pass
    
    def _geometric_mean(self, results: Dict[str, MetricResult]) -> float:
        """Calculate geometric mean of results"""
        pass
    
    # ============== UTILITY METHODS ==============
    
    def get_metrics_by_scope(self, scope: MetricScope) -> Dict[str, BaseMetric]:
        """Get all metrics filtered by scope"""
        filtered_metrics = {}
        for category_metrics in self.metrics.values():
            for name, metric in category_metrics.items():
                if metric.scope == scope:
                    filtered_metrics[name] = metric
        return filtered_metrics
    
    def get_available_metrics(self) -> Dict[str, Dict[str, str]]:
        """Get information about all available metrics"""
        metrics_info = {}
        for category, category_metrics in self.metrics.items():
            metrics_info[category.value] = {
                name: f"{metric.scope.value} - {metric.__class__.__name__}"
                for name, metric in category_metrics.items()
            }
        return metrics_info


def complete_optimization_example():
    """
    Complete example showing how to use the entire optimization pipeline
    """
    
    # Step 1: Prepare your components (replace with your actual components)
    embeddings_dict = {
        'openai': None,  # YourOpenAIEmbeddings(),
        'huggingface': None,  # YourHuggingFaceEmbeddings()
    }
    
    vector_stores_dict = {
        'faiss': None,  # YourFAISSVectorStore(),
        'chroma': None,  # YourChromaVectorStore()
    }
    
    llms_dict = {
        'openai_gpt-3.5-turbo': None,  # YourOpenAILLM(model="gpt-3.5-turbo"),
        'openai_gpt-4': None,  # YourOpenAILLM(model="gpt-4"),
        'anthropic_claude-3-sonnet': None,  # YourAnthropicLLM()
    }
    
    rerankers_dict = {
        'cross_encoder': None,  # YourCrossEncoderReranker(),
        'colbert': None,  # YourColBERTReranker()
    }
    
    # Step 2: Initialize optimization workflow
    workflow = RAGOptimizationWorkflow(
        embeddings_dict=embeddings_dict,
        vector_stores_dict=vector_stores_dict,
        llms_dict=llms_dict,
        rerankers_dict=rerankers_dict
    )
    
    # Step 3: Prepare evaluation data
    evaluation_queries, ground_truths = workflow.prepare_evaluation_dataset()
    print(f"📊 Prepared {len(evaluation_queries)} evaluation queries")
    
    # Step 4: Run optimization
    optimization_results = workflow.run_comprehensive_optimization(
        evaluation_queries=evaluation_queries,
        ground_truths=ground_truths,
        optimization_method='both',  # 'bayesian', 'grid', or 'both'
        n_iterations=20
    )
    
    # Step 5: Analyze results
    if 'comparison' in optimization_results:
        print("\n📈 Optimization Method Comparison:")
        comparison = optimization_results['comparison']
        print(f"Grid Search: {comparison['grid_search']}")
        print(f"Bayesian: {comparison['bayesian_optimization']}")
    
    # Step 6: Deploy best configuration
    best_rag = workflow.deploy_best_configuration(
        optimization_results=optimization_results,
        method='bayesian'
    )
    
    # Step 7: Test deployed system
    test_query = "What is the future of artificial intelligence?"
    answer = best_rag.get_answer(test_query)
    print(f"\n🤖 Test Query: {test_query}")
    print(f"📝 Answer: {answer}")
    
    return optimization_results, best_rag


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example usage
    query = "What is the capital of France?"
    contexts = ["France is a country in Europe.", "Paris is the capital of France."]
    response = "The capital of France is Paris."
    ground_truth = "Paris"
    
    # Evaluate all metrics
    results = evaluator.evaluate_all(query, contexts, response, ground_truth)
    
    # Evaluate specific category
    retrieval_results = evaluator.evaluate_category(
        MetricCategory.RETRIEVAL, query, contexts, response, ground_truth
    )
    
    # Evaluate specific metrics
    specific_results = evaluator.evaluate_specific_metrics(
        ['cost', 'latency', 'faithfulness'], 
        query=query, contexts=contexts, response=response
    )
    
    # Get detailed cost breakdown
    cost_details = evaluator.evaluate_cost_detailed(
        api_calls=10, processing_time=2.5, storage_mb=100
    )