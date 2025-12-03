from typing import Optional, Any, Callable
from typing_extensions import Annotated, Doc, TypeAlias, Literal
from concurrent.futures import Future, as_completed, Executor
from rag_opt.dataset import TrainDataset, EvaluationDataset
from rag_opt.search_space import RAGSearchSpace
from rag_opt._utils import get_shared_executor
from rag_opt._sampler import SamplerType
from rag_opt._config import RAGConfig
from rag_opt.rag import RAGWorkflow
from langchain.chat_models.base import BaseChatModel
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from threading import Lock
from loguru import logger
import torch
import os

ComponentType: TypeAlias = Literal["llms", "embeddings", "vector_stores", "rerankers"]


class RAGPipelineManager:
    """
    RAG Pipeline Manager with Gateway API support.
    """

    def __init__(
        self,
        search_space: Annotated[RAGSearchSpace, Doc("RAG search space to optimize")],
        *,
        max_workers: Annotated[int, Doc("Maximum workers for parallel operations")] = 5,
        eager_load: Annotated[bool, Doc("Load all components immediately")] = False,
        verbose: Annotated[bool, Doc("Enable verbose logging")] = False,
        executor: Annotated[Optional[Executor], Doc("Thread pool executor")] = None,
        api_key: Annotated[Optional[str], Doc("Universal API key (for gateway or providers)")] = None,
        **kwargs
    ):
        self._search_space = search_space
        self.max_workers = max_workers
        self.eager_load = eager_load
        self._verbose = verbose
        self.executor = executor or get_shared_executor(max_workers)
        
        # Universal API key handling
        self.api_key = api_key or os.getenv('AI_GATEWAY_API_KEY')
        
        # Component registry and lock
        self._registry: dict[str, Any] = {}
        self._lock = Lock()
        self._init_kwargs = kwargs
        
        # Load components if eager loading is enabled
        if eager_load:
            logger.debug("RAGPipelineManager: Eagerly loading all components")
            self._load_all_components_parallel()

    @property
    def search_space(self) -> RAGSearchSpace:
        """Get the search space."""
        return self._search_space

    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self._verbose

    def _build_cache_key(self, component_type: ComponentType, **kwargs) -> str:
        """Generate unique cache key for component type and parameters."""
        if component_type not in {"llms", "embeddings", "vector_stores", "rerankers"}:
            raise ValueError(f"Invalid component type: {component_type}")
        
        parts = [component_type]
        for key, value in sorted(kwargs.items()):
            if value is not None:
                parts.append(f"{key}={value}")
        return "|".join(parts)

    def _get_or_create_component(
        self,
        cache_key: str,
        factory_func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """Thread-safe component creation with caching."""
        # Fast path: check without lock
        if cache_key in self._registry:
            return self._registry[cache_key]
        
        with self._lock:
            if cache_key not in self._registry:
                try:
                    self._registry[cache_key] = factory_func(*args, **kwargs)
                    if self._verbose:
                        logger.debug(f"Created and cached: {cache_key}")
                except Exception as e:
                    logger.error(f"Failed to create component {cache_key}: {e}")
                    raise
            return self._registry[cache_key]

    def _is_gateway_model(self, model: str) -> bool:
        """Check if model string is in gateway format (provider/model)."""
        return '/' in model

    def get_llm(self, model: str, provider: str, api_key: Optional[str] = None) -> BaseChatModel:
        """Get or create LLM instance - supports both gateway and standard formats."""
        # Use provided API key, fallback to instance key
        effective_api_key = api_key or self.api_key
        
        # Gateway format: provider/model
        if self._is_gateway_model(model) or provider == 'gateway':
            gateway_model = model if '/' in model else f"{provider}/{model}"
            cache_key = self._build_cache_key("llms", model=gateway_model)
            
            from rag_opt.llm import init_chat_model
            return self._get_or_create_component(
                cache_key,
                init_chat_model,
                model=gateway_model,
                api_key=effective_api_key
            )
        else:
            # Standard provider
            cache_key = self._build_cache_key("llms", model=model, provider=provider)
            
            from rag_opt import init_chat_model
            return self._get_or_create_component(
                cache_key,
                init_chat_model,
                model=model,
                model_provider=provider,
                api_key=effective_api_key
            )

    def get_embeddings(self, provider: str, model: str, api_key: Optional[str] = None) -> Embeddings:
        """Get or create embeddings instance - supports both gateway and standard formats."""
        # Use provided API key, fallback to instance key
        effective_api_key = api_key or self.api_key

        # Gateway format: provider/model
        if self._is_gateway_model(model) or provider == 'gateway':
            gateway_model = model if '/' in model else f"{provider}/{model}"
            cache_key = self._build_cache_key("embeddings", model=gateway_model)
            
            from rag_opt.llm import init_embeddings
            return self._get_or_create_component(
                cache_key,
                init_embeddings,
                model=gateway_model,
                api_key=effective_api_key
            )
        else:
            # Standard provider
            cache_key = self._build_cache_key("embeddings", provider=provider, model=model)
            
            from rag_opt import init_embeddings
            return self._get_or_create_component(
                cache_key,
                init_embeddings,
                model_provider=provider,
                model=model,
                api_key=effective_api_key
            )

    def get_reranker(self, model: str, provider: str, api_key: Optional[str] = None) -> Optional[Any]:
        """Get or create reranker instance."""
        if not provider or not model:
            return None
        
        effective_api_key = api_key or self.api_key
        cache_key = self._build_cache_key("rerankers", model=model, provider=provider)
        
        from rag_opt import init_reranker
        return self._get_or_create_component(
            cache_key,
            init_reranker,
            model=model,
            provider=provider,
            api_key=effective_api_key
        )

    def get_vector_store(
        self,
        provider: str,
        embeddings: Embeddings,
        documents: list[Document],
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        initialize: bool = False
    ) -> Any:
        """Get or create vector store instance with smart caching."""
        from rag_opt import init_vectorstore
        
        effective_api_key = api_key or self.api_key
        
        # Cache based on provider and embedding type only
        cache_key = self._build_cache_key(
            "vector_stores",
            provider=provider,
            embedding_class=embeddings.__class__.__name__
        )
        
        return self._get_or_create_component(
            cache_key,
            init_vectorstore,
            provider,
            embeddings,
            documents=documents,
            index_name=index_name,
            api_key=effective_api_key,
            initialize=initialize
        )

    def _load_component_batch(
        self,
        component_type: ComponentType,
        configs: list[dict],
        getter_func: Callable
    ) -> None:
        """Load a batch of components in parallel."""
        futures = []
        
        for config in configs:
            future = self.executor.submit(getter_func, **config)
            futures.append(future)
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Component load failed: {e}")

    def _load_all_components_parallel(self) -> None:
        """Load all components from search space in parallel."""
        load_tasks = []

        # Load LLMs
        llm_configs = []
        if self._search_space.llm and self._search_space.llm.choices:
            for provider_name, config in self._search_space.llm.choices.items():
                # Handle gateway format (list of model strings)
                if provider_name == 'gateway' and isinstance(config, list):
                    for model in config:
                        llm_configs.append({
                            "model": model,
                            "provider": provider_name,
                            "api_key": self.api_key
                        })
                # Handle standard provider format (config object)
                elif hasattr(config, 'models'):
                    for model in config.models:
                        llm_configs.append({
                            "model": model,
                            "provider": provider_name,
                            "api_key": config.api_key or self.api_key
                        })
        if llm_configs:
            load_tasks.append(("llms", llm_configs, self.get_llm))

        # Load Embeddings
        emb_configs = []
        if self._search_space.embedding and self._search_space.embedding.choices:
            for provider_name, config in self._search_space.embedding.choices.items():
                # Handle gateway format (list of model strings)
                if provider_name == 'gateway' and isinstance(config, list):
                    for model in config:
                        emb_configs.append({
                            "provider": provider_name,
                            "model": model,
                            "api_key": self.api_key
                        })
                # Handle standard provider format (config object)
                elif hasattr(config, 'models'):
                    for model in config.models:
                        emb_configs.append({
                            "provider": provider_name,
                            "model": model,
                            "api_key": config.api_key or self.api_key
                        })
        if emb_configs:
            load_tasks.append(("embeddings", emb_configs, self.get_embeddings))

        # Load Rerankers
        reranker_configs = []
        if self._search_space.reranker and self._search_space.reranker.choices:
            for provider_name, config in self._search_space.reranker.choices.items():
                # Handle gateway format (list of model strings)
                if provider_name == 'gateway' and isinstance(config, list):
                    for model in config:
                        if model is not None:
                            reranker_configs.append({
                                "model": model,
                                "provider": provider_name,
                                "api_key": self.api_key
                            })
                # Handle standard provider format (config object)
                elif hasattr(config, 'models'):
                    for model in config.models:
                        if model is not None: 
                            reranker_configs.append({
                                "model": model,
                                "provider": provider_name,
                                "api_key": config.api_key or self.api_key
                            })
        if reranker_configs:
            load_tasks.append(("rerankers", reranker_configs, self.get_reranker))

        # Execute loading
        for component_type, configs, getter_func in load_tasks:
            self._load_component_batch(component_type, configs, getter_func)

    def initiate_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Get LLM for evaluation - supports gateway format (provider/model)."""
        # Handle gateway format
        if model_name and '/' in model_name:
            provider, model = model_name.split('/', 1)
            return self.get_llm(model=model, provider=provider, api_key=self.api_key)
        
        # Try to get cached LLM by name
        if model_name:
            for key, component in self._registry.items():
                if "llms" in key and model_name in key:
                    return component
        
        # Get first available from search space
        if hasattr(self._search_space, 'llm') and self._search_space.llm.choices:
            for provider_name, config in self._search_space.llm.choices.items():
                if config.models:
                    return self.get_llm(
                        model=config.models[0],
                        provider=provider_name,
                        api_key=config.api_key or self.api_key
                    )
        
        raise ValueError("No LLMs available in search space")

    def initiate_embedding(self, model_name: Optional[str] = None) -> Embeddings:
        """Get embedding for evaluation - supports gateway format (provider/model)."""
        # Handle gateway format
        if model_name and '/' in model_name:
            provider, model = model_name.split('/', 1)
            return self.get_embeddings(provider=provider, model=model, api_key=self.api_key)
        
        # Try to get cached embedding by name
        if model_name:
            for key, component in self._registry.items():
                if "embeddings" in key and model_name in key:
                    return component
        
        # Get first available from search space
        if hasattr(self._search_space, 'embedding') and self._search_space.embedding.choices:
            for provider_name, config in self._search_space.embedding.choices.items():
                if config.models:
                    return self.get_embeddings(
                        provider=provider_name,
                        model=config.models[0],
                        api_key=config.api_key or self.api_key
                    )
        
        raise ValueError("No embeddings available in search space")

    def create_rag_instance(
        self,
        config: RAGConfig,
        documents: Optional[list[Document]] = None,
        retrieval_config: Optional[dict] = None,
        initialize: bool = False
    ) -> RAGWorkflow:
        """Create RAGWorkflow instance from configuration using cached components."""
        llm = self.get_llm(
            model=config.llm.model,
            provider=config.llm.provider,
            api_key=config.llm.api_key or self.api_key
        )
        
        embeddings = self.get_embeddings(
            provider=config.embedding.provider,
            model=config.embedding.model,
            api_key=config.embedding.api_key or self.api_key
        )
        
        reranker = None
        if config.reranker:
            reranker = self.get_reranker(
                model=config.reranker.model,
                provider=config.reranker.provider,
                api_key=config.reranker.api_key or self.api_key
            )
        
        vector_store = self.get_vector_store(
            provider=config.vector_store.provider,
            embeddings=embeddings,
            documents=documents or [],
            index_name=config.vector_store.index_name,
            api_key=config.vector_store.api_key or self.api_key,
            initialize=initialize
        )
        
        return RAGWorkflow(
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            reranker=reranker,
            llm_provider_name=config.llm.provider,
            llm_model_name=config.llm.model,
            embedding_provider_name=config.embedding.provider,
            embedding_model_name=config.embedding.model,
            reranker_provider_name=config.reranker.provider if config.reranker else None,
            reranker_model_name=config.reranker.model if config.reranker else None,
            vector_store_provider_name=config.vector_store.provider,
            retrieval_config=retrieval_config or {"search_type": config.search_type, "k": config.k},
            corpus_documents=documents,
            max_workers=self.max_workers,
            executor=self.executor
        )

    def generate_initial_data(
        self,
        train_data: TrainDataset,
        n_samples: int = 20,
        sampler_type: SamplerType = SamplerType.SOBOL,
        **kwargs
    ) -> tuple[list[RAGConfig], list[EvaluationDataset]]:
        """Generate initial data with parallel execution."""
        rag_configs = self._search_space.sample(n_samples=n_samples, sampler_type=sampler_type)
        
        documents = train_data.to_langchain_docs()
        configs: list[RAGConfig] = []
        datasets: list[EvaluationDataset] = []
        future_map: dict[Future[EvaluationDataset], RAGConfig] = {}
        
        # Submit all tasks in parallel
        for rag_config in rag_configs:
            rag = self.create_rag_instance(rag_config, documents=documents, initialize=True, **kwargs)
            future = self.executor.submit(
                rag.get_batch_answers,
                dataset=train_data,
                **rag_config.to_dict()
            )
            future_map[future] = rag_config
        
        # Collect results as they complete
        for future in as_completed(future_map):
            rag_config = future_map[future]
            try:
                dataset = future.result(timeout=300)
                if len(dataset.items) > 0:
                    datasets.append(dataset)
                    configs.append(rag_config)
                else:
                    logger.warning(f"Empty dataset for config {rag_config}")
            except Exception as e:
                logger.error(f"Error processing config {rag_config}: {e}")
        
        return configs, datasets

    def create_rag_instance_by_sample(
        self,
        sampler_type: SamplerType = SamplerType.SOBOL,
        documents: Optional[list[Document]] = None,
        retrieval_config: Optional[dict] = None
    ) -> RAGWorkflow:
        """Create RAGWorkflow instance from a sampled search space config."""
        sample = self._search_space.sample(n_samples=1, sampler_type=sampler_type)
        if not sample:
            logger.error("No sample found in search space")
            raise ValueError("No sample found in search space")
        return self.create_rag_instance(sample[0], documents=documents, retrieval_config=retrieval_config)

    def generate_evaluation_data(
        self, 
        config: RAGConfig, 
        train_data: TrainDataset, 
        **kwargs
    ) -> EvaluationDataset:
        """Generate evaluation dataset from a sampled search space config."""
        documents = train_data.to_langchain_docs()
        rag = self.create_rag_instance(config, documents=documents, **kwargs)
        return rag.get_batch_answers(dataset=train_data, **kwargs)

    def clear_cache(self) -> None:
        """Clear all cached components."""
        with self._lock:
            self._registry.clear()
            logger.info("Component cache cleared")

    def sample(self, n_samples: int = 1, **kwargs) -> list[RAGConfig]:
        """Sample configurations from the search space."""
        return self._search_space.sample(n_samples, **kwargs)

    def get_problem_bounds(self) -> torch.Tensor:
        """Get parameter bounds from search space."""
        return self._search_space.get_parameter_bounds()

    def decode_sample_to_rag_config(self, sample: torch.Tensor) -> RAGConfig:
        """Decode a sample generated tensor to RAGConfig."""
        return self._search_space.decode_sample_to_rag_config(sample)

    def encode_rag_config_to_tensor(self, config: RAGConfig) -> torch.Tensor:
        """Encode a RAGConfig to a tensor."""
        return self._search_space.config_to_tensor(config)

    @classmethod
    def from_search_space(
        cls,
        search_space: RAGSearchSpace,
        max_workers: int = 4,
        eager_load: bool = False,
        **kwargs
    ) -> "RAGPipelineManager":
        """Factory method to create manager from search space."""
        return cls(
            search_space=search_space,
            max_workers=max_workers,
            eager_load=eager_load,
            **kwargs
        )