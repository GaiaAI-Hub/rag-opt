from typing_extensions import Any, Optional, Doc, Annotated, TypeAlias, Literal, Callable, TypeVar
from rag_opt import init_vectorstore, init_embeddings, init_reranker, init_chat_model
from concurrent.futures import Future, as_completed, Executor
from rag_opt.dataset import TrainDataset, EvaluationDataset
from rag_opt.rag import RAGWorkflow, BaseReranker
from langchain.chat_models.base import BaseChatModel
from langchain.schema.embeddings import Embeddings
from rag_opt.search_space import RAGSearchSpace
from rag_opt._utils import get_shared_executor
from rag_opt._sampler import SamplerType
from rag_opt._config import RAGConfig
from langchain.schema import Document
from abc import ABC, abstractmethod
from threading import Lock
from loguru import logger
import torch
from dataclasses import dataclass

ComponentType: TypeAlias = Literal["llms", "embeddings", "vector_stores", "rerankers"]
T = TypeVar("T")


@dataclass
class ComponentConfig:
    """Configuration for loading a specific component."""
    provider: str
    models: list[str]
    api_key: Optional[str] = None


class AbstractRAGPipelineManager(ABC):
    """
    Abstract base class for RAG Pipeline Management.
    
    Defines the contract for managing RAG components (LLMs, embeddings, vector stores, rerankers)
    and provides extension points for custom optimization strategies.
    """

    def __init__(
        self,
        search_space: Annotated[RAGSearchSpace, Doc("The RAG search space to be optimized")],
        *,
        max_workers: Annotated[int, Doc("Maximum workers for parallel operations")] = 5,
        eager_load: Annotated[bool, Doc("Load all components immediately")] = False,
        verbose: Annotated[bool, Doc("Enable verbose logging")] = False,
        executor: Annotated[Optional[Executor], Doc("Thread pool executor")] = None,
        **kwargs
    ):
        self._search_space = search_space
        self.max_workers = max_workers
        self.eager_load = eager_load
        self._verbose = verbose
        self.executor = executor or get_shared_executor(max_workers)
        self._init_kwargs = kwargs


    @property
    def search_space(self) -> RAGSearchSpace:
        """Get the search space."""
        return self._search_space

    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self._verbose

    @abstractmethod
    def get_llm(self, model: str, provider: str, api_key: Optional[str] = None) -> BaseChatModel:
        """Get or create an LLM instance."""
        pass

    @abstractmethod
    def get_embeddings(self, provider: str, model: str, api_key: Optional[str] = None) -> Embeddings:
        """Get or create an embeddings instance."""
        pass

    @abstractmethod
    def get_reranker(self, model: str, provider: str, api_key: Optional[str] = None) -> Optional[Any]:
        """Get or create a reranker instance."""
        pass

    @abstractmethod
    def get_vector_store(
        self,
        provider: str,
        embeddings: Embeddings,
        documents: list[Document],
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        initialize: bool = False
    ) -> Any:
        """Get or create a vector store instance."""
        pass

    @abstractmethod
    def create_rag_instance(
        self,
        config: RAGConfig,
        documents: Optional[list[Document]] = None,
        retrieval_config: Optional[dict] = None,
        initialize: bool = False
    ) -> RAGWorkflow:
        """Create a RAGWorkflow instance from configuration."""
        pass

    def initiate_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Get an LLM instance for evaluation. Can be overridden."""
        raise NotImplementedError("Subclass must implement initiate_llm")

    def initiate_embedding(self, model_name: Optional[str] = None) -> Embeddings:
        """Get an embedding instance for evaluation. Can be overridden."""
        raise NotImplementedError("Subclass must implement initiate_embedding")

    @classmethod
    def from_search_space(
        cls,
        search_space: RAGSearchSpace,
        max_workers: int = 4,
        eager_load: bool = False,
        **kwargs
    ) -> "AbstractRAGPipelineManager":
        """Factory method to create manager from search space."""
        return cls(
            search_space=search_space,
            max_workers=max_workers,
            eager_load=eager_load,
            **kwargs
        )

class RAGPipelineManager(AbstractRAGPipelineManager):
    """
    Manages loading and caching of RAG components (LLMs, embeddings, vector stores, rerankers).
    
    Optimizes RAGWorkflow instantiation during Bayesian Optimization by caching components
    and supporting parallel initialization.
    """

    def __init__(
        self,
        search_space: Annotated[RAGSearchSpace, Doc("the RAG search space (hyperparameters) to be optimized")],
        *,
        max_workers: Annotated[int, Doc("Maximum workers for parallel component loading")] = 5,
        eager_load: Annotated[bool, Doc("Load all search space components immediately")] = False,
        verbose: Annotated[bool, Doc("Enable verbose logging")] = False,
        executor: Annotated[Optional[Executor], Doc("The thread pool executor for batch evaluation")] = None,
        provider_overrides: Annotated[Optional[dict[str, str]], Doc("Override provider mappings")] = None,
        **kwargs
    ):
        super().__init__(search_space, max_workers=max_workers, eager_load=eager_load, 
                         verbose=verbose, executor=executor, **kwargs)
        
        self._registry: dict[str, Any] = {}
        self._lock = Lock()
        self._provider_overrides = provider_overrides or {}
        
        # Component type to factory function mapping
        self._component_factories = {
            "llms": init_chat_model,
            "embeddings": init_embeddings,
            "rerankers": init_reranker,
        }
        
        if eager_load:
            logger.debug("RAGPipelineManager: Loading all RAG Components")
            self._load_all_components()

    def _build_cache_key(self, component_type: ComponentType, **kwargs) -> str:
        """Generate unique cache key for component type and parameters."""
        if component_type not in {"llms", "embeddings", "vector_stores", "rerankers"}:
            raise ValueError(f"Invalid component type: {component_type}")
        
        parts = [component_type]
        for key, value in sorted(kwargs.items()):
            parts.append(f"{key}={value}")
        return "|".join(parts)

    def _get_or_create_component(
        self,
        cache_key: str,
        factory_func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Thread-safe component creation with caching."""
        with self._lock:
            if cache_key not in self._registry:
                self._registry[cache_key] = factory_func(*args, **kwargs)
            return self._registry[cache_key]

    def _normalize_provider(self, provider: str, component_type: ComponentType) -> str:
        """Apply provider overrides and normalizations."""
        # Apply custom overrides first
        if provider in self._provider_overrides:
            return self._provider_overrides[provider]
        
        # Handle sentence transformer compatibility
        if component_type == "embeddings" and provider == "sentence-transformer":
            logger.warning("Using HuggingFace provider for sentence transformer models")
            return "huggingface"
        
        return provider

    def get_llm(self, model: str, provider: str, api_key: Optional[str] = None) -> BaseChatModel:
        """Get or create LLM instance."""
        provider = self._normalize_provider(provider, "llms")
        cache_key = self._build_cache_key("llms", model=model, provider=provider)
        return self._get_or_create_component(
            cache_key,
            init_chat_model,
            model=model,
            model_provider=provider,
            api_key=api_key
        )

    def get_embeddings(self, provider: str, model: str, api_key: Optional[str] = None) -> Embeddings:
        """Get or create embeddings instance."""
        provider = self._normalize_provider(provider, "embeddings")
        cache_key = self._build_cache_key("embeddings", provider=provider, model=model)
        return self._get_or_create_component(
            cache_key,
            init_embeddings,
            model_provider=provider,
            model=model,
            api_key=api_key
        )

    def get_reranker(self, model: str, provider: str, api_key: Optional[str] = None) -> Optional[BaseReranker]:
        """Get or create reranker instance."""
        if not provider or not model:
            return None
        
        provider = self._normalize_provider(provider, "rerankers")
        cache_key = self._build_cache_key("rerankers", model=model, provider=provider)
        return self._get_or_create_component(
            cache_key,
            init_reranker,
            model=model,
            provider=provider,
            api_key=api_key
        )

    def get_vector_store(
        self,
        provider: str,
        embeddings: Embeddings,
        documents: list[Document],
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        initialize: bool = False
    ):
        """Get or create vector store instance."""
        cache_key = self._build_cache_key("vector_stores", provider=provider, model=embeddings.__class__.__name__)
        return self._get_or_create_component(
            cache_key,
            init_vectorstore,
            provider,
            embeddings,
            documents=documents,
            index_name=index_name,
            api_key=api_key,
            initialize=initialize
        )

    def _get_component_by_name(self, component_type: ComponentType, model_name: str) -> Optional[Any]:
        """Generic method to retrieve cached component by model name."""
        if not model_name:
            return None
        
        for key in self._registry.keys():
            if model_name in key and component_type in key:
                return self._registry[key]
        return None

    def get_llm_by_model_name(self, model_name: str) -> Optional[BaseChatModel]:
        """Get cached LLM by model name."""
        return self._get_component_by_name("llms", model_name)

    def get_embedding_by_model_name(self, model_name: str) -> Optional[Embeddings]:
        """Get cached embedding by model name."""
        return self._get_component_by_name("embeddings", model_name)

    def _get_first_available_component(
        self,
        component_type: ComponentType,
        getter_func: Callable[[str, str, Optional[str]], Any]
    ) -> Optional[Any]:
        """Generic method to get first available component from search space."""
        # Map component type to search space attribute
        space_attr_map = {
            "llms": "llm",
            "embeddings": "embedding",
            "rerankers": "reranker"
        }
        
        space_attr = getattr(self._search_space, space_attr_map[component_type], None)
        if not space_attr or not space_attr.choices:
            logger.error(f"No {component_type} found in search space")
            raise ValueError(f"No {component_type} found in search space")
        
        for config in space_attr.choices.values():
            for model in config.models:
                if model and config.provider:
                    return getter_func(model, config.provider, config.api_key)
        
        return None

    def initiate_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Helper method to get an LLM to be used in evaluation process."""
        llm = self.get_llm_by_model_name(model_name) if model_name else None
        if llm:
            return llm
        
        return self._get_first_available_component(
            "llms",
            lambda m, p, k: self.get_llm(model=m, provider=p, api_key=k)
        )

    def initiate_embedding(self, model_name: Optional[str] = None) -> Embeddings:
        """Helper method to get an embedding to be used in evaluation process."""
        embedding = self.get_embedding_by_model_name(model_name) if model_name else None
        if embedding:
            return embedding
        
        return self._get_first_available_component(
            "embeddings",
            lambda m, p, k: self.get_embeddings(provider=p, model=m, api_key=k)
        )

    def _load_component_type(
        self,
        component_type: ComponentType,
        parallel: bool = True
    ) -> None:
        """Load all components of a specific type."""
        space_attr_map = {
            "llms": ("llm", self.get_llm, ["model", "provider", "api_key"]),
            "embeddings": ("embedding", self.get_embeddings, ["provider", "model", "api_key"]),
            "rerankers": ("reranker", self.get_reranker, ["model", "provider", "api_key"])
        }
        
        if component_type not in space_attr_map:
            return
        
        space_attr_name, getter_func, _ = space_attr_map[component_type]
        space_attr = getattr(self._search_space, space_attr_name, None)
        
        if not space_attr or not space_attr.choices:
            return
        
        for config in space_attr.choices.values():
            for model in config.models:
                kwargs = {
                    "model": model,
                    "provider": config.provider,
                    "api_key": config.api_key
                }
                
                if parallel:
                    self.executor.submit(getter_func, **kwargs)
                else:
                    getter_func(**kwargs)

    def _load_all_components(self, parallel: bool = True) -> None:
        """Initialize all components from search space."""
        for component_type in ["llms", "embeddings", "rerankers"]:
            self._load_component_type(component_type, parallel)

    def generate_initial_data(
        self,
        train_data: TrainDataset,
        n_samples: int = 20,
        sampler_type: SamplerType = SamplerType.SOBOL,
        **kwargs
    ) -> tuple[list[RAGConfig], list[EvaluationDataset]]:
        """Generate initial data from a sampled search space config."""
        rag_configs = self._search_space.sample(n_samples=n_samples, sampler_type=sampler_type)
        
        documents = train_data.to_langchain_docs()
        configs: list[RAGConfig] = []
        datasets: list[EvaluationDataset] = []
        future_map: dict[Future[EvaluationDataset], RAGConfig] = {}


        # for rag_config in rag_configs:
        #     rag = self.create_rag_instance(rag_config, documents=documents, initialize=True, **kwargs)
        #     dataset = rag.get_batch_answers(dataset=train_data, **rag_config.to_dict())
        #     if len(dataset.items) > 0:
        #         datasets.append(dataset)
        #         configs.append(rag_config)
        #     else:
        #         logger.error(f"No dataset returned for config {rag_config}")
        for rag_config in rag_configs:
            rag = self.create_rag_instance(rag_config, documents=documents, initialize=True, **kwargs)
            future = self.executor.submit(
                rag.get_batch_answers,
                dataset=train_data,
                **rag_config.to_dict()
            )
            future_map[future] = rag_config
        
        for future in as_completed(future_map):
            rag_config = future_map[future]
            try:
                datasets.append(future.result())
                configs.append(rag_config)
            except Exception as e:
                logger.error(f"Error processing sample {rag_config}: {e}")
        
        return configs, datasets

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
            api_key=config.llm.api_key
        )
        
        embeddings = self.get_embeddings(
            provider=config.embedding.provider,
            model=config.embedding.model,
            api_key=config.embedding.api_key
        )
        
        reranker = None
        if config.reranker:
            reranker = self.get_reranker(
                model=config.reranker.model,
                provider=config.reranker.provider,
                api_key=config.reranker.api_key
            )
        
        vector_store = self.get_vector_store(
            provider=config.vector_store.provider,
            embeddings=embeddings,
            documents=documents or [],
            index_name=config.vector_store.index_name,
            api_key=config.vector_store.api_key,
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
            corpus_documents=documents
        )

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

    def generate_evaluation_data(self, config: RAGConfig, train_data: TrainDataset, **kwargs) -> EvaluationDataset:
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