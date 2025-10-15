from typing_extensions import Any, Optional, Doc, Annotated, TypeAlias, Literal, Callable, TypeVar
from rag_opt import init_vectorstore, init_embeddings, init_reranker, init_chat_model
from concurrent.futures import  Future, as_completed, Executor
from rag_opt.dataset import TrainDataset, EvaluationDataset
from rag_opt.rag import RAGWorkflow, BaseReranker
from langchain.chat_models.base import BaseChatModel
from langchain.schema.embeddings import Embeddings
from rag_opt.search_space import RAGSearchSpace
from rag_opt._utils import get_shared_executor
from rag_opt._sampler import SamplerType
from rag_opt._config import RAGConfig
from langchain.schema import Document
from threading import Lock
from loguru import logger
import torch 
import uuid

ComponentType: TypeAlias = Literal["llms", "embeddings", "vector_stores", "rerankers"]
T = TypeVar("T")


class RAGPipelineManager:
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
        executor: Annotated[Optional[Executor], Doc("The thread pool executor for batch evaluation ")] = None,
        **kwargs

    ):
        self.eager_load = eager_load
        self._registry: dict[str, Any] = {}
        self._lock = Lock()
        self._verbose = verbose

        # required services 
        self._search_space = search_space

        if executor is None:
            self.executor = get_shared_executor(max_workers)

        # Load all RAG components (llms, embeddings,...) 
        # before starting optimization (preferable for small search space)
        if eager_load:
            logger.debug("RAGPipelineManager: Loading all RAG Component")
            self._load_all_components()
    
    @classmethod
    def from_search_space(
        cls, 
        search_space: RAGSearchSpace, 
        max_workers: int = 4, 
        eager_load: bool = False
    ) -> "RAGPipelineManager":
        """Create manager from search space configuration."""
        return cls(search_space, max_workers, eager_load)

    def get_problem_bounds(self) -> torch.Tensor:
        return self._search_space.get_parameter_bounds()
    
    def decode_sample_to_rag_config(self, sample: torch.Tensor) -> RAGConfig:
        """ Decode a sample generated tensor to RAGConfig """
        return self._search_space.decode_sample_to_rag_config(sample)

    def encode_rag_config_to_tensor(self, config: RAGConfig) -> torch.Tensor:
        """ Encode a RAGConfig to a tensor """
        return self._search_space.config_to_tensor(config)
    
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

    def get_llm(self, model: str, provider: str, api_key: Optional[str] = None) -> BaseChatModel:
        """Get or create LLM instance."""
        cache_key = self._build_cache_key("llms", model=model, provider=provider)
        return self._get_or_create_component(
            cache_key, 
            init_chat_model, 
            model=model, 
            model_provider=provider, 
            api_key=api_key
        )
    
    def get_llm_by_model_name(self,model_name:str) -> BaseChatModel:
        for key in self._registry.keys():
            if model_name in key and "llms" in key:
                return self._registry[key]
        return None
    
    def get_embedding_by_model_name(self, model_name: str) -> Embeddings:
        if not model_name:
            return None
        for key in self._registry.keys():
            if model_name in key and "embeddings" in key:
                return self._registry[key]
        return None


    def get_embeddings(self, provider: str, model: str, api_key: Optional[str] = None) -> Embeddings:
        """Get or create embeddings instance."""
        # Handle sentence transformer compatibility
        if provider == "sentence-transformer":
            provider = "huggingface"
            logger.warning("Using HuggingFace provider for sentence transformer models")
        
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
        api_key: Optional[str]= None,
        initialize: bool = False
    ): 
        """Get or create vector store instance."""
        cache_key = self._build_cache_key("vector_stores", provider=provider,  model=embeddings.__class__.__name__)

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
    
    def initiate_llm(self, model_name:str=None) -> BaseChatModel:
        """ Helper method to get a random llm to be used in evaluation process """
        llm = self.get_llm_by_model_name(model_name)
        if not llm:
            if not self._search_space.llm.choices:
                logger.error("No LLMs found in search space")
                raise ValueError("No LLMs found in search space")
            
            for llm_config in self._search_space.llm.choices.values():
                for model in llm_config.models:
                    if not (model and llm_config.provider):
                        continue
                    return self.get_llm(
                        model=model, 
                        provider=llm_config.provider, 
                        api_key=llm_config.api_key
                    )

    def initiate_embedding(self, model_name:str=None) -> Embeddings:
        """ Helper method to get a random embedding to be used in evaluation process """
        embedding = self.get_embedding_by_model_name(model_name)
        if not embedding:
            if not self._search_space.embedding.choices:
                logger.error("No Embeddings found in search space")
                raise ValueError("No Embeddings found in search space")
            
            for embedding_config in self._search_space.embedding.choices.values():
                for model in embedding_config.models:
                    if not (model and embedding_config.provider):
                        continue
                    return self.get_embeddings(
                        provider=embedding_config.provider,
                        model=model,
                        api_key=embedding_config.api_key
                    )
            

    def _load_all_components(self, parallel: bool = True) -> None:
        """Initialize all components from search space."""
        if parallel:
            self._load_components_parallel()
        else:
            self._load_components_sequential()
 
    def generate_initial_data(
        self,
        train_data: TrainDataset,
        n_samples: int = 20,
        sampler_type: SamplerType = SamplerType.SOBOL,
        **kwargs
    ) -> tuple[list[RAGConfig], list[EvaluationDataset]]:
        """Generate initial data from a sampled search space config. (RAGConfig) 
        
        Args:
            train_data (TrainDataset): Ground truth dataset to use (contains list of queries).
            n_samples (int, optional): Number of samples (number of RAG configurations to sample hyperparameters).
            sampler_type (SamplerType, optional): Sampler type. Defaults to SamplerType.SOBOL.
            **kwargs: Additional keyword arguments to pass to 
        
        Returns:
            tuple[list[RAGConfig], list[EvaluationDataset]]: Sampled configs and their evaluation datasets.
        """
        rag_configs = self._search_space.sample(n_samples=n_samples, sampler_type=sampler_type)
        
        documents = train_data.to_langchain_docs()

        configs: list[RAGConfig] = []
        datasets: list[EvaluationDataset] = []

        future_map: dict[Future[EvaluationDataset], RAGConfig] = {}

        for rag_config in rag_configs:
            rag = self.create_rag_instance(rag_config, documents=documents,initialize=True, **kwargs)
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
    
    def _load_components_parallel(self) -> None:
        """Load all components in parallel."""
        for llm_config in self._search_space.llm.choices:
            for model in llm_config.models:
                self.executor.submit(
                    self.get_llm,
                    model=model, 
                    provider=llm_config.provider, 
                    api_key=llm_config.api_key
                )
        
        for embedding_config in self._search_space.embedding.choices:
            for model in embedding_config.models:
                self.executor.submit(
                    self.get_embeddings,
                    provider=embedding_config.provider,
                    model=model,
                    api_key=embedding_config.api_key
                )
        
        for reranker_config in self._search_space.reranker.choices:
            for model in reranker_config.models:
                self.executor.submit(
                    self.get_reranker,
                    model=model,
                    provider=reranker_config.provider,
                    api_key=reranker_config.api_key
                )
        
    def _load_components_sequential(self) -> None:
        """Load all components sequentially."""
        for llm_config in self._search_space.llm.choices:
            for model in llm_config.models:
                self.get_llm(
                    model=model, 
                    provider=llm_config.provider, 
                    api_key=llm_config.api_key
                )
        
        for embedding_config in self._search_space.embedding.choices:
            for model in embedding_config.models:
                self.get_embeddings(
                    provider=embedding_config.provider,
                    model=model,
                    api_key=embedding_config.api_key
                )
        
        for reranker_config in self._search_space.reranker.choices:
            for model in reranker_config.models:
                self.get_reranker(
                    model=model,
                    provider=reranker_config.provider,
                    api_key=reranker_config.api_key
                )
        
        # NOTE:: Vector Stores wont be loaded statically here (as they will be depending on specific embedding model)
        # which is to be selected in the BO loop later 
 
    def create_rag_instance(
        self, 
        config: RAGConfig, 
        documents: Optional[list[Document]] = None,
        retrieval_config: Optional[dict] = None,
        initialize: bool = False
    ) -> RAGWorkflow:
        """Create RAGWorkflow instance from configuration using cached components."""
        # Initialize core components
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
        
        if config.reranker:
            reranker = self.get_reranker(
                model=config.reranker.model,
                provider=config.reranker.provider, 
                api_key=config.reranker.api_key
            )
        else:
            reranker = None
        
      
        index_name = config.vector_store.index_name
        vector_store = self.get_vector_store(
            provider=config.vector_store.provider, 
            embeddings=embeddings, 
            documents=documents or [], 
            index_name=index_name,
            api_key=config.vector_store.api_key,
            initialize=initialize
        )
        return RAGWorkflow(
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            reranker=reranker, 

            # TODO:: create wrapper around models(llms,embedding,..) which includes provider, model names
            llm_provider_name=config.llm.provider,
            llm_model_name=config.llm.model,
            embedding_provider_name=config.embedding.provider,
            embedding_model_name=config.embedding.model,
            reranker_provider_name=config.reranker.provider if config.reranker else None, 
            reranker_model_name=config.reranker.model if config.reranker else None,
            vector_store_provider_name=config.vector_store.provider,

            retrieval_config=retrieval_config or {"search_type": config.search_type, "k": config.k},
            corpus_documents=documents # needed for hybrid search
        )
    
    def create_rag_instance_by_sample(self, 
                                        sampler_type: SamplerType = SamplerType.SOBOL,
                                        documents: Optional[list[Document]] = None, 
                                        retrieval_config: Optional[dict] = None) -> RAGWorkflow:
        """Create RAGWorkflow instance from a sampled search space config."""
        sample = self._search_space.sample(n_samples=1,sampler_type=sampler_type)
        if not sample:
            logger.error("No sample found in search space")
            raise ValueError("No sample found in search space")
        return self.create_rag_instance(sample[0], documents=documents, retrieval_config=retrieval_config)

    def generate_evaluation_data(self,config: RAGConfig,  train_data: TrainDataset, **kwargs) -> EvaluationDataset:
        """Generate evaluation dataset from a sampled search space config."""
        documents = train_data.to_langchain_docs()
        rag = self.create_rag_instance(config, documents=documents, **kwargs)
        return rag.get_batch_answers(dataset=train_data, **kwargs)
    
    def clear_cache(self) -> None:
        """Clear all cached components."""
        with self._lock:
            self._registry.clear()
            logger.info("Component cache cleared")
