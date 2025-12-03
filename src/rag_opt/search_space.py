from rag_opt._config import SearchType, LLMConfig, EmbeddingConfig, VectorStoreConfig, RerankerConfig, RerankerType, SearchSpaceType
from typing import Any, Union, Optional, Generic, TypeVar, ClassVar
from rag_opt.rag._pricing import (LLMTokenCost, 
                                  EmbeddingCost, 
                                  RerankerCost, 
                                  RerankerPricingType, 
                                  VectorStoreCost,
                                  PricingRegistry
                                )
from dataclasses import dataclass, fields
from ._sampler import SamplingMixin
from loguru import logger
from pathlib import Path
import yaml

HyperParameterConfig = TypeVar("HyperParameterConfig", SearchType, LLMConfig, EmbeddingConfig, VectorStoreConfig, RerankerConfig, RerankerType)
BoundedType = TypeVar("BoundedType", int, float)


@dataclass 
class ContinuousConfig(Generic[BoundedType]):
    """Configuration for continuous hyperparameters"""
    searchspace_type: SearchSpaceType
    bounds: list[BoundedType]
    dtype: Optional[type] = None

    def __post_init__(self):
        """Validate continuous configuration"""
        if self.searchspace_type != "continuous":
            raise ValueError(f"searchspace_type must be 'continuous', got '{self.searchspace_type}'")
        
        if not isinstance(self.bounds, list) or len(self.bounds) != 2:
            raise ValueError("bounds must be a list of exactly 2 values")
        
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"Lower bound ({self.bounds[0]}) must be less than upper bound ({self.bounds[1]})")
        
@dataclass
class CategoricalConfig(Generic[HyperParameterConfig]):
    """Configuration for categorical hyperparameters"""
    searchspace_type: SearchSpaceType
    choices: Union[list[HyperParameterConfig], dict[str, HyperParameterConfig]]

    def __post_init__(self):
        """Validate categorical configuration"""
        if self.searchspace_type != "categorical":
            raise ValueError(f"searchspace_type must be 'categorical', got '{self.searchspace_type}'")
        if isinstance(self.choices, list):
            if not self.choices:
                raise ValueError("choices cannot be empty")
        elif isinstance(self.choices, dict):
            if not self.choices:
                raise ValueError("choices cannot be empty")
        else:
            raise ValueError("choices must be either a list or dictionary")

@dataclass
class BooleanConfig:
    """Configuration for boolean hyperparameters"""
    searchspace_type: SearchSpaceType
    allow_multiple: bool = True

    def __post_init__(self):
        """Validate boolean configuration"""
        if self.searchspace_type != "boolean":
            raise ValueError(f"searchspace_type must be 'boolean', got '{self.searchspace_type}'")


@dataclass
class RAGSearchSpace(SamplingMixin):
    """Configuration class for RAG hyperparameters with typing and validation"""
    # Indexing parameters 
    chunk_size: ContinuousConfig[int]
    max_tokens: ContinuousConfig[int]
    chunk_overlap: ContinuousConfig[int]
    search_type: CategoricalConfig[SearchType]
    vector_store: CategoricalConfig[VectorStoreConfig]
    embedding: CategoricalConfig[EmbeddingConfig]
    
    # retrieval parameters
    k: ContinuousConfig[int]
    temperature: ContinuousConfig[float]
    
    # Generation parameters
    llm: CategoricalConfig[LLMConfig]

    # Reranking parameters
    use_reranker: BooleanConfig
    reranker: Optional[CategoricalConfig[RerankerConfig]] = None

    _pricing_registery: ClassVar[PricingRegistry] = PricingRegistry()


    def _validate_config(self) -> None:
        """Validate the configuration parameters"""
        errors = []
        
        try:
            if hasattr(self.chunk_size, 'bounds') and hasattr(self.chunk_overlap, 'bounds'):
                max_chunk_size = self.chunk_size.bounds[1]
                max_overlap = self.chunk_overlap.bounds[1]
                
                if max_overlap >= max_chunk_size:
                    errors.append(f"Maximum chunk overlap ({max_overlap}) must be less than maximum chunk size ({max_chunk_size})")
        
            if hasattr(self.max_tokens, 'bounds') and hasattr(self.chunk_size, 'bounds'):
                max_chunk_size = self.chunk_size.bounds[1]
                max_max_tokens = self.max_tokens.bounds[1]
                
                if max_max_tokens >= max_chunk_size:
                    errors.append(f"Maximum max tokens ({max_max_tokens}) must be less than maximum chunk size ({max_chunk_size})")
        except Exception as e:
            errors.append(f"Error validating chunk size/overlap: {str(e)}")
        
        try:
            if hasattr(self.k, 'bounds'):
                if self.k.bounds[0] < 1:
                    errors.append("Minimum k value must be at least 1")
                if self.k.bounds[1] > 100:
                    errors.append("Maximum k value should not exceed 100 for performance reasons")
        except Exception as e:
            errors.append(f"Error validating k parameter: {str(e)}")
        
        try:
            if hasattr(self.temperature, 'bounds'):
                if self.temperature.bounds[0] < 0.0:
                    errors.append("Temperature lower bound cannot be negative")
                if self.temperature.bounds[1] > 2.0:
                    errors.append("Temperature upper bound should not exceed 2.0")
        except Exception as e:
            errors.append(f"Error validating temperature: {str(e)}")

        if errors:
            logger.error(f"Configuration validation failed: {'; '.join(errors)}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
    @classmethod
    def _create_config_object(cls, param_name: str, param_config: dict) -> Any:
        """Create appropriate configuration object based on parameter type and name"""
        space_type = param_config.get('searchspace_type')
        
        if space_type == 'continuous':
            dtype_str = param_config.get('dtype', 'float')
            dtype = int if dtype_str == 'int' else float
            return ContinuousConfig(
                searchspace_type=space_type,
                bounds=param_config['bounds'],
                dtype=dtype
            )
        elif space_type == 'categorical':
            choices: dict[str, Any] = param_config['choices']
            
            # Process provider configs
            if isinstance(choices, dict) and param_name in ['vector_store', 'embedding', 'llm', 'reranker']:
                processed_choices = {}
                for provider_name, provider_config in choices.items():
                    # Handle gateway-style list configuration
                    if provider_name == 'gateway' and isinstance(provider_config, list):
                        # Keep gateway as a list of model strings
                        processed_choices[provider_name] = provider_config
                        continue
                    
                    # Handle both dict and pre-processed configs
                    if not isinstance(provider_config, dict):
                        processed_choices[provider_name] = provider_config
                        continue
                    
                    if param_name == 'vector_store':
                        processed_choices[provider_name] = VectorStoreConfig(
                            provider=provider_name,
                            **{k: v for k, v in provider_config.items() if k != 'pricing'}
                        )
                    elif param_name == 'embedding':
                        models = provider_config.get('models', [provider_name])
                        processed_choices[provider_name] = EmbeddingConfig(
                            provider=provider_name,
                            models=models,
                            **{k: v for k, v in provider_config.items() if k not in ['models', 'pricing']}
                        )
                    elif param_name == 'llm':
                        models = provider_config.get('models', [provider_name])
                        processed_choices[provider_name] = LLMConfig(
                            provider=provider_name,
                            models=models,
                            **{k: v for k, v in provider_config.items() if k not in ['models', 'pricing']}
                        )
                    elif param_name == 'reranker':
                        models = provider_config.get('models', [provider_name])
                        processed_choices[provider_name] = RerankerConfig(
                            provider=provider_name,
                            models=models,
                            **{k: v for k, v in provider_config.items() if k not in ['models', 'pricing']}
                        )
                choices = processed_choices
            
            return CategoricalConfig(
                searchspace_type=space_type,
                choices=choices
            )
        elif space_type == 'boolean':
            return BooleanConfig(
                searchspace_type=space_type,
                allow_multiple=param_config.get('allow_multiple', True)
            )
        else:
            raise ValueError(f"Unknown search space type: {space_type}")


    @classmethod
    def get_default_search_space_config(cls) -> "RAGSearchSpace": 
        """Get the search space configuration with multiple choice support"""
        return cls(
            chunk_size=ContinuousConfig(
                searchspace_type="continuous",
                bounds=[200, 2000],
                dtype=int
            ),
            max_tokens=ContinuousConfig(
                searchspace_type="continuous",
                bounds=[100, 2000],
                dtype=int
            ),
            chunk_overlap=ContinuousConfig(
                searchspace_type="continuous", 
                bounds=[0, 500],
                dtype=int
            ),
            search_type=CategoricalConfig(
                searchspace_type="categorical",
                choices=["similarity",  "mmr", "bm25", "tfidf", "hybrid"]
            ),
            vector_store=CategoricalConfig(
                searchspace_type="categorical",
                choices={
                    "faiss": VectorStoreConfig("faiss"),
                    "chroma": VectorStoreConfig("chroma"),
                    "pinecone": VectorStoreConfig("pinecone"),
                    "weaviate": VectorStoreConfig("weaviate")
                }
            ),
            embedding=CategoricalConfig(
                searchspace_type="categorical",
                choices={
                    "openai": EmbeddingConfig("openai", ["text-embedding-ada-002"]),
                    "huggingface": EmbeddingConfig("huggingface", ["all-MiniLM-L6-v2"]),
                    "sentence-transformers": EmbeddingConfig("sentence-transformers", ["all-MiniLM-L6-v2"])
                }
            ),
            k=ContinuousConfig(
                searchspace_type="continuous",
                bounds=[1, 20],
                dtype=int
            ),
            temperature=ContinuousConfig(
                searchspace_type="continuous",
                bounds=[0.0, 2.0],
                dtype=float
            ),
            use_reranker=BooleanConfig(
                searchspace_type="boolean",
                allow_multiple=True
            ),
            reranker=CategoricalConfig(
                searchspace_type="categorical",
                choices={
                    "cross_encoder": RerankerConfig("cross_encoder", ["msmarco-MiniLM-L-6-v3"]),
                    "colbert": RerankerConfig("colbert", ["msmarco-MiniLM-L-6-v3"]),
                    "bge": RerankerConfig("bge", ["bge-reranker-base"])
                }
            ),
            llm=CategoricalConfig(
                searchspace_type="categorical",
                choices={
                    "openai": LLMConfig("openai", ["gpt-5-nano"]),
                    "deepseek": LLMConfig("deepseek", ["deepseek-v3.2-exp"])
                }
            )
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RAGSearchSpace":
        """Load custom search space configuration from YAML"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.error(f"Configuration file not found: {yaml_path}")
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        logger.success(f"Loading configuration from {yaml_path}")   
        with open(yaml_path, 'r') as f:
            raw_config: dict[str, dict] = yaml.safe_load(f)
        
        search_space = {}
        
        for param_name, param_config in raw_config.items():
            search_space[param_name] = cls._create_config_object(param_name, param_config)
        

        
        return cls(**search_space)

    def to_dict(self) -> dict[str, Any]:
        """Convert RAGSearchSpace to dictionary"""
        result = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if hasattr(value, '__dict__') and value.__dict__ is not None:
                result[field_info.name] = {
                    k: v for k, v in value.__dict__.items() 
                    if not k.startswith('_')
                }
            elif value is not None:
                result[field_info.name] = value
        return result

    def _get_hyperparameters(self):
        return {param_name: config for param_name, config in self.to_dict().items() }
    
    def to_yaml(self, path: Union[str, Path] = "./rag_config.yaml") -> None:
        """Save RAGSearchSpace to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)