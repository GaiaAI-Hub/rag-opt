from botorch.utils.sampling import draw_sobol_samples, draw_sobol_normal_samples
from typing import Any, Optional, TypedDict, NotRequired
from rag_opt._config import  (
                            RAGConfig,
                            SearchSpaceType,
                            EmbeddingConfig,
                            EmbeddingModel,
                            LLMConfig,
                            LLMModel,
                            VectorStoreConfig,
                            VectorStoreItem,
                            RerankerConfig,
                            RerankerModel,
                            AIModel
                        )
from abc import ABC, abstractmethod
from enum import Enum, auto 
from loguru import logger
import numpy as np 
import random 
import torch 


ModelBasedConfig = EmbeddingConfig | LLMConfig | VectorStoreConfig | RerankerConfig

class SamplerType(Enum):
    """Supported sampling methods"""
    SOBOL = auto()
    QUASI_MONTE_CARLO = auto()
    RANDOM = auto()

class GeneralConfig(TypedDict, total=False):
    """ General configuration for the search space (represents a RAG hyperparameter)"""
    searchspace_type: NotRequired[SearchSpaceType] 
    choices: NotRequired[dict[str, Any]]
    bounds: NotRequired[list]
    dtype: NotRequired[type] 

class SamplingMixin(ABC):
    """ Mixin class for sampling RAG configurations with different sampling strategies """
    @abstractmethod
    def _get_hyperparameters(self) -> dict[str, GeneralConfig]:
        raise NotImplementedError
    
    @classmethod
    def _clip_value(cls, value: float, lower: float, upper: float, sample_type: SamplerType = SamplerType.SOBOL) -> float:
        """Clip value to a specific range

        Args:
            value (float): Value to clip
            lower (float): Lower bound
            upper (float): Upper bound
            sample_type: define how sample has been generated to define the clipping method

        Returns:
            float: Clipped value
        """
        if sample_type in (SamplerType.SOBOL, SamplerType.QUASI_MONTE_CARLO):
            return torch.clip(torch.tensor(value), min=torch.tensor(lower), max=torch.tensor(upper)).item()
        elif sample_type == SamplerType.RANDOM:
            return lower + value * (upper - lower)
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

    def _bound_samples(self, raw_samples: torch.Tensor, sample_type: SamplerType = SamplerType.SOBOL) -> torch.Tensor:
        """Clip samples to a specific range and adjust the type according to the search space"""
        samples = torch.zeros_like(raw_samples)
        expanded_params = self._get_expanded_hyperparameters()
        
        for i, (_, config) in enumerate(expanded_params.items()):
            bounds = config.get("bounds")
            dtype = torch.float64 if config.get("dtype") == float else torch.int32
            if bounds:
                lower, upper = bounds
                samples[:, i] = torch.tensor([self._clip_value(v.item(), lower, upper, sample_type=sample_type) 
                                            for v in raw_samples[:, i]] , dtype=dtype)
            else:
                samples[:, i] = raw_samples[:, i].clone().to(dtype)
        return samples

    def _get_expanded_hyperparameters(self) -> dict[str, GeneralConfig]:
        """Expand hyperparameters to include model selection dimensions"""
        expanded = {}
        base_params = self._get_hyperparameters()
        
        for param_name, config in base_params.items():
            if not config:
                continue
            expanded[param_name] = config.copy()
            
            choices = config.get("choices", {})
            if isinstance(choices, dict):
                choices_list = list(choices.values())
                for choice in choices_list:
                    if isinstance(choice, (EmbeddingConfig, LLMConfig, RerankerConfig)) and choice.models:
                        model_param_name = f"{param_name}_model_idx"
                        expanded[model_param_name] = {
                            "searchspace_type": "categorical",
                            "choices": {str(i): i for i in range(len(choice.models))},
                            "bounds": [0, len(choice.models) - 1],
                            "dtype": int
                        }
                        break  # Only need one model selection dimension per config type
        
        return expanded


    def get_parameter_bounds(self) -> torch.Tensor:
        """Extract bounds for all parameters including model selection dimensions"""
        bounds = []
        expanded_params = self._get_expanded_hyperparameters()
        
        for _, config in expanded_params.items():
            if config:
                search_space_type = config.get("searchspace_type")
                choices = config.get("choices", [])
                config_bounds = config.get("bounds", [])
                
                if search_space_type == "continuous":
                    bounds.append(config_bounds)
                elif search_space_type in ("categorical", "boolean"):
                    bounds.append([0, len(choices) - 1] if search_space_type == "categorical" else [0, 1])
                else:
                    logger.warning(f"Unknown search space type: {search_space_type}")
                    raise ValueError(f"Unknown search space type: {search_space_type}")
        
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64).T if bounds else torch.empty(0, dtype=torch.float64)
        return bounds_tensor
    
    def config_to_tensor(self, config: RAGConfig) -> torch.Tensor:
        """
        Encode a RAGConfig back to its tensor representation.
        This is the inverse of decode_sample_to_rag_config.
        
        Args:
            config: RAGConfig instance to encode
            
        Returns:
            torch.Tensor: Encoded configuration as 1D tensor
        """
        expanded_params = self._get_expanded_hyperparameters()
        base_params = self._get_hyperparameters()
        
        sample = torch.zeros(len(expanded_params), dtype=torch.float64)
        param_idx = 0
        
        for param_name, param_config in base_params.items():
            config_value = getattr(config, param_name)
            param_type = param_config.get("searchspace_type")
            choices = param_config.get("choices")
            dtype = param_config.get("dtype")
            # bounds = param_config.get("bounds", [0, 1])
            if param_type == 'continuous':
                sample[param_idx] = float(config_value) if dtype == float else int(config_value)
                    
            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    
                    # Find which choice matches the config value
                    choice_idx = self._find_matching_choice(config_value, choices_list)
                    # Normalize to [0, 1] for categorical
                    n_choices = len(choices_list)
                    sample[param_idx] = choice_idx / (n_choices - 1) if n_choices > 1 else 0.0
                    
                    # Handle model index if applicable
                    model_param_name = f"{param_name}_model_idx"
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx = self._extract_model_index(config_value, choices_list[choice_idx])
                        n_models = len(choices_list[choice_idx].models) 
                        sample[param_idx] = model_idx / (n_models - 1) if n_models > 1 else 0.0
                else:
                    sample[param_idx] = 0.0
                    
            elif param_type == 'boolean':
                sample[param_idx] = 1.0 if config_value else 0.0
            
            param_idx += 1
        
        return sample
    
    def configs_to_tensor(self, configs: list[RAGConfig]) -> torch.Tensor:
        """
        Convert list of RAGConfig objects to tensor.
        
        Args:
            configs: List of RAGConfig instances
            
        Returns:
            torch.Tensor: Tensor of shape (n_configs, n_params)
        """
        if not configs:
            expanded_params = self._get_expanded_hyperparameters()
            return torch.empty(0, len(expanded_params))
        
        tensors = [self.config_to_tensor(config) for config in configs]
        return torch.stack(tensors).double().detach().clone()
    
    def _extract_model_index(self, config_value: AIModel, choice_config: ModelBasedConfig) -> int:
        """Extract the model index from a model config"""
        if hasattr(config_value, 'model') and hasattr(choice_config, 'models'):
            try:
                return choice_config.models.index(config_value.model)
            except ValueError:
                return 0
        return 0
    
    def _find_matching_choice(self, config_value: AIModel, choices_list: list[ModelBasedConfig]) -> int:
        """Find the index of the matching choice in choices_list"""
        for idx, choice in enumerate(choices_list):
            if hasattr(config_value, 'provider') and hasattr(config_value, 'model'):
                if (config_value.provider == choice.provider and 
                    hasattr(choice, 'models') and 
                    config_value.model in choice.models):
                    return idx
            elif hasattr(config_value, 'provider') and hasattr(choice, 'provider'):
                if config_value.provider == choice.provider:
                    return idx
            elif config_value == choice:
                return idx
        return 0 
    
    def tensor_to_config(self, tensor: torch.Tensor) -> RAGConfig:
        """Decode a tensor back to a RAGConfig instance"""
        config = self.decode_sample_to_rag_config(tensor)
        return config

    def decode_sample_to_rag_config(self, sample: torch.Tensor) ->RAGConfig:
        """Decode a sample back to parameter values using sampled model indices"""
        decoded = {}
        base_params = self._get_hyperparameters()
        expanded_params = self._get_expanded_hyperparameters()
        
        # Extract model indices from sample
        model_indices = {}
        param_idx = 0
        
        for param_name, config in base_params.items():
            value = sample[param_idx].item()
            param_type = config.get("searchspace_type")
            choices = config.get("choices")
            dtype = config.get("dtype")
            
            if param_type == 'continuous':
                decoded[param_name] = value if dtype == float  else int(value)
            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    choice_idx = min(int(round(value * (len(choices_list) - 1))), len(choices_list) - 1)
                    choice = choices_list[choice_idx]
                    
                    model_param_name = f"{param_name}_model_idx"
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx_value = sample[param_idx].item()
                        model_indices[param_name] = int(round(model_idx_value * (len(choice.models) - 1)))
                    
                    if isinstance(choice, (EmbeddingConfig, LLMConfig, RerankerConfig)):
                        decoded[param_name] = self._select_model_from_config(choice, model_indices.get(param_name, 0))
                    elif isinstance(choice, VectorStoreConfig):
                        decoded[param_name] = VectorStoreItem(
                            provider=choice.provider,
                            index_name=choice.index_name,
                            pricing=choice.pricing,
                            api_key=choice.api_key
                        )
                    else:
                        decoded[param_name] = choice
                elif isinstance(choices, list):
                    choice_idx = min(int(round(value * (len(choices) - 1))), len(choices) - 1)
                    decoded[param_name] = choices[choice_idx]
                else:
                    decoded[param_name] = choices
            elif param_type == 'boolean':
                decoded[param_name] = value >= 0.5
            
            param_idx += 1
                
        return RAGConfig(**decoded)

    def _select_model_from_config(self, config: ModelBasedConfig, model_idx: int) -> ModelBasedConfig:
        """Select a specific model from a config object using the sampled index"""
        if not config.models:
            return config 
        
        # Ensure model_idx is within bounds
        model_idx = max(0, min(model_idx, len(config.models) - 1))
        selected_model = config.models[model_idx]
        
        if isinstance(config, EmbeddingConfig):
            return EmbeddingModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing and selected_model else None
            )
        elif isinstance(config, LLMConfig):
            return LLMModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing and selected_model else None
            )
        elif isinstance(config, RerankerConfig):
            return RerankerModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing and selected_model else None
            )
        else:
            logger.warning(f"Unknown config type: {type(config)}")
            return config

    def sample(self, 
               n_samples: int = 1, 
               sampler_type: SamplerType = SamplerType.SOBOL,
               seed: Optional[int] = None) -> list[RAGConfig]:
        """Sample RAG configurations using different sampling strategies"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        bounds = self.get_parameter_bounds()
        expanded_params = self._get_expanded_hyperparameters()
        n_params = len(expanded_params)

        if sampler_type == SamplerType.SOBOL:
            samples = draw_sobol_samples(
                bounds=bounds, 
                n=n_samples, 
                q=1,
                seed=seed
            ).squeeze(1)  
        elif sampler_type in (SamplerType.QUASI_MONTE_CARLO, SamplerType.RANDOM):
            if sampler_type == SamplerType.QUASI_MONTE_CARLO:
                raw_samples = draw_sobol_normal_samples(d=n_params, n=n_samples, seed=seed)
            else:
                raw_samples = torch.rand(n_samples, n_params)
            samples = self._bound_samples(raw_samples, sample_type=sampler_type)
        else:
            logger.error(f"Unknown sampler type: {sampler_type}")
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        configs = []
        for i in range(n_samples):
            sample = samples[i]
            config = self.decode_sample_to_rag_config(sample)
            configs.append(config)
        
        return configs

    def sample_batch(self,
                     batch_size: int,
                     sampler_type: SamplerType = SamplerType.SOBOL,
                     seed: Optional[int] = None) -> list[RAGConfig]:
        """Convenience method for batch sampling"""
        return self.sample(n_samples=batch_size, sampler_type=sampler_type, seed=seed)
