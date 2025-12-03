from botorch.utils.sampling import draw_sobol_samples
from typing import Optional
from rag_opt._config import (
    RAGConfig,
    EmbeddingConfig,
    EmbeddingModel,
    LLMConfig,
    LLMModel,
    RerankerConfig,
    RerankerModel,
    VectorStoreConfig,
    VectorStoreItem,
)
from abc import ABC, abstractmethod
from enum import Enum, auto
from loguru import logger
import numpy as np
import random
import torch


class SamplerType(Enum):
    """Supported sampling methods"""
    SOBOL = auto()
    RANDOM = auto()


class SamplingMixin(ABC):
    """Mixin for sampling RAG configurations with different strategies"""

    @abstractmethod
    def _get_hyperparameters(self) -> dict:
        """Get hyperparameter configuration"""
        raise NotImplementedError

    def _get_expanded_hyperparameters(self) -> dict:
        """Expand hyperparameters to include model selection dimensions"""
        expanded = {}
        base_params = self._get_hyperparameters()

        for param_name, config in base_params.items():
            if not config:
                continue
                
            expanded[param_name] = config.copy()
            choices = config.get("choices", {})
            
            if not isinstance(choices, dict):
                continue
            
            # Add model index parameter for model-based configs
            for choice_key, choice in choices.items():
                if self._is_model_based_config(choice):
                    models = self._get_models_from_config(choice)
                    if models:
                        model_param_name = f"{param_name}_model_idx"
                        expanded[model_param_name] = {
                            "searchspace_type": "categorical",
                            "choices": {str(i): i for i in range(len(models))},
                            "bounds": [0, len(models) - 1],
                            "dtype": int
                        }
                        break

        return expanded

    def _get_models_from_config(self, config) -> list[str]:
        """Extract models list from config (handles both Config objects and lists)"""
        if isinstance(config, list):
            return config
        elif hasattr(config, 'models'):
            return config.models
        return []

    def _is_model_based_config(self, config) -> bool:
        """Check if config has models (either as attribute or is a list)"""
        return isinstance(config, list) or (
            isinstance(config, (EmbeddingConfig, LLMConfig, RerankerConfig)) and 
            hasattr(config, 'models')
        )

    def get_parameter_bounds(self) -> torch.Tensor:
        """Get normalized [0,1] bounds for BoTorch"""
        expanded_params = self._get_expanded_hyperparameters()
        n_params = len(expanded_params)
        
        if n_params == 0:
            return torch.empty((2, 0), dtype=torch.float64)
        
        # All parameters normalized to [0, 1] for BoTorch
        bounds = torch.zeros((2, n_params), dtype=torch.float64)
        bounds[1, :] = 1.0
        
        return bounds

    def _normalize_value(self, value: float, lower: float, upper: float) -> float:
        """Normalize value from [lower, upper] to [0, 1]"""
        if upper == lower:
            return 0.5
        return (value - lower) / (upper - lower)

    def _denormalize_value(self, normalized: float, lower: float, upper: float, dtype: type) -> float:
        """Denormalize value from [0, 1] to [lower, upper]"""
        normalized = max(0.0, min(1.0, normalized))
        value = lower + normalized * (upper - lower)
        return value if dtype == float else int(round(value))

    def config_to_tensor(self, config: RAGConfig) -> torch.Tensor:
        """Encode RAGConfig to normalized tensor [0,1]"""
        expanded_params = self._get_expanded_hyperparameters()
        base_params = self._get_hyperparameters()

        sample = torch.zeros(len(expanded_params), dtype=torch.float64)
        param_idx = 0

        for param_name, param_config in base_params.items():
            config_value = getattr(config, param_name)
            param_type = param_config.get("searchspace_type")
            choices = param_config.get("choices")
            dtype = param_config.get("dtype")
            bounds = param_config.get("bounds", [0, 1])

            if param_type == 'continuous':
                raw_value = float(config_value) if dtype == float else int(config_value)
                sample[param_idx] = self._normalize_value(raw_value, bounds[0], bounds[1])

            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    choice_idx = self._find_matching_choice(config_value, choices_list)
                    n_choices = len(choices_list)
                    sample[param_idx] = choice_idx / (n_choices - 1) if n_choices > 1 else 0.5

                    # Handle model index
                    model_param_name = f"{param_name}_model_idx"
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx = self._extract_model_index(config_value, choices_list[choice_idx])
                        models = self._get_models_from_config(choices_list[choice_idx])
                        n_models = len(models)
                        sample[param_idx] = model_idx / (n_models - 1) if n_models > 1 else 0.5
                else:
                    sample[param_idx] = 0.5

            elif param_type == 'boolean':
                sample[param_idx] = 1.0 if config_value else 0.0

            param_idx += 1

        return sample
    
    def tensor_to_config(self, tensor: torch.Tensor) -> RAGConfig:
        """Decode tensor to RAGConfig"""
        return self.decode_sample_to_rag_config(tensor)

    def configs_to_tensor(self, configs: list[RAGConfig]) -> torch.Tensor:
        """Convert list of configs to tensor"""
        if not configs:
            expanded_params = self._get_expanded_hyperparameters()
            return torch.empty((0, len(expanded_params)), dtype=torch.float64)

        return torch.stack([self.config_to_tensor(config) for config in configs])

    def _extract_model_index(self, config_value, choice_config) -> int:
        """Extract model index from config"""
        # Handle gateway list case
        if isinstance(choice_config, list):
            if hasattr(config_value, 'model'):
                try:
                    return choice_config.index(config_value.model)
                except (ValueError, AttributeError):
                    return 0
            return 0
        
        # Handle standard config case
        if not (hasattr(config_value, 'model') and hasattr(choice_config, 'models')):
            return 0
        try:
            return choice_config.models.index(config_value.model)
        except (ValueError, AttributeError):
            return 0

    def _find_matching_choice(self, config_value, choices_list: list) -> int:
        """Find matching choice index"""
        for idx, choice in enumerate(choices_list):
            # Match gateway list
            if isinstance(choice, list):
                if hasattr(config_value, 'model') and config_value.model in choice:
                    return idx
            # Match by provider and model
            elif hasattr(config_value, 'provider') and hasattr(config_value, 'model'):
                if (config_value.provider == choice.provider and
                    hasattr(choice, 'models') and
                    config_value.model in choice.models):
                    return idx
            # Match by provider only
            elif hasattr(config_value, 'provider') and hasattr(choice, 'provider'):
                if config_value.provider == choice.provider:
                    return idx
            # Direct match
            elif config_value == choice:
                return idx
        return 0

    def decode_sample_to_rag_config(self, sample: torch.Tensor) -> RAGConfig:
        """Decode normalized sample to RAGConfig"""
        decoded = {}
        base_params = self._get_hyperparameters()
        expanded_params = self._get_expanded_hyperparameters()

        param_idx = 0
        for param_name, config in base_params.items():
            value = max(0.0, min(1.0, sample[param_idx].item()))
            param_type = config.get("searchspace_type")
            choices = config.get("choices")
            dtype = config.get("dtype")
            bounds = config.get("bounds", [0, 1])

            if param_type == 'continuous':
                decoded[param_name] = self._denormalize_value(value, bounds[0], bounds[1], dtype)

            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    if not choices_list:
                        raise ValueError(f"No choices for parameter '{param_name}'")
                    
                    choice_idx = min(int(round(value * (len(choices_list) - 1))), len(choices_list) - 1)
                    choice = choices_list[choice_idx]

                    # Get model index if present
                    model_param_name = f"{param_name}_model_idx"
                    model_idx = 0
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx_value = max(0.0, min(1.0, sample[param_idx].item()))
                        models = self._get_models_from_config(choice)
                        model_idx = int(round(model_idx_value * (len(models) - 1)))

                    decoded[param_name] = self._select_model_from_config(choice, model_idx, list(choices.keys())[choice_idx], param_name)

                elif isinstance(choices, list):
                    if not choices:
                        raise ValueError(f"No choices for parameter '{param_name}'")
                    choice_idx = min(int(round(value * (len(choices) - 1))), len(choices) - 1)
                    decoded[param_name] = choices[choice_idx]
                else:
                    decoded[param_name] = choices

            elif param_type == 'boolean':
                decoded[param_name] = value >= 0.5

            param_idx += 1

        return RAGConfig(**decoded)

    def _select_model_from_config(self, config, model_idx: int, provider_key: str = None, param_name: str = None):
        """Select specific model from config (handles both Config objects and gateway lists)"""
        # Handle gateway list case
        if isinstance(config, list):
            if not config:
                raise ValueError("Empty model list in gateway configuration")
            model_idx = max(0, min(model_idx, len(config) - 1))
            selected_model = config[model_idx]
            
            # Parse provider/model from gateway format (e.g., "openai/gpt-4o")
            if '/' in selected_model:
                provider, model = selected_model.split('/', 1)
            else:
                provider = provider_key or "gateway"
                model = selected_model
            
            # Return appropriate model type based on parameter name
            if param_name == 'embedding':
                return EmbeddingModel(
                    provider=provider,
                    model=model,
                    api_key=None,
                    api_base=None,
                    pricing=None
                )
            elif param_name == 'llm':
                return LLMModel(
                    provider=provider,
                    model=model,
                    api_key=None,
                    api_base=None,
                    pricing=None
                )
            elif param_name == 'reranker':
                return RerankerModel(
                    provider=provider,
                    model=model,
                    api_key=None,
                    api_base=None,
                    pricing=None
                )
            else:
                # Default to EmbeddingModel if unknown
                logger.warning(f"Unknown param_name '{param_name}' for gateway config, defaulting to EmbeddingModel")
                return EmbeddingModel(
                    provider=provider,
                    model=model,
                    api_key=None,
                    api_base=None,
                    pricing=None
                )
        
        # Handle VectorStoreConfig case
        if isinstance(config, VectorStoreConfig):
            return VectorStoreItem(
                provider=config.provider,
                index_name=config.index_name,
                api_key=config.api_key,
                pricing=config.pricing
            )
        
        # Handle standard Config objects
        models = self._get_models_from_config(config)
        if not models:
            return config

        model_idx = max(0, min(model_idx, len(models) - 1))
        selected_model = models[model_idx]

        if isinstance(config, EmbeddingConfig):
            return EmbeddingModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing else None
            )
        elif isinstance(config, LLMConfig):
            return LLMModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing else None
            )
        elif isinstance(config, RerankerConfig):
            return RerankerModel(
                provider=config.provider,
                model=selected_model,
                api_key=config.api_key,
                api_base=config.api_base,
                pricing=config.pricing.get(selected_model) if config.pricing else None
            )
        else:
            logger.warning(f"Unknown config type: {type(config)}")
            return config

    def sample(
        self,
        n_samples: int = 1,
        sampler_type: SamplerType = SamplerType.SOBOL,
        seed: Optional[int] = None
    ) -> list[RAGConfig]:
        """Sample RAG configurations"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        bounds = self.get_parameter_bounds()
        n_params = bounds.shape[1]

        if n_params == 0:
            return []

        if sampler_type == SamplerType.SOBOL:
            samples = draw_sobol_samples(
                bounds=bounds,
                n=n_samples,
                q=1,
                seed=seed
            ).squeeze(1)
        elif sampler_type == SamplerType.RANDOM:
            samples = torch.rand(n_samples, n_params, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        return [self.decode_sample_to_rag_config(samples[i]) for i in range(n_samples)]

    def sample_batch(
        self,
        batch_size: int,
        sampler_type: SamplerType = SamplerType.SOBOL,
        seed: Optional[int] = None
    ) -> list[RAGConfig]:
        """Convenience method for batch sampling"""
        return self.sample(n_samples=batch_size, sampler_type=sampler_type, seed=seed)