from botorch.utils.sampling import draw_sobol_samples
from typing import Any, Optional, TypedDict, NotRequired
from rag_opt._config import (
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
    RANDOM = auto()


class GeneralConfig(TypedDict, total=False):
    """General configuration for the search space (represents a RAG hyperparameter)"""
    searchspace_type: NotRequired[SearchSpaceType]
    choices: NotRequired[dict[str, Any]]
    bounds: NotRequired[list]
    dtype: NotRequired[type]


class SamplingMixin(ABC):
    """Mixin class for sampling RAG configurations with different sampling strategies"""

    @abstractmethod
    def _get_hyperparameters(self) -> dict[str, GeneralConfig]:
        raise NotImplementedError

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
                        break

        return expanded

    def get_parameter_bounds(self) -> torch.Tensor:
        """Extract bounds for all parameters - returns [0,1] bounds for BoTorch compatibility"""
        expanded_params = self._get_expanded_hyperparameters()
        n_params = len(expanded_params)
        
        if n_params == 0:
            return torch.empty((2, 0), dtype=torch.float64)
        
        # BoTorch expects all parameters normalized to [0, 1]
        bounds = torch.zeros((2, n_params), dtype=torch.float64)
        bounds[1, :] = 1.0 
        
        return bounds

    def _normalize_value(self, value: float, lower: float, upper: float) -> float:
        """Normalize a value from [lower, upper] to [0, 1]"""
        if upper == lower:
            return 0.5
        return (value - lower) / (upper - lower)

    def _denormalize_value(self, normalized: float, lower: float, upper: float, dtype: type) -> float:
        """Denormalize a value from [0, 1] to [lower, upper]"""
        # Clamp to [0, 1] first
        normalized = max(0.0, min(1.0, normalized))
        value = lower + normalized * (upper - lower)
        return value if dtype == float else int(round(value))

    def config_to_tensor(self, config: RAGConfig) -> torch.Tensor:
        """Encode a RAGConfig to its normalized tensor representation [0,1]"""
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
                # Normalize continuous values to [0, 1]
                raw_value = float(config_value) if dtype == float else int(config_value)
                sample[param_idx] = self._normalize_value(raw_value, bounds[0], bounds[1])

            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    choice_idx = self._find_matching_choice(config_value, choices_list)
                    n_choices = len(choices_list)
                    sample[param_idx] = choice_idx / (n_choices - 1) if n_choices > 1 else 0.5

                    # Handle model index if applicable
                    model_param_name = f"{param_name}_model_idx"
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx = self._extract_model_index(config_value, choices_list[choice_idx])
                        n_models = len(choices_list[choice_idx].models)
                        sample[param_idx] = model_idx / (n_models - 1) if n_models > 1 else 0.5
                else:
                    sample[param_idx] = 0.5

            elif param_type == 'boolean':
                sample[param_idx] = 1.0 if config_value else 0.0

            param_idx += 1

        return sample
    
    def tensor_to_config(self, tensor: torch.Tensor) -> RAGConfig:
        """Decode a tensor back to a RAGConfig instance"""
        config = self.decode_sample_to_rag_config(tensor)
        return config

    def configs_to_tensor(self, configs: list[RAGConfig]) -> torch.Tensor:
        """Convert list of RAGConfig objects to tensor"""
        if not configs:
            expanded_params = self._get_expanded_hyperparameters()
            return torch.empty((0, len(expanded_params)), dtype=torch.float64)

        tensors = [self.config_to_tensor(config) for config in configs]
        return torch.stack(tensors)

    def _extract_model_index(self, config_value: AIModel, choice_config: ModelBasedConfig) -> int:
        """Extract the model index from a model config"""
        if hasattr(config_value, 'model') and hasattr(choice_config, 'models'):
            try:
                return choice_config.models.index(config_value.model)
            except (ValueError, AttributeError):
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

    def decode_sample_to_rag_config(self, sample: torch.Tensor) -> RAGConfig:
        """Decode a normalized sample [0,1] back to parameter values"""
        decoded = {}
        base_params = self._get_hyperparameters()
        expanded_params = self._get_expanded_hyperparameters()

        param_idx = 0
        for param_name, config in base_params.items():
            value = sample[param_idx].item()
            param_type = config.get("searchspace_type")
            choices = config.get("choices")
            dtype = config.get("dtype")
            bounds = config.get("bounds", [0, 1])

            # Clamp value to [0,1]
            value = max(0.0, min(1.0, value))

            if param_type == 'continuous':
                decoded[param_name] = self._denormalize_value(value, bounds[0], bounds[1], dtype)

            elif param_type == 'categorical':
                if isinstance(choices, dict):
                    choices_list = list(choices.values())
                    if not choices_list:
                        raise ValueError(f"No choices available for parameter '{param_name}'")
                    
                    choice_idx = min(int(round(value * (len(choices_list) - 1))), len(choices_list) - 1)
                    choice = choices_list[choice_idx]

                    # Handle model index if present
                    model_param_name = f"{param_name}_model_idx"
                    model_idx = 0
                    if model_param_name in expanded_params:
                        param_idx += 1
                        model_idx_value = max(0.0, min(1.0, sample[param_idx].item()))
                        model_idx = int(round(model_idx_value * (len(choice.models) - 1)))

                    if isinstance(choice, (EmbeddingConfig, LLMConfig, RerankerConfig)):
                        decoded[param_name] = self._select_model_from_config(choice, model_idx)
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
                    if not choices:
                        raise ValueError(f"No choices available for parameter '{param_name}'")
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

        configs = [self.decode_sample_to_rag_config(samples[i]) for i in range(n_samples)]
        return configs

    def sample_batch(self,
                     batch_size: int,
                     sampler_type: SamplerType = SamplerType.SOBOL,
                     seed: Optional[int] = None) -> list[RAGConfig]:
        """Convenience method for batch sampling"""
        return self.sample(n_samples=batch_size, sampler_type=sampler_type, seed=seed)