import json
import os
from typing import Optional, Union

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    GptOssConfig,
    Llama4Config,
    Llama4TextConfig,
    LlamaConfig,
    Phi3Config,
    PretrainedConfig,
    Qwen2Config,
    Qwen3Config,
    Qwen3MoeConfig,
    modeling_utils,
)

from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.llama3_specblock import LlamaForCausalLMSpecBlock
from .target.custom_backend import (
    GptOssForCausalLM,
    Llama4ForCausalLM,
    LlamaForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)


class AutoDraftModel(AutoModelForCausalLMBase):
    """
    Unified Draft Model loader for Eagle3 and SpecBlock-Shift draft models.
    """

    # Model mapping: (algorithm, config_type) -> model_class
    _model_mapping = {
        ("eagle3", LlamaConfig): LlamaForCausalLMEagle3,
        ("specblock", LlamaConfig): LlamaForCausalLMSpecBlock,
    }

    # Architecture name to algorithm mapping
    _arch_to_algorithm = {
        "LlamaForCausalLMEagle3": "eagle3",
        "LlamaForCausalLMSpecBlock": "specblock",
    }

    @classmethod
    def _get_algorithm_from_config(cls, config: PretrainedConfig) -> str:
        """Determine algorithm type from config's architectures field."""
        architectures = getattr(config, "architectures", [])
        if not architectures:
            return "eagle3"  # Default to eagle3

        arch = architectures[0]
        algorithm = cls._arch_to_algorithm.get(arch)
        if algorithm is None:
            if "SpecBlock" in arch:
                return "specblock"
            else:
                return "eagle3"
        return algorithm

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        """
        Create a draft model from configuration.

        Args:
            config (PretrainedConfig): A configuration object.
            torch_dtype: Optional dtype for model parameters.
            **config_kwargs: Additional arguments passed to model constructor.

        Returns:
            A draft model instance.
        """
        algorithm = cls._get_algorithm_from_config(config)
        config_type = type(config)

        _model_cls = cls._model_mapping.get((algorithm, config_type))
        if _model_cls is None:
            raise ValueError(
                f"No model found for algorithm='{algorithm}', config_type={config_type}. "
                f"Available mappings: {list(cls._model_mapping.keys())}"
            )

        model = _model_cls(config, **config_kwargs)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        # Load config to determine algorithm and config type
        config = kwargs.get("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        algorithm = cls._get_algorithm_from_config(config)
        config_type = type(config)

        model_cls = cls._model_mapping.get((algorithm, config_type))
        if model_cls is None:
            raise ValueError(
                f"No model found for algorithm='{algorithm}', config_type={config_type}. "
                f"Available mappings: {list(cls._model_mapping.keys())}"
            )

        # Filter embedding weight warnings
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model




class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
        Qwen3MoeConfig: [Qwen3MoeForCausalLM],
        Qwen2Config: [Qwen2ForCausalLM],
        LlamaConfig: [LlamaForCausalLM],
        Qwen3Config: [Qwen3ForCausalLM],
        Phi3Config: [Phi3ForCausalLM],
        GptOssConfig: [GptOssForCausalLM],
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **config_kwargs,
    ):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
        )

        if isinstance(config, Llama4Config):
            config = config.text_config

        assert (
            type(config) in cls._model_mapping
        ), f"Unsupported config type: {type(config)}"
        model_cls = cls._model_mapping[type(config)][0]
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **config_kwargs,
        )

        if device is not None:
            model = model.to(device)
        else:
            model = model.cuda()
        return model


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "LlamaForCausalLMSpecBlock": LlamaConfig,
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "tie_word_embeddings" in config:
            print("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        # If draft_vocab_size is not in config or is None, set draft_vocab_size to vocab_size
        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        # Create base config object from standard HuggingFace config
        config_obj = cls._config_mapping[architecture].from_dict(config)

        # Preserve ALL custom parameters by iterating through the original JSON
        # and adding any keys that don't start with "_" (comment keys)
        for key, value in config.items():
            if not key.startswith("_"):
                # Always set the attribute (will override or add new ones)
                setattr(config_obj, key, value)

        return config_obj
