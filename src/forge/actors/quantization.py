"""
Quantization integration for forge's GeneratorWorker.

This module provides utilities to quantize models in-place after loading,
seamlessly integrating torch tao quantization into the vllm generation pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchao.quantization import (
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    quantize_,
)

logger = logging.getLogger(__name__)


class ForgeQuantizationConfig:
    """Configuration for quantizing models in forge GeneratorWorker."""

    def __init__(
        self,
        enabled: bool = True,
        method: str = "int8",
        group_size: Optional[int] = None,
        inner_k_tokens: Optional[int] = None,
    ):
        """
        Initialize forge quantization config.

        Args:
            enabled: Whether to enable quantization for the model (default: True).
            method: Quantization method - "int8" or "int4".
            group_size: Group size for quantization (for int4, None for per-channel).
            inner_k_tokens: Inner K tokens for activation-aware quantization.
        """
        self.enabled = enabled
        self.method = method
        self.group_size = group_size
        self.inner_k_tokens = inner_k_tokens

    def get_torch_ao_config(self):
        """Get the corresponding torch tao quantization config."""
        if self.method == "int8":
            return Int8WeightOnlyConfig()
        elif self.method == "int4":
            return Int4WeightOnlyConfig(
                group_size=self.group_size,
                inner_k_tokens=self.inner_k_tokens,
            )
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")


def quantize_loaded_model(
    model: nn.Module,
    config: ForgeQuantizationConfig,
) -> None:
    """
    Quantize a loaded model in-place using torch tao.

    This function is designed to be called in GeneratorWorker after model loading
    and before KV cache setup, to optimize the model for inference.

    Args:
        model: The loaded model to quantize (will be modified in-place).
        config: ForgeQuantizationConfig with quantization parameters.

    Example:
        >>> config = ForgeQuantizationConfig(enabled=True, method="int8")
        >>> quantize_loaded_model(model, config)
    """
    if not config.enabled:
        logger.debug("Quantization is disabled, skipping model quantization")
        return

    logger.info(
        f"Starting model quantization with method: {config.method}, "
        f"group_size={config.group_size}"
    )

    try:
        torch_ao_config = config.get_torch_ao_config()
        quantize_(model, torch_ao_config)
        logger.info(f"Model quantization completed successfully with {config.method}")
    except Exception as e:
        logger.error(f"Model quantization failed: {e}")
        raise


def create_quantization_config_from_dict(config_dict: dict) -> ForgeQuantizationConfig:
    """
    Create a ForgeQuantizationConfig from a dictionary.

    This is useful for loading quantization settings from configs or engine args.

    Args:
        config_dict: Dictionary with quantization configuration.

    Returns:
        ForgeQuantizationConfig instance.
    """
    return ForgeQuantizationConfig(**config_dict)
