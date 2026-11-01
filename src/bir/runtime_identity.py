"""Output-relevant environment identity without importing a GPU framework.

This remains usable for a cache-only resume on the same environment without
visible GPUs. Device visibility and filesystem locations are not identities.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform


DEPENDENCIES = (
    "torch", "torchvision", "transformers", "tokenizers", "huggingface-hub",
    "peft", "accelerate", "safetensors", "av", "numpy", "Pillow",
    "qwen-vl-utils", "einops", "flash-attn", "flash-linear-attention", "causal-conv1d",
)
NUMERICAL_ENVIRONMENT = (
    "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_SDP_KERNEL",
)


def runtime_identity():
    packages = {}
    for name in DEPENDENCIES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(schema_version="bir_runtime_identity_v1",
                python=platform.python_version(), implementation=platform.python_implementation(),
                system=platform.system(), machine=platform.machine(), packages=packages,
                numerical_environment={name: os.environ.get(name) for name in NUMERICAL_ENVIRONMENT})
