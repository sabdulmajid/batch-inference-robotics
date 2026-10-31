"""Placement-only controls for the unchanged FLORA model loader.

This module has no torch import until a placement context is entered.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math
from pathlib import Path
from unittest.mock import patch

GIB = 1024 ** 3


def memory_budgets(devices, headroom_gib=4.0, *, weight_bytes=None):
    """Return byte limits for logical CUDA devices from (total, free) pairs.

    The caller supplies only devices visible to its scheduler allocation.
    Headroom is reserved on every device. This is a feasibility floor, not
    proof that a model, its largest layer, and its activations will fit.
    """
    if not math.isfinite(headroom_gib) or headroom_gib <= 0:
        raise ValueError('Placement headroom must be finite and positive')
    headroom = math.ceil(headroom_gib * GIB)
    budgets = {}
    for index, (total, free) in enumerate(devices):
        if not (0 < free <= total):
            raise ValueError('Invalid CUDA total/free memory measurement')
        available = min(total, free) - headroom
        if available <= 0:
            raise RuntimeError('A visible GPU has no memory after placement headroom')
        budgets[index] = available
    if not budgets:
        raise RuntimeError('No CUDA device is visible')
    if weight_bytes is not None:
        if not isinstance(weight_bytes, int) or isinstance(weight_bytes, bool) or weight_bytes <= 0:
            raise ValueError('Checkpoint weight bytes must be a positive integer')
        if sum(budgets.values()) < weight_bytes:
            raise RuntimeError('Faithful checkpoint weights exceed all-GPU placement budget')
    return budgets


_DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
                'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4,
                'I64': 8, 'U64': 8, 'F64': 8}


def inspect_checkpoint(path, *, keep_fp32_names=('A_log', 'dt_bias')):
    """Read small safetensors headers and estimate native runtime weight bytes.

    Captioner is serialized in FP32, but the unchanged native loader casts
    base weights to BF16 and preserves selected Gated DeltaNet weights in
    FP32. Serialized bytes must not be used as runtime allocation bytes.
    No tensor payload is read. This estimate excludes activations and caches.
    """
    root = Path(path).resolve(strict=True)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Checkpoint index has no tensor map')
    storage_bytes = runtime_bytes = elements = 0
    dtype_elements = Counter()
    fp32_elements = 0
    tensors_seen = set()
    for filename in sorted(set(mapping.values())):
        shard = (root / filename).resolve(strict=True)
        if not shard.is_relative_to(root):
            raise ValueError('Checkpoint shard escapes the snapshot directory')
        with shard.open('rb') as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError('Truncated safetensors header size')
            header_size = int.from_bytes(size_bytes, 'little')
            if not 2 <= header_size <= 16 * 1024 * 1024:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(header_size))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            if mapping.get(name) != filename or name in tensors_seen:
                raise ValueError('Checkpoint index and shard header disagree')
            tensors_seen.add(name)
            dtype = spec['dtype']
            if dtype not in _DTYPE_BYTES:
                raise ValueError('Unsupported native checkpoint dtype: ' + dtype)
            shape = spec['shape']
            if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape):
                raise ValueError('Invalid checkpoint tensor shape')
            count = math.prod(shape)
            raw_bytes = count * _DTYPE_BYTES[dtype]
            offsets = spec['data_offsets']
            if len(offsets) != 2 or offsets[1] - offsets[0] != raw_bytes or offsets[0] < 0:
                raise ValueError('Checkpoint tensor byte count disagrees with its shape')
            if 8 + header_size + offsets[1] > shard.stat().st_size:
                raise ValueError('Truncated checkpoint tensor payload')
            keep_fp32 = any(part in keep_fp32_names for part in name.split('.'))
            floating = dtype.startswith(('F', 'BF'))
            native_bytes = count * (4 if keep_fp32 else 2) if floating else raw_bytes
            storage_bytes += raw_bytes
            runtime_bytes += native_bytes
            elements += count
            dtype_elements[dtype] += count
            if floating and keep_fp32:
                fp32_elements += count
    if tensors_seen != set(mapping):
        raise ValueError('Checkpoint tensor map contains missing tensors')
    declared = index.get('metadata', {}).get('total_size')
    if declared is not None and declared != storage_bytes:
        raise ValueError('Checkpoint declared total_size disagrees with its headers')
    return {'tensor_count': len(tensors_seen), 'elements': elements,
            'storage_tensor_bytes': storage_bytes, 'storage_elements_by_dtype': dict(dtype_elements),
            'native_weight_bytes_estimate': runtime_bytes,
            'native_fp32_elements': fp32_elements,
            'keep_fp32_names': list(keep_fp32_names),
            'excludes_activations_caches_and_allocator_overhead': True}


def checkpoint_weight_bytes(path):
    """Return a header-based native mixed-dtype weight estimate."""
    return inspect_checkpoint(path)['native_weight_bytes_estimate']


def verify_residency(model):
    """Reject parameter/buffer offload and report final logical-device bytes."""
    parameters = list(model.named_parameters())
    if not parameters:
        raise RuntimeError('The loaded model has no parameters')
    bad_parameters = [n for n, p in parameters if p.device.type != 'cuda']
    bad_buffers = [n for n, p in model.named_buffers() if p.numel() and p.device.type != 'cuda']
    bad_dtypes = [n for n, p in parameters if str(p.dtype) not in ('torch.bfloat16', 'torch.float32')]
    if bad_parameters or bad_buffers or bad_dtypes:
        raise RuntimeError('Native residency/dtype check failed: ' + json.dumps({
            'non_cuda_parameters': bad_parameters, 'non_cuda_buffers': bad_buffers,
            'non_native_parameter_dtypes': bad_dtypes}, sort_keys=True))
    mapping = getattr(model, 'hf_device_map', {})
    if any(str(v) in ('cpu', 'disk', 'meta') for v in mapping.values()):
        raise RuntimeError('Native inference cannot use CPU/disk/meta weight placement')
    if getattr(model, 'is_quantized', False):
        raise RuntimeError('Quantized inference does not satisfy the native contract')
    by_device = Counter()
    for _, tensor in parameters:
        by_device[str(tensor.device)] += tensor.numel() * tensor.element_size()
    return {
        'parameter_bytes_by_device': dict(by_device),
        'parameter_dtypes': dict(Counter(str(p.dtype) for _, p in parameters)),
        'buffer_devices': dict(Counter(str(p.device) for _, p in model.named_buffers() if p.numel())),
        'device_map': {k: str(v) for k, v in mapping.items()},
        'all_parameters_and_nonempty_buffers_on_cuda': True,
    }


@contextmanager
def captioner_placement(headroom_gib, audit):
    """Change only the native Captioner max_memory argument for one load.

    The backend holds its process-wide lock while this context is active.
    All remaining from_pretrained arguments pass through unchanged.
    """
    import torch
    from transformers import AutoModelForImageTextToText
    original = AutoModelForImageTextToText.from_pretrained

    def load(path, *args, **kwargs):
        if kwargs.get('device_map') != 'balanced':
            return original(path, *args, **kwargs)
        if kwargs.get('dtype') != torch.bfloat16 or kwargs.get('local_files_only') is not True:
            raise RuntimeError('The pinned Captioner loader contract changed')
        if kwargs.get('attn_implementation') != {'text_config': 'sdpa', 'vision_config': 'sdpa'}:
            raise RuntimeError('The pinned Captioner attention contract changed')
        devices, measurements = [], []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            devices.append((int(total), int(free)))
            measurements.append(dict(logical_device=i, total_bytes=int(total), free_bytes=int(free),
                allocated_bytes=int(torch.cuda.memory_allocated(i)), reserved_bytes=int(torch.cuda.memory_reserved(i))))
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
        keep_fp32 = tuple(getattr(Qwen3_5PreTrainedModel, '_keep_in_fp32_modules_strict', ()))
        if not {'A_log', 'dt_bias'} <= set(keep_fp32):
            raise RuntimeError('Native Gated DeltaNet FP32 preservation is not installed')
        checkpoint = inspect_checkpoint(path, keep_fp32_names=keep_fp32)
        size = checkpoint['native_weight_bytes_estimate']
        audit.update(policy='placement_only', headroom_gib=headroom_gib,
                     native_weight_bytes_estimate=size, checkpoint_header_inspection=checkpoint,
                     device_memory=measurements, feasibility='checking')
        try:
            budgets = memory_budgets(devices, headroom_gib, weight_bytes=size)
        except Exception as error:
            audit.update(feasibility='failed', error_type=type(error).__name__, error=str(error))
            raise
        audit.update(max_memory_bytes=budgets, feasibility='passed')
        kwargs['max_memory'] = budgets
        return original(path, *args, **kwargs)

    with patch.object(AutoModelForImageTextToText, 'from_pretrained', side_effect=load):
        yield
