"""Conservative, opt-in Captioner microbatches for prepared native inputs.

CPU collation is available for verification. GPU generation requires an exact
measured admission record; importing this module never enables engine batching.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import time
from typing import Mapping

MODEL_KEYS = frozenset({'input_ids', 'attention_mask', 'mm_token_type_ids',
                       'pixel_values_videos', 'video_grid_thw'})
SEQUENCE_KEYS = frozenset({'input_ids', 'attention_mask', 'mm_token_type_ids'})


class BatchCompatibilityError(ValueError):
    pass


class BatchAdmissionError(RuntimeError):
    pass


class BatchExecutionError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__('Native batch execution failed; retain the attached record')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def input_tensor_audit(inputs):
    """Use the same tensor representation as the pinned native audit helper."""
    import torch
    audit = {}
    for key, value in inputs.items():
        if key == 'video_metadata':
            continue
        if not torch.is_tensor(value):
            raise BatchCompatibilityError('Model inputs must be tensors')
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        audit[key] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                      'sha256': hashlib.sha256(raw).hexdigest()}
    return {'input_tensors': audit, 'input_tensor_digest': _digest(audit)}


@dataclass(frozen=True)
class CaptionRequest:
    request_id: str
    episode_id: str
    event_id: str
    profile: str
    contract_id: str
    inputs: Mapping
    max_new_tokens: int = 512


@dataclass(frozen=True)
class PreparedBatch:
    inputs: Mapping
    requests: tuple[CaptionRequest, ...]
    routing: tuple[dict, ...]
    compatibility_signature: str
    input_tokens: int
    video_token_id: int
    collated_tensor_bytes: int
    prepared_bytes: int
    collation_seconds: float


def _describe(request, video_token_id, merge_size):
    import torch
    for field in ('request_id', 'episode_id', 'event_id', 'profile', 'contract_id'):
        if not isinstance(getattr(request, field), str) or not getattr(request, field):
            raise BatchCompatibilityError('Caption requests require nonempty identities and a pinned contract')
    if request.max_new_tokens != 512:
        raise BatchCompatibilityError('Only the native 512-token caption budget is supported')
    keys = set(request.inputs)
    if keys - MODEL_KEYS - {'video_metadata'} or MODEL_KEYS - keys:
        raise BatchCompatibilityError('Unrecognized or missing native caption tensor keys')
    values = {key: request.inputs[key] for key in MODEL_KEYS}
    if any(not torch.is_tensor(value) or value.device.type != 'cpu' for value in values.values()):
        raise BatchCompatibilityError('Collation requires CPU tensors from native preparation')
    sequence = values['input_ids']
    if sequence.ndim != 2 or sequence.shape[0] != 1 or sequence.shape[1] < 1:
        raise BatchCompatibilityError('Each request must have one nonempty input sequence')
    length = int(sequence.shape[1])
    if length + request.max_new_tokens > 32768:
        raise BatchCompatibilityError('The native context budget would be exceeded')
    for key in SEQUENCE_KEYS:
        if values[key].shape != sequence.shape or values[key].dtype != torch.int64:
            raise BatchCompatibilityError('Sequence tensors must have equal [1,N] int64 shapes')
    if not bool(torch.all(values['attention_mask'] == 1)):
        raise BatchCompatibilityError('Padded or masked sequences are not supported by this batch path')
    types = values['mm_token_type_ids']
    if not bool(torch.all((types == 0) | (types == 2))):
        raise BatchCompatibilityError('Only text and video modality token types are supported')
    if not torch.equal(types == 2, sequence == video_token_id):
        raise BatchCompatibilityError('Video placeholder and modality token locations disagree')
    grids = values['video_grid_thw']
    if grids.ndim != 2 or grids.shape[0] < 1 or grids.shape[1] != 3 or grids.dtype != torch.int64:
        raise BatchCompatibilityError('Video grids must have shape [blocks,3] and dtype int64')
    if not bool(torch.all(grids > 0)) or not bool(torch.all(grids[:, 1:] % merge_size == 0)):
        raise BatchCompatibilityError('Video grids violate native merge geometry')
    patches = values['pixel_values_videos']
    if patches.ndim != 2 or patches.dtype != torch.float32 or patches.shape[1] != 1536:
        raise BatchCompatibilityError('Only native FP32 RGB/temporal-pair/16-pixel video patches are supported')
    expected_patches = int(grids.prod(dim=1).sum())
    if patches.shape[0] != expected_patches:
        raise BatchCompatibilityError('Visual patch rows do not match the video grids')
    if int((sequence == video_token_id).sum()) != expected_patches // merge_size ** 2:
        raise BatchCompatibilityError('Visual tokens do not match the video grids')
    description = {'profile': request.profile, 'contract_id': request.contract_id,
                   'max_new_tokens': request.max_new_tokens, 'video_token_id': video_token_id,
                   'merge_size': merge_size, 'video_grid_thw': grids.tolist(),
                   'tensors': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                               for key, value in sorted(values.items())}}
    return values, description


def collate_caption_requests(requests, *, video_token_id, max_batch_size,
                             max_prepared_bytes, merge_size=2):
    """Concatenate exact compatible inputs without padding, resizing, or sampling.

    This CPU operation is not GPU admission. Every request must have the same
    profile, contract, tensor shapes, grid values, and input-token count.
    """
    import torch
    started = time.perf_counter()
    requests = tuple(requests)
    for name, value in [('max_batch_size', max_batch_size), ('max_prepared_bytes', max_prepared_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BatchCompatibilityError(name + ' must be a positive explicit bound')
    if not 1 <= len(requests) <= max_batch_size:
        raise BatchCompatibilityError('The batch exceeds its explicit request bound')
    if merge_size != 2 or isinstance(video_token_id, bool) or not isinstance(video_token_id, int) or video_token_id < 0:
        raise BatchCompatibilityError('Unsupported native video token or merge size')
    if len({r.request_id for r in requests}) != len(requests):
        raise BatchCompatibilityError('Duplicate caption request IDs are not allowed')
    if len({(r.episode_id, r.event_id) for r in requests}) != len(requests):
        raise BatchCompatibilityError('Duplicate episode/event routes are not allowed')
    prepared, descriptions = zip(*(_describe(r, video_token_id, merge_size) for r in requests))
    if any(description != descriptions[0] for description in descriptions[1:]):
        raise BatchCompatibilityError('Caption inputs are not exactly batch-compatible')
    total_bytes = sum(t.numel() * t.element_size() for values in prepared for t in values.values())
    if 2 * total_bytes > max_prepared_bytes:
        raise BatchCompatibilityError('The prepared input byte bound would be exceeded')
    routing, pixel_offset, grid_offset = [], 0, 0
    for row, (request, values) in enumerate(zip(requests, prepared, strict=True)):
        pixels, grids = values['pixel_values_videos'].shape[0], values['video_grid_thw'].shape[0]
        routing.append({'output_row': row, 'request_id': request.request_id, 'episode_id': request.episode_id,
                        'event_id': request.event_id, 'pixel_rows': [pixel_offset, pixel_offset + pixels],
                        'grid_rows': [grid_offset, grid_offset + grids], **input_tensor_audit(values)})
        pixel_offset += pixels
        grid_offset += grids
    # Sequences acquire a batch dimension. Visual rows remain flattened in the
    # same request order consumed by Qwen3.5's visual scatter and position code.
    inputs = {key: torch.cat([values[key] for values in prepared], dim=0) for key in sorted(MODEL_KEYS)}
    return PreparedBatch(inputs, requests, tuple(routing), _digest(descriptions[0]),
                         int(inputs['input_ids'].shape[1]), video_token_id, total_bytes, 2 * total_bytes,
                         time.perf_counter() - started)


def verify_batch_integrity(batch):
    """Reject mutation or changed routes between CPU preparation and execution."""
    if set(batch.inputs) != MODEL_KEYS or len(batch.routing) != len(batch.requests):
        raise BatchCompatibilityError('The prepared batch structure changed')
    for key in SEQUENCE_KEYS:
        if list(batch.inputs[key].shape) != [len(batch.requests), batch.input_tokens]:
            raise BatchCompatibilityError('The prepared sequence shape changed')
    pixel_offset = grid_offset = 0
    for row, (request, route) in enumerate(zip(batch.requests, batch.routing, strict=True)):
        if any(route[key] != value for key, value in {
            'output_row': row, 'request_id': request.request_id,
            'episode_id': request.episode_id, 'event_id': request.event_id}.items()):
            raise BatchCompatibilityError('Prepared output routing changed')
        pixels = request.inputs['pixel_values_videos'].shape[0]
        grids = request.inputs['video_grid_thw'].shape[0]
        if route['pixel_rows'] != [pixel_offset, pixel_offset + pixels] or route['grid_rows'] != [grid_offset, grid_offset + grids]:
            raise BatchCompatibilityError('Prepared visual row routing changed')
        values = {key: batch.inputs[key][row:row + 1] for key in SEQUENCE_KEYS}
        values.update(pixel_values_videos=batch.inputs['pixel_values_videos'][pixel_offset:pixel_offset + pixels],
                      video_grid_thw=batch.inputs['video_grid_thw'][grid_offset:grid_offset + grids])
        for candidate in (request.inputs, values):
            if input_tensor_audit(candidate)['input_tensor_digest'] != route['input_tensor_digest']:
                raise BatchCompatibilityError('Prepared input tensors changed after collation')
        pixel_offset += pixels
        grid_offset += grids
    if batch.inputs['pixel_values_videos'].shape[0] != pixel_offset or batch.inputs['video_grid_thw'].shape[0] != grid_offset:
        raise BatchCompatibilityError('Prepared visual rows contain unassigned inputs')


@dataclass(frozen=True)
class MeasuredBatchAdmission:
    """Operator-approved evidence for one exact batch shape and GPU setup.

    No admission is shipped. These fields must come from a completed real pilot
    and an independent native batch-one comparison. Values are not predictions.
    """
    evidence_id: str
    compatibility_signature: str
    runtime_fingerprint: str
    batch_size: int
    baseline_allocated_bytes: tuple[int, ...]
    baseline_reserved_bytes: tuple[int, ...]
    peak_allocated_bytes: tuple[int, ...]
    peak_reserved_bytes: tuple[int, ...]
    safety_headroom_bytes: int
    max_generated_tokens_observed: int
    native_agreement_verified: bool = False

    def require(self, batch, *, runtime_fingerprint, total_bytes, free_bytes, allocated_bytes, reserved_bytes):
        if not isinstance(self.evidence_id, str) or not self.evidence_id or self.native_agreement_verified is not True:
            raise BatchAdmissionError('Real memory and native-agreement evidence is required')
        if self.max_generated_tokens_observed != 512:
            raise BatchAdmissionError('The native 512-token output memory envelope has not been measured')
        if self.batch_size != len(batch.requests) or self.compatibility_signature != batch.compatibility_signature:
            raise BatchAdmissionError('Batch size or compatibility signature differs from its measured pilot')
        if self.runtime_fingerprint != runtime_fingerprint:
            raise BatchAdmissionError('Hardware, placement, or software differs from its measured pilot')
        if isinstance(self.safety_headroom_bytes, bool) or not isinstance(self.safety_headroom_bytes, int) or self.safety_headroom_bytes <= 0:
            raise BatchAdmissionError('Positive explicit memory headroom is required')
        counts = [len(x) for x in (self.baseline_allocated_bytes, self.baseline_reserved_bytes,
                                  self.peak_allocated_bytes, self.peak_reserved_bytes,
                                  total_bytes, free_bytes, allocated_bytes, reserved_bytes)]
        if not counts[0] or len(set(counts)) != 1:
            raise BatchAdmissionError('Every visible device needs measured memory evidence')
        measurements = zip(self.baseline_allocated_bytes, self.baseline_reserved_bytes,
                           self.peak_allocated_bytes, self.peak_reserved_bytes,
                           total_bytes, free_bytes, allocated_bytes, reserved_bytes, strict=True)
        for baseline_allocated, baseline_reserved, peak_allocated, peak, total, free, allocated, reserved in measurements:
            values = (baseline_allocated, baseline_reserved, peak_allocated, peak, total, free, allocated, reserved)
            if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
                raise BatchAdmissionError('Memory evidence must use integer bytes')
            if not (0 < baseline_allocated <= baseline_reserved <= peak <= total and
                    baseline_allocated <= peak_allocated <= peak and 0 <= allocated <= reserved <= total and
                    0 <= free <= total):
                raise BatchAdmissionError('Invalid measured memory evidence')
            if peak + self.safety_headroom_bytes > total:
                raise BatchAdmissionError('The measured batch has insufficient device headroom')
            if allocated > baseline_allocated:
                raise BatchAdmissionError('Current live allocation exceeds the measured pilot baseline')
            reusable_cache = reserved - allocated
            # Live model/state tensors cannot be reused as activation storage.
            # Charge the pilot workspace above its live baseline, including
            # allocator reserve, and subtract only currently unused cache.
            workspace = peak - baseline_allocated
            needed = max(0, workspace - reusable_cache) + self.safety_headroom_bytes
            if free < needed:
                raise BatchAdmissionError('Current free device memory is below the measured envelope')


def extract_batch_outputs(output, batch, tokenizer):
    """Extract each row through its own first EOS; padding is never a caption."""
    import torch
    if not torch.is_tensor(output) or output.dtype != torch.int64 or output.ndim != 2 or output.shape[0] != len(batch.requests):
        raise RuntimeError('Generation returned an unexpected batch shape')
    if output.shape[1] < batch.input_tokens or output.shape[1] > batch.input_tokens + 512:
        raise RuntimeError('Generation changed the native input/output length contract')
    prefix = output[:, :batch.input_tokens].detach().cpu()
    if not torch.equal(prefix, batch.inputs['input_ids']):
        raise RuntimeError('Generated row prefixes do not match request routing')
    eos_value = tokenizer.eos_token_id
    eos = {eos_value} if isinstance(eos_value, int) else set(eos_value or [])
    pad = tokenizer.pad_token_id
    if not eos or not all(isinstance(value, int) for value in eos) or not isinstance(pad, int):
        raise RuntimeError('The native tokenizer requires EOS and padding token IDs')
    rows = []
    for route, sequence in zip(batch.routing, output[:, batch.input_tokens:].detach().cpu().tolist(), strict=True):
        record = {key: route[key] for key in ('output_row', 'request_id', 'episode_id', 'event_id', 'input_tensor_digest')}
        stops = [i for i, token in enumerate(sequence) if token in eos]
        if not stops:
            record.update(status='failed_no_eos', caption='', generated_tokens=len(sequence))
        else:
            stop = stops[0]
            ids = sequence[:stop + 1]
            trailing = sequence[stop + 1:]
            if any(token != pad for token in trailing):
                record.update(status='failed_tokens_after_eos', caption='', generated_tokens=len(ids))
            else:
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                record.update(status='complete' if text else 'failed_empty_caption', caption=text,
                              generated_tokens=len(ids), padding_tokens=len(trailing), eos_token_id=ids[-1])
        rows.append(record)
    return rows


@dataclass(frozen=True)
class VerifiedNativeBinding:
    """In-process receipt issued from the verified native session's load audit."""
    source_revision: str
    model_revision: str
    processor_revision: str
    execution_contract_id: str
    parameter_verification_sha256: str
    model_object_id: int
    processor_object_id: int


def bind_verified_native(model, processor, *, source_revision, model_revision,
                         processor_revision, execution_contract_id, model_audit):
    from .native import UPSTREAM_COMMIT, MODEL_REVISIONS
    from .placement import verify_residency
    if source_revision != UPSTREAM_COMMIT or model_revision != MODEL_REVISIONS['captioner'] or processor_revision != model_revision:
        raise BatchAdmissionError('Batching requires the pinned native Captioner and processor revisions')
    if not execution_contract_id:
        raise BatchAdmissionError('A pinned engine execution contract is required')
    if model.__class__.__name__ != 'Qwen3_5ForConditionalGeneration' or processor.__class__.__name__ != 'Qwen3VLProcessor':
        raise BatchAdmissionError('Only the native Captioner model and processor classes are supported')
    gdn = model_audit.get('gdn', {})
    info = model_audit.get('loading_info', {})
    if gdn.get('all_values_equal_saved_checkpoint') is not True or gdn.get('all_runtime_fp32') is not True:
        raise BatchAdmissionError('The native checkpoint parameter verification receipt is missing')
    if any(info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise BatchAdmissionError('The native load audit contains incomplete checkpoint loading')
    residency = verify_residency(model)
    if model_audit.get('parameter_dtypes') != residency['parameter_dtypes']:
        raise BatchAdmissionError('Model parameter dtypes differ from the verified load receipt')
    return VerifiedNativeBinding(source_revision, model_revision, processor_revision,
        execution_contract_id, _digest(model_audit), id(model), id(processor))


def runtime_fingerprint(model, processor, batch, *, topology_fingerprint, native_binding):
    """Bind real processor identity, native provenance, code, and GPU placement."""
    import torch
    from pathlib import Path
    from .native import UPSTREAM_COMMIT, MODEL_REVISIONS
    from .batching_contract import runtime_source_hashes
    if not isinstance(native_binding, VerifiedNativeBinding):
        raise BatchAdmissionError('A verified native-session binding is required')
    if native_binding.model_object_id != id(model) or native_binding.processor_object_id != id(processor):
        raise BatchAdmissionError('The verified native model or processor instance changed')
    if (native_binding.source_revision != UPSTREAM_COMMIT or
            native_binding.model_revision != MODEL_REVISIONS['captioner'] or
            native_binding.processor_revision != MODEL_REVISIONS['captioner']):
        raise BatchAdmissionError('The native source, model, or processor revision changed')
    if any(request.contract_id != native_binding.execution_contract_id for request in batch.requests):
        raise BatchAdmissionError('The prepared batch uses a different native execution contract')
    if batch.video_token_id != processor.video_token_id:
        raise BatchAdmissionError('The collated video token does not match the verified processor')
    if not topology_fingerprint:
        raise BatchAdmissionError('A verified same-node topology receipt is required')
    tokenizer = processor.tokenizer
    tokenizer_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    processor_settings = {'class': processor.__class__.__name__, 'chat_template': processor.chat_template,
                          'video_processor': processor.video_processor.to_dict(),
                          'image_processor': processor.image_processor.to_dict(),
                          'video_token_id': processor.video_token_id,
                          'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id}
    config = model.config.to_dict()
    config.pop('_name_or_path', None)
    devices = []
    for i in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(i)
        uuid = getattr(properties, 'uuid', None)
        if not uuid:
            raise BatchAdmissionError('Current physical UUID mapping is required for every visible CUDA device')
        devices.append({'name': torch.cuda.get_device_name(i), 'total_bytes': properties.total_memory,
                        'capability': list(torch.cuda.get_device_capability(i)), 'uuid': str(uuid)})
    details = {'torch': torch.__version__, 'cuda': torch.version.cuda,
               'transformers': importlib.metadata.version('transformers'),
               'topology_fingerprint': topology_fingerprint, 'model_class': model.__class__.__name__,
               'model_config_sha256': _digest(config),
               'model_generation_config_sha256': _digest(model.generation_config.to_dict()),
               'tokenizer_sha256': tokenizer_sha, 'processor_sha256': _digest(processor_settings),
               'batching_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'native_runtime_source_sha256': runtime_source_hashes(),
               'native_binding': {'source_revision': native_binding.source_revision,
                                  'model_revision': native_binding.model_revision,
                                  'processor_revision': native_binding.processor_revision,
                                  'execution_contract_id': native_binding.execution_contract_id,
                                  'parameter_verification_sha256': native_binding.parameter_verification_sha256},
               'generation_contract': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 512,
                                       'max_time': 300, 'use_cache': True},
               'parameter_layout_sha256': _digest([(name, list(value.shape), str(value.dtype), str(value.device))
                                                   for name, value in model.named_parameters()]),
               'placement': {k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
               'devices': devices}
    return _digest(details), details


def generate_caption_batch(model, processor, batch, *, admission, topology_fingerprint, native_binding):
    """Run only an admitted native batch. All failures retain elapsed work."""
    started = time.perf_counter()
    try:
        return _generate_caption_batch(model, processor, batch, admission=admission,
            topology_fingerprint=topology_fingerprint, native_binding=native_binding,
            workflow_started=started)
    except BatchExecutionError:
        raise
    except Exception as error:
        elapsed = time.perf_counter() - started
        record = {'schema_version': 'bir_caption_batch_v1', 'complete': False,
                  'status': 'failed_preflight', 'rows': [], 'gpu_memory': [],
                  'error_type': type(error).__name__, 'error': str(error),
                  'admission_and_integrity_seconds': elapsed,
                  'runner_total_seconds': elapsed,
                  'collation_seconds': batch.collation_seconds,
                  'total_work_including_collation_seconds': elapsed + batch.collation_seconds}
        if isinstance(error, (BatchAdmissionError, BatchCompatibilityError)):
            error.record = record
            raise
        wrapped = BatchAdmissionError('Native batch preflight failed; retain the attached record')
        wrapped.record = record
        raise wrapped from error


def _generate_caption_batch(model, processor, batch, *, admission, topology_fingerprint,
                            native_binding, workflow_started):
    import torch
    from .placement import verify_residency
    if not isinstance(admission, MeasuredBatchAdmission):
        raise BatchAdmissionError('Batch generation is disabled without measured admission')
    if importlib.metadata.version('transformers') != '5.8.1' or str(torch.__version__).split('+')[0] != '2.11.0' or torch.version.cuda != '12.8':
        raise BatchAdmissionError('The native batch runner requires the pinned annotation environment')
    if not torch.cuda.is_available():
        raise BatchAdmissionError('CUDA is required for native batch generation')
    verify_residency(model)
    fingerprint, runtime = runtime_fingerprint(model, processor, batch,
        topology_fingerprint=topology_fingerprint, native_binding=native_binding)
    totals, free, allocated, reserved = [], [], [], []
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
        available, total = torch.cuda.mem_get_info(i)
        totals.append(int(total)); free.append(int(available))
        allocated.append(torch.cuda.memory_allocated(i)); reserved.append(torch.cuda.memory_reserved(i))
    admission.require(batch, runtime_fingerprint=fingerprint, total_bytes=totals,
                      free_bytes=free, allocated_bytes=allocated, reserved_bytes=reserved)
    verify_batch_integrity(batch)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    device = next(model.parameters()).device
    started = time.perf_counter()
    record = {'schema_version': 'bir_caption_batch_v1', 'rows': [], 'complete': False,
              'compatibility_signature': batch.compatibility_signature,
              'admission_evidence_id': admission.evidence_id, 'runtime': runtime,
              'collation_seconds': batch.collation_seconds,
              'admission_and_integrity_seconds': started - workflow_started, 'gpu_memory': [],
              'baseline_gpu_memory': [{'logical_device': i, 'allocated_bytes': allocated[i],
                                       'reserved_bytes': reserved[i], 'free_bytes': free[i],
                                       'total_bytes': totals[i]} for i in range(len(totals))]}
    error = None
    try:
        inputs = {key: value.to(device) for key, value in batch.inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512,
                eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True, max_time=300)
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        record['rows'] = extract_batch_outputs(output, batch, processor.tokenizer)
        record['complete'] = all(row['status'] == 'complete' for row in record['rows'])
    except Exception as caught:
        error = caught
        record.update(error_type=type(caught).__name__, error=str(caught))
        record['rows'] = [{'request_id': route['request_id'], 'episode_id': route['episode_id'],
                           'event_id': route['event_id'], 'output_row': route['output_row'],
                           'status': 'failed_batch_execution', 'caption': ''} for route in batch.routing]
    finally:
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(i)
                record['gpu_memory'].append({'logical_device': i,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(i)})
            except Exception as caught:
                record['gpu_memory'].append({'logical_device': i, 'measurement_error': str(caught)})
                record['complete'] = False
        record['generation_seconds'] = time.perf_counter() - started
        record['runner_total_seconds'] = time.perf_counter() - workflow_started
        record['total_work_including_collation_seconds'] = record['runner_total_seconds'] + batch.collation_seconds
    if error is not None:
        raise BatchExecutionError(record) from error
    record['exceeded_recorded_pilot_peak'] = any(
        'measurement_error' in memory or memory['peak_reserved_bytes'] > pilot
        for memory, pilot in zip(record['gpu_memory'], admission.peak_reserved_bytes, strict=True))
    record['max_generated_tokens_observed'] = max((row.get('generated_tokens', 0) for row in record['rows']), default=0)
    record['measured_envelope_respected'] = all(
        'measurement_error' not in memory and memory['peak_reserved_bytes'] + admission.safety_headroom_bytes <= total
        for memory, total in zip(record['gpu_memory'], totals, strict=True))
    if not record['measured_envelope_respected'] or record['exceeded_recorded_pilot_peak']:
        record['complete'] = False
    return record
