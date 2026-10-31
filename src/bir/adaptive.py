"""Experimental caption spatial budgets. Native execution does not import this.

A compact preset changes only the processor's total pixel budget per temporal
block. It preserves all selected frames and source clocks. This is not a quality
guarantee. Use independent references and account for every attempted pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import time
from unittest.mock import patch

TRANSFORMERS_VERSION = '5.8.1'
COMPACT_TOTAL_PIXELS = 40 * 131072


@dataclass(frozen=True)
class SpatialPreset:
    name: str = 'compact_block_5242880'
    max_total_pixels_per_block: int = COMPACT_TOTAL_PIXELS

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError('The spatial preset requires a name')
        value = self.max_total_pixels_per_block
        if isinstance(value, bool) or not isinstance(value, int) or value < 1024:
            raise ValueError('The total pixel budget must be an integer of at least 1024')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name)


def _clock(metadata):
    indices = _field(metadata, 'frames_indices')
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    return {'total_num_frames': int(_field(metadata, 'total_num_frames')),
            'fps': float(_field(metadata, 'fps')), 'frames_indices': list(indices)}


def _frame_receipt(videos, metadata):
    if not isinstance(videos, (list, tuple)) or not videos or len(videos) != len(metadata):
        raise ValueError('The experiment requires ordered native temporal blocks and their clocks')
    blocks = []
    for frames, clock_value in zip(videos, metadata, strict=True):
        if not isinstance(frames, (list, tuple)) or not frames:
            raise ValueError('Each native block must contain its selected PIL images')
        clock = _clock(clock_value)
        if clock['total_num_frames'] != len(frames) or len(clock['frames_indices']) != len(frames):
            raise ValueError('Frame count and source timestamps disagree')
        if not math.isfinite(clock['fps']) or clock['fps'] <= 0:
            raise ValueError('The native source clock requires a positive FPS')
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in clock['frames_indices']):
            raise ValueError('Source frame indices must be nonnegative integers')
        if clock['frames_indices'] != sorted(clock['frames_indices']):
            raise ValueError('Source frame timestamps changed order')
        if any(not hasattr(frame, 'tobytes') or not hasattr(frame, 'size') or frame.mode != 'RGB' for frame in frames):
            raise ValueError('The experiment requires native RGB PIL frames')
        sizes = [list(frame.size) for frame in frames]
        if any(size != sizes[0] for size in sizes):
            raise ValueError('A native temporal block must use one spatial geometry')
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        blocks.append({'clock': clock, 'frame_sizes': sizes, 'frame_sha256': hashes})
    return blocks


class CaptionSpatialProcessor:
    """Proxy that supplies the supported flat size override per native call.

    All methods and attributes other than __call__ use the same processor.
    The original processor and native frame objects are not changed. Native
    callers pass keyword arguments, ordered RGB PIL blocks, and video metadata.
    """
    def __init__(self, processor, preset=SpatialPreset(), *, audit_sink=None):
        if importlib.metadata.version('transformers') != TRANSFORMERS_VERSION:
            raise RuntimeError('The spatial override is pinned to Transformers 5.8.1')
        self.processor = processor
        self.preset = preset
        self.audit_sink = audit_sink
        self.records = []

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
        if args:
            raise ValueError('Use the unchanged native keyword processor call')
        if kwargs.get('images') is not None:
            raise ValueError('This caption experiment accepts video blocks only')
        if 'size' in kwargs or ('videos_kwargs' in kwargs and 'size' in kwargs['videos_kwargs']):
            raise ValueError('A caller spatial override would conflict with the declared preset')
        if 'videos_kwargs' in kwargs:
            raise ValueError('Use native flat video kwargs; mixing nested kwargs can discard source clocks')
        if kwargs.get('do_sample_frames') is not False:
            raise ValueError('The spatial experiment requires do_sample_frames=False')
        video_processor = self.processor.video_processor
        if video_processor.__class__.__name__ != 'Qwen3VLVideoProcessor':
            raise ValueError('The experiment requires the pinned Qwen3VLVideoProcessor')
        for key in ('patch_size', 'temporal_patch_size', 'merge_size', 'do_resize'):
            if key in kwargs and kwargs[key] != getattr(video_processor, key):
                raise ValueError('The experiment cannot change the native ' + key)
        if not video_processor.do_resize:
            raise ValueError('The native video processor must have resizing enabled')
        native_size = video_processor.size
        minimum = int(_field(native_size, 'shortest_edge'))
        maximum = int(_field(native_size, 'longest_edge'))
        compact_maximum = self.preset.max_total_pixels_per_block
        if not minimum <= compact_maximum <= maximum:
            raise ValueError('The compact budget must fit within the native processor bounds')
        videos = kwargs.get('videos')
        metadata = kwargs.get('video_metadata')
        if metadata is None:
            raise ValueError('Explicit native video metadata is required')
        started = time.perf_counter()
        before = _frame_receipt(videos, metadata)
        frame_digest = _digest(before)
        text_digest = _digest(kwargs.get('text'))
        patch, merge, temporal = video_processor.patch_size, video_processor.merge_size, video_processor.temporal_patch_size
        predictions = []
        for block in before:
            width, height = block['frame_sizes'][0]
            count = block['clock']['total_num_frames']
            if count % temporal:
                raise ValueError('Native temporal pair padding must already be explicit')
            native_h, native_w = smart_resize(count, height, width, temporal_factor=temporal,
                factor=patch * merge, min_pixels=minimum, max_pixels=maximum)
            compact_h, compact_w = smart_resize(count, height, width, temporal_factor=temporal,
                factor=patch * merge, min_pixels=minimum, max_pixels=compact_maximum)
            predictions.append({'native_grid_thw_from_geometry': [count // temporal, native_h // patch, native_w // patch],
                                'compact_grid_thw_from_geometry': [count // temporal, compact_h // patch, compact_w // patch]})
        compact_size = {'shortest_edge': minimum, 'longest_edge': compact_maximum}
        # In 5.8.1 a nested videos_kwargs dict suppresses flat video kwargs.
        # Native calls are flat, so keep them flat to preserve the source clock.
        call_kwargs = dict(kwargs, size=compact_size)
        result = self.processor(**call_kwargs)
        after = _frame_receipt(videos, metadata)
        if _digest(after) != frame_digest or _digest(kwargs.get('text')) != text_digest:
            raise RuntimeError('The processor changed source frames, source timestamps, or the input prompt')
        grids = result['video_grid_thw'].tolist()
        if grids != [item['compact_grid_thw_from_geometry'] for item in predictions]:
            raise RuntimeError('Observed video grids disagree with the pinned spatial-only contract')
        output_metadata = result.get('video_metadata')
        if output_metadata is not None and [_clock(item) for item in output_metadata] != [b['clock'] for b in before]:
            raise RuntimeError('The processor changed output video timestamps')
        block_audits = []
        for expected, actual in zip(predictions, grids, strict=True):
            native = expected['native_grid_thw_from_geometry']
            block_audits.append({**expected, 'actual_compact_grid_thw': actual,
                'native_visual_tokens_from_geometry': math.prod(native) // merge ** 2,
                'actual_compact_visual_tokens': math.prod(actual) // merge ** 2,
                'realized_spatial_area_fraction': actual[1] * actual[2] / (native[1] * native[2]),
                'spatial_no_op': actual == native})
        record = {'schema_version': 'bir_caption_spatial_audit_v1', 'preset': self.preset.name,
                  'budget_units': 'total_pixels_per_temporal_block',
                  'native_processor_size': {'shortest_edge': minimum, 'longest_edge': maximum},
                  'compact_processor_size': compact_size,
                  'frame_and_clock_sha256_before': frame_digest, 'frame_and_clock_sha256_after': _digest(after),
                  'prompt_sha256': text_digest, 'blocks': block_audits,
                  'temporal_frame_selection_changed': False, 'processor_elapsed_s': time.perf_counter() - started,
                  'native_grid_is_geometry_prediction_not_a_measured_native_pass': True}
        if self.audit_sink is not None:
            self.audit_sink(record)
        else:
            self.records.append(record)
        return result


class InvalidCompactOutput(RuntimeError):
    """The generation stopped without a complete native-format caption."""


def fallback_decision(text, *, generation_complete=True):
    """A format gate only. It does not assess caption truth or quality."""
    if not generation_complete:
        return {'fallback': True, 'reason': 'incomplete_generation'}
    if not isinstance(text, str) or not text.strip():
        return {'fallback': True, 'reason': 'empty_or_invalid_caption'}
    return {'fallback': False, 'reason': 'native_format_valid'}


def run_compact_then_native(compact_call, native_call, *, synchronize=None,
                            invalid_output_errors=(InvalidCompactOutput,)):
    """Run a format-only fallback rule and retain every attempted pass cost.

    Callbacks return caption strings. Only declared invalid-output exceptions
    trigger fallback. Other failures stop the work item. Scheduler allocation
    cost is external; these durations are measured application work only.
    """
    synchronize = synchronize or (lambda: None)
    attempts = []

    def invoke(name, callback):
        started = time.perf_counter()
        record = {'category': name, 'status': 'running'}
        try:
            synchronize()
            text = callback()
            record.update(status='complete', text=text)
        except invalid_output_errors as error:
            record.update(status='invalid_output', text=None, error_type=type(error).__name__, error=str(error))
        except Exception as error:
            record.update(status='failed', text=None, error_type=type(error).__name__, error=str(error))
        finally:
            try:
                synchronize()
            except Exception as error:
                record.update(status='failed', synchronization_error_type=type(error).__name__,
                              synchronization_error=str(error))
            record['wall_seconds'] = time.perf_counter() - started
            attempts.append(record)
        return record

    compact = invoke('compact', compact_call)
    if compact['status'] == 'failed':
        return {'status': 'failed', 'caption': None, 'chosen_pass': None, 'attempts': attempts,
                'gate': None, 'application_work_seconds': sum(x['wall_seconds'] for x in attempts),
                'quality_assessed': False}
    started = time.perf_counter()
    decision = fallback_decision(compact['text'], generation_complete=compact['status'] == 'complete')
    attempts.append({'category': 'gate', 'status': 'complete', 'wall_seconds': time.perf_counter() - started})
    chosen, name = compact, 'compact'
    if decision['fallback']:
        chosen, name = invoke('native_fallback', native_call), 'native_fallback'
    valid = chosen['status'] == 'complete' and not fallback_decision(chosen['text'])['fallback']
    return {'status': 'complete' if valid else 'failed', 'caption': chosen['text'] if valid else None,
            'chosen_pass': name if valid else None, 'attempts': attempts, 'gate': decision,
            'application_work_seconds': sum(x['wall_seconds'] for x in attempts),
            'quality_assessed': False}


def run_reference_interval(backend, prepared, interval, *, episode_id,
                           reference_index, policy, freeze_sha256,
                           synchronize=None):
    """Run one explicit fixed-reference caption experiment, without TempSeg.

    The caller must measure shared preparation and load the verified native
    Captioner before calling this function. The native pipeline still owns
    frame selection, context, clocks, prompts, generation, and parsing.
    Only the compact pass receives the spatial processor proxy.
    """
    if policy not in ('native', 'compact', 'adaptive'):
        raise ValueError('Unknown fixed-reference caption policy')
    if prepared.profile not in ('Human', 'Human actions') or set(prepared.videos) != {'main'}:
        raise ValueError('This frozen experiment supports human single-view inputs only')
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError('A stable episode identifier is required')
    if type(reference_index) is not int or reference_index < 0:
        raise ValueError('A zero-based reference interval index is required')
    if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64 or any(c not in '0123456789abcdef' for c in freeze_sha256):
        raise ValueError('The experiment requires a SHA256 of its frozen contract')
    start, end = interval['start_s'], interval['end_s']
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not (
            0 <= start < end <= prepared.metadata['duration_s'] + 0.001):
        raise ValueError('Reference interval is outside the source video clock')
    if backend.session.kind != 'captioner' or backend.session.native.model is None:
        raise RuntimeError('Load and time the verified native Captioner before interval policies')
    if synchronize is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the fixed-reference experiment')
        def synchronize():
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
    record = {'schema_version': 'bir_reference_caption_v1', 'episode_id': episode_id,
              'reference_index': reference_index, 'policy': policy, 'freeze_sha256': freeze_sha256,
              'start_s': start, 'end_s': end, 'status': 'failed', 'caption': None,
              'interval_provenance': 'fixed_independent_reference_interval',
              'input_sha256': dict(prepared.metadata['input_sha256']),
              'attempts': [], 'gate': None, 'spatial_audits': [],
              'caption_input_audits': [], 'generation_events': [],
              'quality_assessed': False, 'request_batch_size': 1}
    # This is an explicitly supplied reference interval, never a claimed
    # TempSeg prediction or a substitute for the native end-to-end gate.
    segments = [{'start_s': start, 'end_s': end,
                 'metadata': {'segmentation_provenance': 'fixed_independent_reference_interval'}}]
    active_pass = None
    native_event = backend.models.runtime_event
    def capture(event, **values):
        if event in ('generation_start', 'generation_end'):
            record['generation_events'].append({'pass': active_pass, 'event': event, **values})
        native_event(event, **values)

    class SpatialSession:
        def __init__(self, category):
            self.processor = None
            self.category = category

        def load(self, kind):
            if kind != 'captioner':
                raise RuntimeError('The spatial experiment cannot load another model stage')
            model, processor = backend.session.load(kind)
            if self.processor is None:
                def receipt(audit):
                    record['spatial_audits'].append({'pass': self.category, **audit})
                self.processor = CaptionSpatialProcessor(processor, audit_sink=receipt)
            return model, self.processor

    def invoke(category, compact):
        nonlocal active_pass
        active_pass = category
        begun = time.perf_counter()
        attempt = {'category': category, 'status': 'running'}
        record['attempts'].append(attempt)
        try:
            synchronize()
            session = SpatialSession(category) if compact else backend.session
            rows, audit = backend.pipeline.caption_single(prepared.videos, segments, prepared.profile,
                session, prepared.job.get('instruction', ''), backend.progress)
            record['caption_input_audits'].append({'pass': category, 'records': audit})
            if len(rows) != 1:
                raise RuntimeError('The native pipeline changed fixed-reference output cardinality')
            row = rows[0]
            if row['start_s'] != start or row['end_s'] != end:
                raise RuntimeError('The native pipeline changed the fixed reference interval')
            if row['status'] != 'complete':
                attempt.update(status=row['status'], text=None)
            elif fallback_decision(row.get('caption'))['fallback']:
                attempt.update(status='invalid_output', text=None)
            else:
                attempt.update(status='complete', text=row['caption'])
        except Exception as error:
            # The upstream helper uses the same exception for context overflow
            # and incomplete text. Only the exact incomplete-output case is a
            # format-gate signal. An OOM or context failure never triggers rerun.
            incomplete = isinstance(error, backend.models.InputErrorForModel) and str(error) == (
                'The model did not return a complete output for this input.')
            attempt.update(status='invalid_output' if incomplete else 'failed', text=None,
                           error_type=type(error).__name__, error=str(error))
        finally:
            try:
                synchronize()
            except Exception as error:
                attempt.update(status='failed', synchronization_error_type=type(error).__name__,
                               synchronization_error=str(error))
            attempt['wall_seconds'] = time.perf_counter() - begun
        return attempt

    begun = time.perf_counter()
    with backend._execution(), patch.object(backend.models, 'runtime_event', side_effect=capture):
        chosen = invoke('native' if policy == 'native' else 'compact', compact=policy != 'native')
        if policy == 'adaptive' and chosen['status'] in ('complete', 'invalid_output'):
            gate_start = time.perf_counter()
            record['gate'] = fallback_decision(chosen['text'], generation_complete=chosen['status'] == 'complete')
            record['attempts'].append({'category': 'gate', 'status': 'complete',
                                       'wall_seconds': time.perf_counter() - gate_start})
            if record['gate']['fallback']:
                chosen = invoke('native_fallback', compact=False)
        record['status'] = chosen['status'] if chosen['status'] != 'invalid_output' else 'failed'
        if chosen['status'] == 'complete':
            record['caption'] = chosen['text']
            record['chosen_pass'] = chosen['category']
        else:
            record['chosen_pass'] = None
    record['policy_wall_seconds'] = time.perf_counter() - begun
    record['application_work_seconds'] = sum(attempt['wall_seconds'] for attempt in record['attempts'])
    return record
