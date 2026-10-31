"""Explicit, disabled-by-default collection of unchanged native caption inputs.

No admission is supplied with this module. The normal NativeBackend and CLI
never call it. The caller must provide completed GPU calibration evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from unittest.mock import patch

from .batching import (BatchAdmissionError, BatchCompatibilityError, BatchExecutionError,
    CaptionRequest, MeasuredBatchAdmission, _describe, _digest, bind_verified_native,
    collate_caption_requests, generate_caption_batch, input_tensor_audit)
from .native import MODEL_REVISIONS, UPSTREAM_COMMIT, _NATIVE_LOCK


@dataclass(frozen=True)
class DeferredCaption:
    """An internal handle that must never reach JSON output."""
    token: str


def _synchronize():
    import torch
    for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)


def _gpu_peaks():
    import torch
    return [{'logical_device': device, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
             'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            for device in range(torch.cuda.device_count())]


class CaptionCollector:
    """Keep bounded CPU tensors; dispatch only exactly admitted groups.

    Half the tensor byte bound is reserved for the concatenated batch. This
    bounds original-plus-collated tensor payload. It is not a host-RAM ceiling.
    """
    def __init__(self, original_generate, *, session, episode_id, request_id,
                 profile, contract_id, admissions, topology_fingerprint,
                 max_pending_requests, max_prepared_bytes, synchronize=None):
        for name, value in [('max_pending_requests', max_pending_requests),
                            ('max_prepared_bytes', max_prepared_bytes)]:
            if type(value) is not int or value < 2:
                raise ValueError(name + ' must be an explicit integer of at least two')
        for value in (episode_id, request_id, profile, contract_id, topology_fingerprint):
            if not isinstance(value, str) or not value:
                raise ValueError('Collector input, contract, and topology identities are required')
        if profile not in ('human', 'Human', 'Human actions'):
            raise BatchCompatibilityError('This collector pilot supports only the human single-view event convention')
        admissions = tuple(admissions)
        if not admissions or any(not isinstance(a, MeasuredBatchAdmission) or
                                 not 2 <= a.batch_size <= max_pending_requests for a in admissions):
            raise BatchAdmissionError('Collector requires explicit measured batch admissions within its request bound')
        self.admissions = {(a.compatibility_signature, a.batch_size): a for a in admissions}
        if len(self.admissions) != len(admissions):
            raise BatchAdmissionError('Duplicate admission shapes are ambiguous')
        self.original, self.session = original_generate, session
        self.episode_id, self.request_id, self.profile, self.contract_id = episode_id, request_id, profile, contract_id
        self.topology = topology_fingerprint
        self.max_requests, self.max_bytes = max_pending_requests, max_prepared_bytes
        self.synchronize = synchronize or _synchronize
        self.pending, self.results, self.captures = [], {}, []
        self.source_bytes = 0
        self.model = self.processor = self.binding = None
        self.audit = dict(schema_version='bir_caption_collector_v1', batch_runs=[], single_runs=[],
                          request_routes=[], pending_source_peak_bytes=0,
                          max_retained_tensor_bytes=max_prepared_bytes,
                          max_pending_requests=max_pending_requests, peak_snapshots=[])

    def __call__(self, model, processor, inputs, budget):
        if budget != 512 or self.session.kind != 'captioner':
            raise RuntimeError('The collector can intercept only native Captioner generation')
        if self.model is None:
            self.model, self.processor = model, processor
            self.audit['peak_snapshots'].append(_gpu_peaks())
        if self.model is not model or self.processor is not processor:
            raise RuntimeError('A deferred caption changed model or processor identity')
        ordinal = len(self.captures)
        captured = input_tensor_audit(inputs)
        token = _digest([self.request_id, ordinal, captured['input_tensor_digest']])
        handle = DeferredCaption(token)
        request = CaptionRequest(token, self.episode_id, f'deferred_{ordinal:04d}',
                                 self.profile, self.contract_id, dict(inputs), budget)
        self.captures.append(dict(handle=token, ordinal=ordinal, **captured))
        try:
            values, description = _describe(request, processor.video_token_id, processor.video_processor.merge_size)
        except BatchCompatibilityError as error:
            item = dict(handle=handle, request=request, audit=captured, bytes=0, signature=None)
            self._single(item, 'unsupported_batch_shape: ' + str(error))
            return handle
        size = sum(t.numel() * t.element_size() for t in values.values())
        item = dict(handle=handle, request=request, audit=captured, bytes=size, signature=_digest(description))
        possible = any(signature == item['signature'] for signature, _ in self.admissions)
        if not possible or size > self.max_bytes // 2:
            self._single(item, 'no_measured_shape_admission' if not possible else 'input_exceeds_collection_bound')
            return handle
        while self.pending and (len(self.pending) >= self.max_requests or
                                self.source_bytes + size > self.max_bytes // 2):
            self._flush_oldest()
        self.pending.append(item)
        self.source_bytes += size
        self.audit['pending_source_peak_bytes'] = max(self.audit['pending_source_peak_bytes'], self.source_bytes)
        self._flush_ready(item['signature'])
        return handle

    def _check(self, item):
        if input_tensor_audit(item['request'].inputs) != item['audit']:
            raise BatchCompatibilityError('Native inputs changed while awaiting caption execution')

    def _single(self, item, reason):
        begun = time.perf_counter()
        record = dict(request_id=item['request'].request_id, reason=reason, status='running',
                      input_tensor_digest=item['audit']['input_tensor_digest'])
        self.audit['single_runs'].append(record)
        try:
            self.synchronize()
            self._check(item)
            text = self.original(self.model, self.processor, item['request'].inputs, 512)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError('The original native generator returned an invalid caption')
            self.results[item['handle'].token] = text
            record['status'] = 'complete'
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                self.synchronize()
                self.audit['peak_snapshots'].append(_gpu_peaks())
            except Exception as error:
                record.update(status='failed', synchronization_error=str(error))
                raise
            finally:
                record['wall_seconds'] = time.perf_counter() - begun

    def _flush_ready(self, signature):
        group = [item for item in self.pending if item['signature'] == signature]
        sizes = sorted((size for sig, size in self.admissions if sig == signature and size <= len(group)), reverse=True)
        if not sizes:
            return False
        selected = group[:sizes[0]]
        for item in selected: self._check(item)
        batch = collate_caption_requests([item['request'] for item in selected],
            video_token_id=self.processor.video_token_id, max_batch_size=self.max_requests,
            max_prepared_bytes=self.max_bytes)
        # The source half-bound ensures other waiting groups plus this batch's
        # concatenation remain within max_prepared_bytes.
        if self.source_bytes + batch.collated_tensor_bytes > self.max_bytes:
            raise BatchCompatibilityError('Collector retained tensor bytes exceed the declared bound')
        if self.binding is None:
            self.binding = bind_verified_native(self.model, self.processor, source_revision=UPSTREAM_COMMIT,
                model_revision=MODEL_REVISIONS['captioner'], processor_revision=MODEL_REVISIONS['captioner'],
                execution_contract_id=self.contract_id, model_audit=self.session.audit['captioner'])
        self.audit['peak_snapshots'].append(_gpu_peaks())
        try:
            result = generate_caption_batch(self.model, self.processor, batch,
                admission=self.admissions[(signature, sizes[0])],
                topology_fingerprint=self.topology, native_binding=self.binding)
        except Exception as error:
            if hasattr(error, 'record'): self.audit['batch_runs'].append(error.record)
            raise
        self.audit['batch_runs'].append(result)
        if result.get('complete') is not True:
            raise BatchExecutionError(result)
        expected = [(item['request'].request_id, item['request'].episode_id, item['request'].event_id)
                    for item in selected]
        if ([(row['request_id'], row['episode_id'], row['event_id']) for row in result['rows']] != expected or
                any(row.get('status') != 'complete' or not isinstance(row.get('caption'), str) or
                    not row['caption'].strip() for row in result['rows'])):
            raise BatchCompatibilityError('Batch outputs changed the deferred request order or completion status')
        for item, row in zip(selected, result['rows'], strict=True):
            self.results[item['handle'].token] = row['caption']
            self.pending.remove(item)
            self.source_bytes -= item['bytes']
        return True

    def _flush_oldest(self):
        item = self.pending[0]
        if not self._flush_ready(item['signature']):
            self._single(item, 'unmatched_bounded_group')
            self.pending.pop(0)
            self.source_bytes -= item['bytes']

    def finish(self, annotation):
        while self.pending: self._flush_oldest()
        audits = annotation['pipeline']['caption_inputs']
        handles = [(index, row['caption']) for index, row in enumerate(annotation['segments'])
                   if isinstance(row.get('caption'), DeferredCaption)]
        if len(handles) != len(self.captures) or len(audits) != len(handles):
            raise BatchCompatibilityError('Native caption audit and deferred output cardinality differ')
        for ordinal, ((index, handle), captured, audit) in enumerate(zip(handles, self.captures, audits, strict=True)):
            if handle.token != captured['handle'] or audit.get('event_id') != f'event_{index:04d}':
                raise BatchCompatibilityError('Native event audit and deferred request routing differ')
            if handle.token not in self.results:
                raise BatchCompatibilityError('A deferred caption has no completed generation result')
            annotation['segments'][index]['caption'] = self.results[handle.token]
            self.audit['request_routes'].append(dict(episode_id=self.episode_id,
                parent_request_id=self.request_id, caption_request_id=handle.token,
                native_event_id=audit['event_id'], generation_ordinal=ordinal,
                input_tensor_digest=captured['input_tensor_digest']))
        return annotation


def caption_with_microbatch(backend, prepared, segmentation, *, episode_id, request_id,
                           contract_id, admissions, topology_fingerprint,
                           max_pending_requests, max_prepared_bytes):
    """Opt-in caption stage. Resolve every handle before returning JSON data.

    Unmatched requests use the exact original batch-one helper. A failed batch
    is not retried as singles. The caller owns backend lifetime and job budget.
    """
    begun = time.perf_counter()
    if prepared.profile not in ('Human', 'Human actions') or set(prepared.videos) != {'main'}:
        raise BatchCompatibilityError('This collector pilot supports only prepared human single-view inputs')
    collector = CaptionCollector(backend.pipeline.generate, session=backend.session,
        episode_id=episode_id, request_id=request_id, profile=prepared.profile,
        contract_id=contract_id, admissions=admissions, topology_fingerprint=topology_fingerprint,
        max_pending_requests=max_pending_requests, max_prepared_bytes=max_prepared_bytes)
    try:
        with _NATIVE_LOCK, patch.object(backend.pipeline, 'generate', side_effect=collector):
            annotation = backend.caption(prepared, segmentation)
            annotation = collector.finish(annotation)
        elapsed = time.perf_counter() - begun
        annotation['engine'].update(mode='placement_adapted_native_microbatch_collector',
            request_batch_size=max((len(r['rows']) for r in collector.audit['batch_runs']), default=1),
            caption_collector=collector.audit)
        annotation['engine']['stage_elapsed_s']['captioner'] = elapsed
        annotation['elapsed_s'] = segmentation['elapsed_s'] + elapsed
        collector.audit['complete_stage_seconds'] = elapsed
        collector.audit['complete'] = True
        return annotation
    except Exception as error:
        collector.audit.update(complete=False, complete_stage_seconds=time.perf_counter() - begun,
                               error_type=type(error).__name__, error=str(error))
        error.collector_record = collector.audit
        raise
