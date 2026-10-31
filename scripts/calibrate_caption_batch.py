#!/usr/bin/env python3
"""Bounded, opt-in batch-two calibration after Gate C. Never submits jobs.

The 512-token stress pass disables EOS stopping. It is a labeled memory test,
not native output evidence. No production admission is written unless separate
native agreement, full stress completion, and measured headroom all pass.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

from bir.batching import (BatchCompatibilityError, CaptionRequest, MeasuredBatchAdmission,
    _describe, _digest, bind_verified_native, collate_caption_requests, extract_batch_outputs,
    input_tensor_audit, runtime_fingerprint, verify_batch_integrity)
from bir.cli import atomic_json
from bir.batching_contract import native_batching_contract, native_batching_contract_id
from bir.caption_batching import CaptionCollector
from bir.manifest import file_sha256, load_manifest
from bir.native import MODEL_REVISIONS, UPSTREAM_COMMIT, NativeBackend

try:
    from .adaptive_benchmark import captioner_placement_signature, gate_placement
    from .evaluate_adaptive import load_freeze, read_json
    from .native_benchmark import EXPECTED, _guard_allocation, _hardware, verify_gate_c
except ImportError:
    from adaptive_benchmark import captioner_placement_signature, gate_placement
    from evaluate_adaptive import load_freeze, read_json
    from native_benchmark import EXPECTED, _guard_allocation, _hardware, verify_gate_c


class CompatiblePairSearch:
    """Retain at most one candidate per shape, then the first distinct pair."""
    def __init__(self, *, video_token_id, max_prepared_bytes, max_candidates):
        if type(max_prepared_bytes) is not int or max_prepared_bytes < 2 or type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('Pair search requires positive explicit tensor and candidate bounds')
        self.video_token_id, self.max_bytes, self.max_candidates = video_token_id, max_prepared_bytes, max_candidates
        self.candidates, self.selected, self.source_bytes = {}, None, 0
        self.peak_retained_source_bytes = 0
        self.observations = []

    def offer(self, request):
        audit = input_tensor_audit(request.inputs)
        observation = dict(request_id=request.request_id, episode_id=request.episode_id,
                           event_id=request.event_id, **audit)
        self.observations.append(observation)
        if self.selected is not None:
            observation['selection'] = 'not_retained_after_pair_found'
            return
        try:
            values, description = _describe(request, self.video_token_id, 2)
        except BatchCompatibilityError as error:
            observation.update(selection='unsupported_batch_shape', reason=str(error))
            return
        signature = _digest(description)
        candidate_key = (request.episode_id, signature)
        size = sum(t.numel() * t.element_size() for t in values.values())
        observation.update(compatibility_signature=signature, source_tensor_bytes=size)
        if 4 * size > self.max_bytes:
            observation['selection'] = 'pair_exceeds_prepared_byte_bound'
            return
        if candidate_key in self.candidates:
            previous, previous_size, previous_digest = self.candidates[candidate_key]
            if request.request_id != previous.request_id and audit['input_tensor_digest'] != previous_digest:
                self.selected = (previous, request)
                self.candidates.clear()
                self.source_bytes = previous_size + size
                observation['selection'] = 'first_exact_distinct_pair'
            else:
                observation['selection'] = 'duplicate_input_not_selected'
        else:
            while self.candidates and (len(self.candidates) >= self.max_candidates or
                                      self.source_bytes + size > self.max_bytes // 2):
                key = next(iter(self.candidates))
                self.source_bytes -= self.candidates.pop(key)[1]
            self.candidates[candidate_key] = (request, size, audit['input_tensor_digest'])
            self.source_bytes += size
            observation['selection'] = 'bounded_candidate'
        self.peak_retained_source_bytes = max(self.peak_retained_source_bytes, self.source_bytes)


def processor_signature(processor):
    return _digest({'class': processor.__class__.__name__, 'chat_template': processor.chat_template,
        'tokenizer': processor.tokenizer.backend_tokenizer.to_str(),
        'special_tokens': processor.tokenizer.special_tokens_map,
        'video_token_id': processor.video_token_id,
        'video_processor': processor.video_processor.to_dict(),
        'image_processor': processor.image_processor.to_dict()})


def generation_settings(model, processor):
    return dict(model_config_sha256=_digest(model.config.to_dict()),
                generation_config_sha256=_digest(model.generation_config.to_dict()),
                processor_sha256=processor_signature(processor))


def memory_stress(model, processor, batch):
    """Generate512 tokens with EOS disabled on a private configuration copy."""
    import torch
    before = generation_settings(model, processor)
    verify_batch_integrity(batch)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in batch.inputs.items()}
    try:
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=copy.deepcopy(model.generation_config),
                do_sample=False, num_beams=1, max_new_tokens=512, eos_token_id=None,
                pad_token_id=processor.tokenizer.pad_token_id, use_cache=True, max_time=300)
        output = output.detach().cpu()
        prefix_valid = torch.equal(output[:, :batch.input_tokens], batch.inputs['input_ids'])
        result = dict(complete=prefix_valid and list(output.shape) == [2, batch.input_tokens + 512],
            tokens_per_row=int(output.shape[1] - batch.input_tokens), prefix_valid=prefix_valid,
            output_token_sha256=hashlib.sha256(output.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            native_generation=False, changed_generation_settings={'eos_token_id': None}, quality_evidence=False)
    finally:
        if generation_settings(model, processor) != before:
            raise RuntimeError('Memory stress changed persistent model, processor, or generation settings')
    return dict(result, persistent_settings_unchanged=True, settings_sha256_before=before,
                settings_sha256_after=generation_settings(model, processor))


def capture_episode(backend, prepared, job, targets, processor, search, contract_id):
    """Execute only the real CPU planner/processor; never save fake captions."""
    placeholder_model = object()
    class ProcessorSession:
        def load(self, kind):
            if kind != 'captioner': raise RuntimeError('Capture cannot load TempSeg')
            return placeholder_model, processor
    captured = []
    def capture(model, proc, inputs, budget):
        if model is not placeholder_model or proc is not processor or budget != 512:
            raise RuntimeError('The native caption preparation contract changed')
        ordinal = len(captured)
        request = CaptionRequest(_digest([job.request_id, ordinal]), job.id,
            f'capture_{ordinal:04d}', prepared.profile, contract_id, dict(inputs))
        captured.append(dict(ordinal=ordinal, provisional_id=request.request_id))
        search.offer(request)
        return 'BIR_CAPTURE_ONLY_NO_MODEL_INFERENCE'
    segments = [{'start_s': row['start_s'], 'end_s': row['end_s'], 'metadata': {}} for row in targets]
    with backend._execution(), patch.object(backend.pipeline, 'generate', side_effect=capture):
        rows, audits = backend.pipeline.caption_single(prepared.videos, segments, prepared.profile,
            ProcessorSession(), prepared.job.get('instruction', ''), backend.progress)
    if len(audits) != len(captured):
        raise RuntimeError('Native capture audit cardinality changed')
    routes = {}
    for capture_row, audit in zip(captured, audits, strict=True):
        event_id = audit['event_id']
        if not event_id.startswith('event_') or not event_id[6:].isdigit():
            raise RuntimeError('Unsupported native human event ID')
        index = int(event_id[6:])
        if not 0 <= index < len(targets) or rows[index]['caption'] != 'BIR_CAPTURE_ONLY_NO_MODEL_INFERENCE':
            raise RuntimeError('Native capture event routing changed')
        routes[capture_row['provisional_id']] = dict(episode_id=job.id,
            reference_index=targets[index]['reference_index'], native_event_id=event_id,
            start_s=targets[index]['start_s'], end_s=targets[index]['end_s'],
            native_input_audit=audit)
    return routes


def memory_snapshot(torch):
    rows = []
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        rows.append(dict(logical_device=device, allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device), free_bytes=int(free), total_bytes=int(total)))
    return rows


def measured_pass(torch, records, category, callback):
    started = time.perf_counter()
    record = dict(category=category, status='running', baseline=memory_snapshot(torch), gpu_memory=[])
    records.append(record)
    for device in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(device)
    try:
        result = callback()
        record['status'] = 'complete'
        return result, record
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
                record['gpu_memory'].append(dict(logical_device=device,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        except Exception as error:
            record.update(status='failed', memory_measurement_error=str(error))
            raise
        finally:
            record['wall_seconds'] = time.perf_counter() - started


def admission_from_evidence(*, batch, fingerprint, evidence_id, native_agreement,
                            batch_record, stress_record, stress_complete, headroom_bytes):
    if native_agreement is not True or stress_complete is not True:
        raise ValueError('Native agreement and a completed full512-token stress pass are required')
    if type(headroom_bytes) is not int or headroom_bytes <= 0:
        raise ValueError('Calibration requires positive integer memory headroom')
    if batch_record['status'] != 'complete' or stress_record['status'] != 'complete':
        raise ValueError('Calibration passes must complete without memory measurement failures')
    records = [batch_record, stress_record]
    count = len(batch_record['baseline'])
    if not count or any(len(r['baseline']) != count or len(r['gpu_memory']) != count for r in records):
        raise ValueError('Every visible device needs both calibration passes')
    for record in records:
        if any(record[key][i]['logical_device'] != i for key in ('baseline', 'gpu_memory') for i in range(count)):
            raise ValueError('Calibration memory evidence changed logical-device order')
    if [r['total_bytes'] for r in batch_record['baseline']] != [r['total_bytes'] for r in stress_record['baseline']]:
        raise ValueError('Calibration device capacities changed between passes')
    allocated = tuple(min(r['baseline'][i]['allocated_bytes'] for r in records) for i in range(count))
    reserved = tuple(min(r['baseline'][i]['reserved_bytes'] for r in records) for i in range(count))
    peaks = tuple(max(r['gpu_memory'][i]['peak_allocated_bytes'] for r in records) for i in range(count))
    peak_reserved = tuple(max(r['gpu_memory'][i]['peak_reserved_bytes'] for r in records) for i in range(count))
    totals = [batch_record['baseline'][i]['total_bytes'] for i in range(count)]
    if any(peak + headroom_bytes > total for peak, total in zip(peak_reserved, totals, strict=True)):
        raise ValueError('Measured calibration has insufficient physical device headroom')
    return MeasuredBatchAdmission(evidence_id, batch.compatibility_signature, fingerprint, 2,
        allocated, reserved, peaks, peak_reserved, headroom_bytes, 512, True)


def calibration_sources():
    scripts = ('calibrate_caption_batch.py', 'adaptive_benchmark.py', 'evaluate_adaptive.py',
               'native_benchmark.py', 'compare_native.py')
    return {name: file_sha256(Path(__file__).with_name(name)) for name in scripts}


def verify_frozen_release(args, report, contract):
    current = native_batching_contract(model_lock=args.model_lock,
        captioner_headroom_gib=args.captioner_headroom_gib,
        max_prepared_bytes=args.max_prepared_mib * 1024**2)
    if current != contract or calibration_sources() != report['calibration_sources_sha256']:
        raise RuntimeError('Calibration implementation or model contract changed during execution')
    if (load_freeze(args.freeze)['freeze_sha256'] != report['freeze_sha256'] or
            verify_gate_c(args.gate_evidence) != report['gate_c']):
        raise RuntimeError('Calibration frozen input or Gate C evidence changed during execution')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gate-evidence', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, default=Path('manifests/adaptive-freeze.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, default=Path('configs/native-models.json'))
    parser.add_argument('--captioner-headroom-gib', type=float, default=4.0)
    parser.add_argument('--batch-headroom-gib', type=float, default=4.0)
    parser.add_argument('--max-prepared-mib', type=int, default=2048)
    parser.add_argument('--max-candidates', type=int, default=8)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Calibration output must be fresh; preserve earlier evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version='bir_caption_batch_calibration_v1', status='started',
                  production_batching_enabled=False, admission_written=False, passes=[])
    path = args.output_dir / 'calibration.json'
    atomic_json(path, report)
    started = time.perf_counter()
    backend = torch = model = runtime_processor = None
    active_pass = None
    try:
        _guard_allocation()
        report['gate_c'] = verify_gate_c(args.gate_evidence)
        gate_marker = read_json(args.gate_evidence)
        expected_placement = gate_placement(gate_marker, args.gate_evidence)
        if not math.isfinite(args.batch_headroom_gib) or args.batch_headroom_gib <= 0:
            raise ValueError('Calibration headroom must be positive and finite')
        if not 1 <= args.max_prepared_mib <= 2048 or not 1 <= args.max_candidates <= 8:
            raise ValueError('Calibration preparation is capped at2GiB retained tensors and8 shape candidates')
        environment = {name: importlib.metadata.version(name) for name in EXPECTED}
        if any(environment[k].split('+')[0] != v for k, v in EXPECTED.items()):
            raise RuntimeError('Calibration requires the pinned native environment')
        import torch as framework
        torch = framework
        if not torch.cuda.is_available() or not 1 <= torch.cuda.device_count() <= 4 or torch.version.cuda != '12.8':
            raise RuntimeError('Calibration requires granted GPUs and CUDA12.8')
        report.update(environment=environment, hardware=_hardware(torch))
        actual_hardware = [(d['name'], d['total_memory_bytes']) for d in report['hardware']['devices']]
        if actual_hardware != [(d['name'], d['total_memory_bytes']) for d in gate_marker['gpu_memory']]:
            raise ValueError('Calibration hardware differs from Gate C')
        topology = args.output_dir / 'topology.txt'
        with topology.open('w') as stream:
            for command in (['nvidia-smi', '-L'], ['nvidia-smi', 'topo', '-m']):
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=30)
        physical = dict(topology_file_sha256=file_sha256(topology), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            logical_uuids=[str(getattr(torch.cuda.get_device_properties(i), 'uuid', 'unavailable'))
                           for i in range(torch.cuda.device_count())])
        if 'unavailable' in physical['logical_uuids']:
            raise RuntimeError('Calibration requires physical UUID mapping for each granted logical GPU')
        report['topology_fingerprint'] = _digest(physical)
        report['topology_identity'] = physical
        freeze = load_freeze(args.freeze)
        contract = native_batching_contract(model_lock=args.model_lock,
            captioner_headroom_gib=args.captioner_headroom_gib,
            max_prepared_bytes=args.max_prepared_mib * 1024**2)
        contract_id = native_batching_contract_id(contract)
        report.update(execution_contract=contract, execution_contract_sha256=contract_id,
                      calibration_script_sha256=file_sha256(Path(__file__)),
                      calibration_sources_sha256=calibration_sources(), freeze_sha256=freeze['freeze_sha256'])
        jobs = load_manifest(args.manifest, contract=contract)
        development = {e['episode_id']: e for e in freeze['episodes'] if e['split'] == 'development'}
        if {j.id for j in jobs} != set(development) or not 1 <= len(jobs) <= 3:
            raise ValueError('Calibration accepts exactly the frozen development videos, never final videos')
        backend = NativeBackend(args.source_root, captioner_headroom_gib=args.captioner_headroom_gib)
        capture_started = time.perf_counter()
        processor = backend.models.processor('captioner', backend.models.snapshot('captioner'))
        capture_processor_signature = processor_signature(processor)
        search = CompatiblePairSearch(video_token_id=processor.video_token_id,
            max_prepared_bytes=args.max_prepared_mib * 1024**2, max_candidates=args.max_candidates)
        routes = {}
        prepared_episodes = {}
        for job in jobs:
            expected = development[job.id]
            if (job.profile != 'human' or dict(job.source_sha256) != {'main': expected['source_sha256']} or
                    hashlib.sha256(job.instruction.encode()).hexdigest() != expected['instruction_sha256']):
                raise ValueError('Calibration source or instruction differs from the frozen development input')
            prepared = backend.prepare(job.as_native_dict())
            if (prepared.metadata['input_sha256'] != dict(job.source_sha256) or
                    abs(prepared.metadata['duration_s'] - expected['duration_s']) > .001):
                raise ValueError('Calibration source changed during preparation')
            prepared_episodes[job.id] = prepared
            routes.update(capture_episode(backend, prepared, job, expected['targets'], processor, search, contract_id))
        report.update(capture_seconds=time.perf_counter() - capture_started,
            capture_mode='real_native_cpu_inputs_no_model_inference_no_annotations',
            capture_observations=search.observations, candidate_peak_source_bytes=search.peak_retained_source_bytes)
        if search.selected is None:
            report.update(status='inconclusive_no_compatible_pair', reason='No distinct exact pair fits the bounded native-input search')
            return 0
        selected = tuple(replace(request, event_id=routes[request.request_id]['native_event_id']) for request in search.selected)
        report['selected_routes'] = [dict(request_id=r.request_id, **routes[r.request_id]) for r in selected]
        batch = collate_caption_requests(selected, video_token_id=processor.video_token_id,
            max_batch_size=2, max_prepared_bytes=args.max_prepared_mib * 1024**2)
        report.update(compatibility_signature=batch.compatibility_signature,
                      input_routes=list(batch.routing), collation_seconds=batch.collation_seconds)
        with backend._execution():
            (model, runtime_processor), _ = measured_pass(torch, report['passes'], 'native_model_load',
                                                          lambda: backend.session.load('captioner'))
        if processor_signature(runtime_processor) != capture_processor_signature:
            raise RuntimeError('Capture processor differs from the verified runtime processor')
        if captioner_placement_signature(backend.session.residency_audit) != expected_placement:
            raise RuntimeError('Actual calibration placement differs from Gate C')
        report.update(model_verification=backend.session.audit, residency=backend.session.residency_audit,
                      placement=backend.session.placement_audit, processor_sha256=capture_processor_signature)
        binding = bind_verified_native(model, runtime_processor, source_revision=UPSTREAM_COMMIT,
            model_revision=MODEL_REVISIONS['captioner'], processor_revision=MODEL_REVISIONS['captioner'],
            execution_contract_id=contract_id, model_audit=backend.session.audit['captioner'])
        fingerprint, runtime = runtime_fingerprint(model, runtime_processor, batch,
            topology_fingerprint=report['topology_fingerprint'], native_binding=binding)
        report.update(runtime_fingerprint=fingerprint, runtime=runtime, generation_events=[])
        original_event = backend.models.runtime_event
        def event(name, **values):
            if name in ('generation_start', 'generation_end'):
                report['generation_events'].append(dict(pass_name=active_pass, event=name, **values))
            original_event(name, **values)
        device = next(model.parameters()).device
        def native_batch():
            verify_batch_integrity(batch)
            inputs = {key: value.to(device) for key, value in batch.inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512,
                    eos_token_id=runtime_processor.tokenizer.eos_token_id,
                    pad_token_id=runtime_processor.tokenizer.pad_token_id, use_cache=True, max_time=300)
            return extract_batch_outputs(output, batch, runtime_processor.tokenizer)
        singles = []
        verify_frozen_release(args, report, contract)
        with backend._execution(), patch.object(backend.models, 'runtime_event', side_effect=event):
            for index, request in enumerate(selected):
                active_pass = f'native_single_{index}'
                text, _ = measured_pass(torch, report['passes'], active_pass,
                    lambda request=request: backend.models.generate(model, runtime_processor, request.inputs, 512))
                singles.append(text)
                report['native_single_captions'] = list(singles)
                atomic_json(path, report)
            active_pass = 'native_batch_two'
            batch_rows, batch_record = measured_pass(torch, report['passes'], active_pass, native_batch)
            agreement = all(row['status'] == 'complete' for row in batch_rows) and [row['caption'] for row in batch_rows] == singles
            report.update(native_batch_rows=batch_rows, native_agreement_verified=agreement)
            atomic_json(path, report)
            headroom = math.ceil(args.batch_headroom_gib * 1024**3)
            native_has_headroom = all(memory['peak_reserved_bytes'] + headroom <= baseline['total_bytes']
                for memory, baseline in zip(batch_record['gpu_memory'], batch_record['baseline'], strict=True))
            if not agreement or not native_has_headroom:
                report.update(status='inconclusive_native_agreement_or_headroom_failed', stress_status='not_run_to_preserve_budget')
                return 0
            active_pass = 'non_native_512_token_memory_stress'
            stress_result, stress_record = measured_pass(torch, report['passes'], active_pass,
                lambda: memory_stress(model, runtime_processor, batch))
            report['memory_stress'] = stress_result
        evidence_id = _digest({key: report[key] for key in ('execution_contract_sha256', 'input_routes',
            'native_single_captions', 'native_batch_rows', 'memory_stress', 'passes', 'topology_fingerprint',
            'gate_c', 'freeze_sha256', 'calibration_script_sha256', 'calibration_sources_sha256')})
        admission = admission_from_evidence(batch=batch, fingerprint=fingerprint, evidence_id=evidence_id,
            native_agreement=agreement, batch_record=batch_record, stress_record=stress_record,
            stress_complete=stress_result['complete'], headroom_bytes=headroom)
        current = memory_snapshot(torch)
        admission.require(batch, runtime_fingerprint=fingerprint,
            total_bytes=[m['total_bytes'] for m in current], free_bytes=[m['free_bytes'] for m in current],
            allocated_bytes=[m['allocated_bytes'] for m in current], reserved_bytes=[m['reserved_bytes'] for m in current])
        # Exercise the actual upstream planner plus deferred collector. These
        # are fixed development reference intervals, not TempSeg predictions.
        episode_id = selected[0].episode_id
        if any(r.episode_id != episode_id for r in selected):
            raise RuntimeError('The collector demonstration cannot batch across episodes')
        selected_routes = [routes[r.request_id] for r in selected]
        expected_input_digests = [row['input_tensor_digest'] for row in batch.routing]
        demo_segments = [dict(start_s=r['start_s'], end_s=r['end_s'], metadata={}) for r in selected_routes]
        # Drop both original source tensors and the collated copy before native
        # preparation creates the demonstration pair. Keep only JSON receipts.
        search.selected = None
        search.candidates.clear()
        search.source_bytes = 0
        selected = batch = request = None
        report['capture_tensors_released_before_demonstration'] = True
        def demonstrate_collector():
            for job in jobs:
                if job.id == episode_id: job.verify_sources()
            prepared = prepared_episodes[episode_id]
            def no_additional_single(*args, **kwargs):
                raise RuntimeError('Calibration demonstration cannot add unmatched single-generation calls')
            collector = CaptionCollector(no_additional_single, session=backend.session,
                episode_id=episode_id, request_id=_digest([contract_id, report['selected_routes']]),
                profile=prepared.profile, contract_id=contract_id, admissions=[admission],
                topology_fingerprint=report['topology_fingerprint'], max_pending_requests=2,
                max_prepared_bytes=args.max_prepared_mib * 1024**2)
            with backend._execution(), patch.object(backend.pipeline, 'generate', side_effect=collector):
                rows, audit = backend.pipeline.caption_single(prepared.videos, demo_segments, prepared.profile,
                    backend.session, prepared.job.get('instruction', ''), backend.progress)
                resolved = collector.finish(dict(segments=rows, pipeline={'caption_inputs': audit}))
            observed = [r['input_tensor_digest'] for r in collector.audit['request_routes']]
            expected = expected_input_digests
            captions = [r['caption'] for r in resolved['segments']]
            complete = (len(collector.audit['batch_runs']) == 1 and not collector.audit['single_runs'] and
                        observed == expected and captions == singles and
                        all(r['status'] == 'complete' for r in resolved['segments']))
            return dict(complete=complete, interval_provenance='fixed_development_reference_intervals',
                original_reference_indices=[r['reference_index'] for r in selected_routes],
                native_captions=captions, input_digests_match=observed == expected,
                native_caption_agreement=captions == singles, collector=collector.audit)
        demo, _ = measured_pass(torch, report['passes'], 'native_collector_demonstration', demonstrate_collector)
        report['collector_demonstration'] = demo
        if not demo['complete']:
            raise RuntimeError('The real native collector demonstration did not preserve inputs, routing, and captions')
        verify_frozen_release(args, report, contract)
        # A candidate is evidence for this exact contract and hardware. It does
        # not modify the engine, CLI defaults, or any other run configuration.
        report.update(status='calibration_passed',
                      admission_evidence_id=evidence_id, candidate_admission=vars(admission))
        return 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return 1
    finally:
        if backend is not None:
            cleanup = time.perf_counter()
            model = runtime_processor = None
            try: backend.close()
            except Exception as error: report.update(status='failed', cleanup_error=str(error))
            report['cleanup_seconds'] = time.perf_counter() - cleanup
        report['total_workflow_seconds'] = time.perf_counter() - started
        report['allocated_gpu_hours'] = None
        report['allocation_accounting'] = 'Coordinator must reconcile final scheduler state, failed/idle time, and all GPUs'
        atomic_json(path, report)
        if report['status'] == 'calibration_passed':
            try:
                verify_frozen_release(args, report, contract)
            except Exception as error:
                report.update(status='failed', error_type=type(error).__name__, error=str(error))
                atomic_json(path, report)
                return 1
            atomic_json(args.output_dir / 'admission.json', report['candidate_admission'])
            report['admission_written'] = True
            atomic_json(path, report)
        if report['status'] == 'failed':
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
