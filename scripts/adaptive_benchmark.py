#!/usr/bin/env python3
"""Bounded fixed-reference Captioner experiment inside an authorized allocation.

This script never submits jobs. Gate C and the frozen experiment are required.
Keep its raw outputs and operational receipts in private project storage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from bir.adaptive import run_reference_interval
from bir.cli import _run_lock, atomic_json, build_execution_contract
from bir.manifest import file_sha256, load_manifest
from bir.native import NativeBackend

try:
    from .evaluate_adaptive import canonical, digest, load_freeze, outcome_evidence, read_json, targets_for
    from .native_benchmark import EXPECTED, _guard_allocation, _hardware, verify_gate_c
except ImportError:
    from evaluate_adaptive import canonical, digest, load_freeze, outcome_evidence, read_json, targets_for
    from native_benchmark import EXPECTED, _guard_allocation, _hardware, verify_gate_c

POLICIES = ('native', 'compact', 'adaptive')


def export_outcomes(path, rows):
    """Publish an atomic JSONL index of the verified per-item checkpoints."""
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for key in sorted(rows): stream.write(canonical(rows[key]) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def item_key(episode_id, reference_index, policy):
    return digest([episode_id, reference_index, policy])


def captioner_placement_signature(residency):
    value = residency.get('captioner', {})
    if value.get('all_parameters_and_nonempty_buffers_on_cuda') is not True or not value.get('parameter_bytes_by_device'):
        raise ValueError('Captioner placement lacks verified per-device parameter evidence')
    return digest({key: value.get(key) for key in
        ('parameter_bytes_by_device', 'parameter_dtypes', 'buffer_devices', 'device_map')})


def gate_placement(marker, path):
    signatures = []
    for raw in marker['run']['exports']:
        artifact = Path(raw)
        if not artifact.is_absolute(): artifact = Path(path).resolve().parent / artifact
        signatures.append(captioner_placement_signature(read_json(artifact)['engine']['residency']))
    if not signatures or len(set(signatures)) != 1:
        raise ValueError('Gate C did not preserve one stable Captioner placement')
    return signatures[0]


def policy_items(freeze, split):
    """Use all frozen targets and rotate order; never inspect held-out output."""
    for (episode_id, index), target in targets_for(freeze, split).items():
        order = POLICIES[index % len(POLICIES):] + POLICIES[:index % len(POLICIES)]
        for policy in order:
            yield episode_id, index, policy, target


def verify_outcome(path, *, contract_sha256, freeze_sha256, episode_id, index, policy, target):
    value = read_json(path)
    if value.get('result_sha256') != digest({k: v for k, v in value.items() if k != 'result_sha256'}):
        raise ValueError('An adaptive checkpoint checksum changed')
    expected = dict(execution_contract_sha256=contract_sha256, freeze_sha256=freeze_sha256,
                    episode_id=episode_id, reference_index=index, policy=policy,
                    start_s=target['start_s'], end_s=target['end_s'],
                    input_sha256={'main': target['episode']['source_sha256']})
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('An adaptive checkpoint belongs to another input or experiment')
    seconds = value.get('policy_wall_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('An adaptive checkpoint lacks a valid measured policy time')
    if value.get('status') not in ('complete', 'failed', 'unsupported_target_frames'):
        raise ValueError('An adaptive checkpoint lacks a terminal state')
    return value


def final_projection(freeze, selection_path, development_path, current_contract):
    """Conservative frozen projection from all development policy outcomes."""
    selection, development = read_json(selection_path), read_json(development_path)
    if (selection.get('schema_version') != 'bir_adaptive_review_result_v1' or
            selection.get('freeze_sha256') != freeze['freeze_sha256'] or selection.get('split') != 'development' or
            selection.get('best_fixed_policy') not in ('native', 'compact')):
        raise ValueError('Final inference requires the matching development comparator record')
    if (development.get('status') != 'complete' or development.get('split') != 'development' or
            development.get('freeze_sha256') != freeze['freeze_sha256'] or
            development.get('unmeasured_interrupted_invocations', 0) or development.get('uncommitted_item_attempts', 0)):
        raise ValueError('Final inference requires a completed measured development run')
    dev_contract = development['contract']
    if any(dev_contract.get(k) != current_contract.get(k) for k in
           ('base_engine', 'script_sha256', 'freeze_sha256', 'gate_c_evidence', 'captioner_placement_signature')):
        raise ValueError('Code or model settings changed after development')
    if development.get('validated_captioner_placement_signature') != current_contract.get('captioner_placement_signature'):
        raise ValueError('Development lacks the matching measured Captioner placement')
    rows = []
    outcomes = Path(development_path).resolve().parent / 'outcomes'
    for episode_id, index, policy, target in policy_items(freeze, 'development'):
        rows.append(verify_outcome(outcomes / (item_key(episode_id, index, policy) + '.json'),
            contract_sha256=development['contract_sha256'], freeze_sha256=freeze['freeze_sha256'],
            episode_id=episode_id, index=index, policy=policy, target=target))
    evidence = outcome_evidence(rows)
    if any(selection.get(key) != evidence[key] for key in
           ('execution_contract_sha256', 'outcome_receipts', 'outcome_set_sha256', 'development_selection_digest')):
        raise ValueError('Development selection belongs to different measured outcome checkpoints')
    maxima = {p: max(r['policy_wall_seconds'] for r in rows if r['policy'] == p) for p in POLICIES}
    observed_maxima = dict(maxima)
    gate_maximum = max((attempt['wall_seconds'] for row in rows for attempt in row['attempts']
                        if attempt['category'] == 'gate'), default=0)
    # This is a conservative planning check, not a guaranteed runtime bound.
    # Even if development never falls back, final admission allows that work.
    maxima['adaptive'] = max(maxima['adaptive'], maxima['compact'] + maxima['native'] + gate_maximum)
    for policy in POLICIES:
        measured = sum(r['policy_wall_seconds'] for r in rows if r['policy'] == policy)
        declared = selection['policy_metrics'][policy]['policy_application_seconds']
        if type(declared) not in (float, int) or not math.isclose(measured, declared, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Development selection costs differ from measured checkpoints')
    durations = {split: sum(e['duration_s'] for e in freeze['episodes'] if e['split'] == split)
                 for split in ('development', 'final')}
    shared = development['cumulative_shared_work_seconds']
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in shared.values()):
        raise ValueError('Development shared-work measurements are invalid')
    preparation = shared.get('preparation', 0) * durations['final'] / durations['development']
    other_setup = sum(v for k, v in shared.items() if k != 'preparation')
    final_count = len(targets_for(freeze, 'final'))
    projected = 1.5 * (other_setup + preparation + final_count * sum(maxima.values()))
    ceiling = freeze['protocol']['adaptive']['final_job_minutes_ceiling'] * 60
    if projected > ceiling:
        raise ValueError('Measured final projection exceeds the frozen job ceiling; keep all targets and report not run')
    return dict(projected_seconds=projected, ceiling_seconds=ceiling,
                policy_maxima_seconds=maxima, observed_policy_maxima_seconds=observed_maxima,
                fallback_allowance='max(observed adaptive, compact + native + gate maxima); planning estimate',
                measured_setup_seconds=other_setup,
                scaled_preparation_seconds=preparation, target_count=final_count,
                headroom_factor=1.5, development_selection_sha256=file_sha256(selection_path),
                development_receipt_sha256=file_sha256(development_path),
                best_fixed_policy=selection['best_fixed_policy'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, default=Path('manifests/adaptive-freeze.json'))
    parser.add_argument('--split', choices=('development', 'final'), required=True)
    parser.add_argument('--gate-evidence', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-lock', type=Path, default=Path('configs/native-models.json'))
    parser.add_argument('--captioner-headroom-gib', type=float, default=4.0)
    parser.add_argument('--development-selection', type=Path)
    parser.add_argument('--development-run', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    with _run_lock(args.output_dir):
        return run_locked(args)


def run_locked(args):
    started = time.perf_counter()
    report = dict(schema_version='bir_adaptive_benchmark_v1', status='started', split=args.split,
                  started_utc=datetime.now(timezone.utc).isoformat(), gpu_memory=[],
                  fresh_items=[], cached_items=[], outcomes=[], shared_work_seconds={},
                  resource_accounting='Coordinator must reconcile all allocated GPU-hours, idle time, and failed attempts')
    receipt = args.output_dir / ('invocation-%04d.json' % (len(list(args.output_dir.glob('invocation-*.json'))) + 1))
    atomic_json(receipt, report)
    backend = torch = None
    rows = {}
    publish_view = not ((args.output_dir / 'run-summary.json').exists() or (args.output_dir / 'outcomes.jsonl').exists())
    active_started = None
    try:
        _guard_allocation()
        report['gate_c'] = verify_gate_c(args.gate_evidence)
        marker = read_json(args.gate_evidence)
        if marker.get('fresh_inference') is not True:
            raise ValueError('A cache-only receipt cannot establish Gate C')
        freeze = load_freeze(args.freeze)
        report['freeze_sha256'] = freeze['freeze_sha256']
        phase_seconds = freeze['protocol']['adaptive'][args.split + '_job_minutes_ceiling'] * 60
        report['phase_wall_ceiling_seconds'] = phase_seconds
        base = build_execution_contract(model_lock=args.model_lock,
            captioner_headroom_gib=args.captioner_headroom_gib, condition='fixed_reference_experiment')
        scripts = (Path(__file__), Path(__file__).with_name('evaluate_adaptive.py'),
                   Path(__file__).with_name('native_benchmark.py'), Path(__file__).with_name('compare_native.py'))
        contract = dict(base_engine=base, freeze_sha256=freeze['freeze_sha256'], split=args.split,
                        script_sha256={p.name: file_sha256(p) for p in scripts},
                        frame_cache_bytes=0, policy_order='rotate_by_reference_index',
                        gate_c_evidence=report['gate_c'],
                        captioner_placement_signature=gate_placement(marker, args.gate_evidence))
        if args.split == 'final':
            if not args.development_selection or not args.development_run:
                raise ValueError('Final inference requires the measured development run and frozen comparator')
            report['final_projection'] = final_projection(freeze, args.development_selection, args.development_run, contract)
            contract.update(development_selection_digest=digest(read_json(args.development_selection)),
                            development_selection_sha256=file_sha256(args.development_selection),
                            development_run_sha256=file_sha256(args.development_run))
        elif args.development_selection or args.development_run:
            raise ValueError('Development must not consume a final or previously tuned comparator')
        report.update(contract=contract, contract_sha256=digest(contract))
        metadata = args.output_dir / 'run-contract.json'
        if metadata.exists():
            if not args.resume or read_json(metadata) != contract:
                raise ValueError('Existing adaptive run requires explicit resume with an unchanged contract')
        else:
            if args.resume or (args.output_dir / 'outcomes').exists():
                raise ValueError('Adaptive resume requires its original run contract')
            atomic_json(metadata, contract)
        jobs = load_manifest(args.manifest, contract=contract)
        expected = {e['episode_id']: e for e in freeze['episodes'] if e['split'] == args.split}
        if set(expected) != {j.id for j in jobs} or not 1 <= len(jobs) <= 3:
            raise ValueError('Manifest must contain exactly the frozen split episodes, at most three')
        for job in jobs:
            target = expected[job.id]
            if (dict(job.source_sha256) != {'main': target['source_sha256']} or job.profile != 'human' or
                    job.yam32 or hashlib.sha256(job.instruction.encode()).hexdigest() != target['instruction_sha256']):
                raise ValueError('Source video, profile, or instruction differs from the freeze')
        items = list(policy_items(freeze, args.split))
        report['expected_items'] = len(items)
        pending = []
        failed_cached = []
        for episode_id, index, policy, target in items:
            key = item_key(episode_id, index, policy)
            path = args.output_dir / 'outcomes' / (key + '.json')
            if path.exists():
                rows[key] = verify_outcome(path, contract_sha256=report['contract_sha256'],
                    freeze_sha256=freeze['freeze_sha256'], episode_id=episode_id, index=index, policy=policy, target=target)
                report['cached_items'].append(key)
                if any(attempt['status'] == 'failed' for attempt in rows[key]['attempts']):
                    failed_cached.append(key)
            else:
                pending.append((episode_id, index, policy, target))
        publish_view = True
        if failed_cached:
            raise ValueError('A recorded runtime failure requires diagnosis and a changed contract, not automatic retry')
        if pending:
            report['shared_work_seconds']['preflight'] = time.perf_counter() - started
            environment = {name: importlib.metadata.version(name) for name in EXPECTED}
            if any(environment[k].split('+')[0] != v for k, v in EXPECTED.items()):
                raise RuntimeError('Adaptive environment differs from the pinned GPU packages')
            import torch as framework
            torch = framework
            if not torch.cuda.is_available() or not 1 <= torch.cuda.device_count() <= 4 or torch.version.cuda != '12.8':
                raise RuntimeError('Adaptive inference requires granted CUDA12.8 GPUs')
            report.update(environment=environment, hardware=_hardware(torch))
            actual = [(d['name'], d['total_memory_bytes']) for d in report['hardware']['devices']]
            previous = [(d['name'], d['total_memory_bytes']) for d in marker['gpu_memory']]
            if actual != previous or args.captioner_headroom_gib != marker['execution_contract']['captioner_headroom_gib']:
                raise RuntimeError('Adaptive hardware or placement headroom differs from Gate C')
            def synchronize():
                for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)
            synchronize()
            for device in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(device)
            setup = time.perf_counter()
            backend = NativeBackend(args.source_root, captioner_headroom_gib=args.captioner_headroom_gib)
            report['shared_work_seconds']['setup'] = time.perf_counter() - setup
            prep_start = time.perf_counter()
            prepared = {}
            for job in jobs:
                job.verify_sources()
                value = backend.prepare(job.as_native_dict())
                if (value.metadata['input_sha256'] != dict(job.source_sha256) or
                        abs(value.metadata['duration_s'] - expected[job.id]['duration_s']) > .001):
                    raise ValueError('Prepared source differs from its frozen identity or duration')
                prepared[job.id] = value
            report['shared_work_seconds']['preparation'] = time.perf_counter() - prep_start
            load_start = time.perf_counter()
            try:
                with backend._execution(): backend.session.load('captioner')
                synchronize()
            finally:
                report['shared_work_seconds']['model_load'] = time.perf_counter() - load_start
            report['model_verification'] = backend.session.audit
            report['residency'] = backend.session.residency_audit
            report['placement'] = backend.session.placement_audit
            observed_placement = captioner_placement_signature(backend.session.residency_audit)
            if observed_placement != contract['captioner_placement_signature']:
                raise RuntimeError('Actual Captioner placement differs from the validated native Gate C placement')
            report['validated_captioner_placement_signature'] = observed_placement
            atomic_json(receipt, report)
            for episode_id, index, policy, target in pending:
                if time.perf_counter() - started >= phase_seconds:
                    report['stop_reason'] = 'Frozen phase deadline reached before the next interval'
                    break
                key = item_key(episode_id, index, policy)
                report['active_item'] = dict(key=key, episode_id=episode_id, reference_index=index, policy=policy)
                atomic_json(receipt, report)
                active_started = time.perf_counter()
                row = run_reference_interval(backend, prepared[episode_id], target,
                    episode_id=episode_id, reference_index=index, policy=policy,
                    freeze_sha256=freeze['freeze_sha256'], synchronize=synchronize)
                row['execution_contract_sha256'] = report['contract_sha256']
                if args.split == 'final': row['development_selection_digest'] = contract['development_selection_digest']
                row['result_sha256'] = digest(row)
                atomic_json(args.output_dir / 'outcomes' / (key + '.json'), row)
                rows[key] = row
                report['fresh_items'].append(key)
                report.pop('active_item')
                active_started = None
                atomic_json(receipt, report)
                if any(a['status'] == 'failed' for a in row['attempts']):
                    raise RuntimeError('An interval had a runtime failure; stop before further GPU work')
        report['status'] = 'complete' if len(rows) == len(items) else 'incomplete'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        if active_started is not None:
            report['uncommitted_item_work'] = dict(report['active_item'],
                wall_seconds=time.perf_counter() - active_started,
                status='failed_before_durable_checkpoint_confirmation')
        if torch is not None:
            for device in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(device)
                    report['gpu_memory'].append(dict(logical_device=device,
                        name=torch.cuda.get_device_name(device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
                except Exception as error:
                    report['gpu_memory'].append(dict(logical_device=device, measurement_error=str(error)))
                    report['status'] = 'failed'
        if backend is not None:
            cleanup = time.perf_counter()
            try: backend.close()
            except Exception as error:
                report.update(status='failed', cleanup_error=str(error))
            report['shared_work_seconds']['cleanup'] = time.perf_counter() - cleanup
        report['outcomes'] = [str(args.output_dir / 'outcomes' / (key + '.json')) for key in sorted(rows)]
        if publish_view: export_outcomes(args.output_dir / 'outcomes.jsonl', rows)
        else: report['published_evidence_preserved_after_rejection'] = True
        report['successful_items'] = sum(r['status'] == 'complete' for r in rows.values())
        report['failed_or_unsupported_items'] = len(rows) - report['successful_items']
        report['fresh_inference'] = bool(report['fresh_items'])
        report['policy_wall_seconds'] = sum(r['policy_wall_seconds'] for r in rows.values())
        report['elapsed_s'] = time.perf_counter() - started
        atomic_json(receipt, report)
        previous = [read_json(path) for path in sorted(args.output_dir.glob('invocation-*.json'))]
        shared = {}
        for invocation in previous:
            for key, seconds in invocation['shared_work_seconds'].items(): shared[key] = shared.get(key, 0) + seconds
        unmeasured = sum('elapsed_s' not in p for p in previous)
        placements = {p['validated_captioner_placement_signature'] for p in previous
                      if p.get('validated_captioner_placement_signature')}
        summary = dict(report, cumulative_shared_work_seconds=shared,
                       cumulative_invocation_seconds=None if unmeasured else sum(p['elapsed_s'] for p in previous),
                       unmeasured_interrupted_invocations=unmeasured,
                       uncommitted_item_attempts=sum(bool(p.get('uncommitted_item_work')) for p in previous),
                       validated_captioner_placement_signature=next(iter(placements)) if len(placements) == 1 else None,
                       invocation_receipts=[p.name for p in sorted(args.output_dir.glob('invocation-*.json'))])
        if publish_view: atomic_json(args.output_dir / 'run-summary.json', summary)
    print(json.dumps({'status': report['status'], 'split': args.split,
                      'fresh_items': len(report['fresh_items']), 'cached_items': len(report['cached_items'])}), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
