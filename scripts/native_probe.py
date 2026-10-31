#!/usr/bin/env python3
"""At most three native episodes with persistent evidence and GPU peak memory.

Run inside an authorized SLURM allocation. The output directory is private:
raw topology, loader logs, source locations, and annotations must not be
published without review. This script never submits or cancels a job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from framework_smoke import atomic_json, reject_restart

EXPECTED = {'torch': '2.11.0', 'torchvision': '0.26.0', 'transformers': '5.8.1',
            'huggingface-hub': '1.33.0', 'peft': '0.18.1', 'accelerate': '1.12.0',
            'safetensors': '0.7.0', 'av': '16.0.1'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stages', choices=['segment', 'caption', 'all'], default='all')
    parser.add_argument('--condition', choices=['reference', 'persistent', 'optimized'], default='optimized')
    parser.add_argument('--model-lock', type=Path, default=Path('configs/native-models.json'))
    parser.add_argument('--captioner-headroom-gib', type=float, default=4.0)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'probe-summary.json'
    report = {'schema_version': 'bir_native_probe_v1', 'status': 'started',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': args.stages,
              'condition': args.condition, 'gpu_memory': []}
    started = time.monotonic()
    atomic_json(report_path, report)
    torch = None
    try:
        reject_restart()
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Native probes require an authorized SLURM allocation')
        report['python_version'] = sys.version
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('The pinned native GPU environment requires Python 3.12')
        for setting in ('HF_HOME', 'FLORA_MODEL_CACHE'):
            if not os.environ.get(setting):
                raise RuntimeError(setting + ' must point to verified project storage')
        report['environment'] = {name: importlib.metadata.version(name) for name in EXPECTED}
        for name, expected in EXPECTED.items():
            if report['environment'][name].split('+')[0] != expected:
                raise RuntimeError(name + ' does not match the pinned annotation environment')
        import torch as torch_module
        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        report['cuda_runtime'] = torch.version.cuda
        if torch.version.cuda != '12.8':
            raise RuntimeError('CUDA runtime differs from the pinned cu128 annotation environment')
        report['visible_device_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
        with (args.output_dir / 'topology.txt').open('w') as stream:
            for command in (['nvidia-smi', '-L'], ['nvidia-smi', 'topo', '-m']):
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=30)
                if completed.returncode:
                    raise RuntimeError('GPU topology inspection failed')
        from bir.cli import build_execution_contract, execute_run
        from bir.manifest import load_manifest
        from bir.native import NativeBackend
        contract = build_execution_contract(model_lock=args.model_lock,
            captioner_headroom_gib=args.captioner_headroom_gib, condition=args.condition)
        jobs = load_manifest(args.manifest, contract=contract)
        if len(jobs) > 3:
            raise ValueError('Initial native probe accepts at most three episodes')
        report.update(execution_contract=contract, episodes=len(jobs),
                      request_ids=[job.request_id for job in jobs])
        # Complete native CPU validation before the first model load.
        inspector = NativeBackend(args.source_root, captioner_headroom_gib=args.captioner_headroom_gib)
        try:
            report['input_validation'] = [inspector.prepare(job.as_native_dict()).metadata for job in jobs]
        finally:
            inspector.close()
        atomic_json(report_path, report)
        cli_args = argparse.Namespace(source=args.source_root, manifest=args.manifest,
            model_lock=args.model_lock, captioner_headroom_gib=args.captioner_headroom_gib,
            batch_size=1, condition=args.condition, run_dir=args.state_dir, stages=args.stages)
        result, code = execute_run(cli_args, time.monotonic())
        report['run'] = result
        report['status'] = 'complete' if result['complete'] else 'incomplete'
        # A cache-only resume is useful recovery evidence, not fresh inference.
        report['fresh_inference'] = result['fresh_stage_attempts'] > 0
        exported = [json.loads(Path(path).read_text()) for path in result['exports']]
        if args.stages in ('caption', 'all'):
            complete_captions = sum(row.get('status') == 'complete' and bool(row.get('caption', '').strip())
                                    for output in exported for row in output.get('segments', []))
            captioner_verified = any('captioner' in output.get('engine', {}).get('residency', {})
                                     for output in exported)
            fresh_caption = any(row['stage'] == 'caption' and row['fresh_execution'] and row['status'] == 'complete'
                                for row in result['records'])
            report['complete_caption_count'] = complete_captions
            report['native_caption_gate'] = bool(result['complete'] and complete_captions > 0
                                                  and captioner_verified and fresh_caption)
            if not report['native_caption_gate'] and report['fresh_inference']:
                report['status'] = 'incomplete'
        else:
            report['native_tempseg_gate'] = any(output.get('temporal_audit') and
                'tempseg' in output.get('engine', {}).get('residency', {}) for output in exported)
            if not report['native_tempseg_gate'] and report['fresh_inference']:
                report['status'] = 'incomplete'
        code = 0 if report['status'] == 'complete' else 1
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        code = 1
    finally:
        if torch is not None and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)
                    report['gpu_memory'].append({
                        'logical_device': i, 'name': torch.cuda.get_device_name(i),
                        'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(i),
                        'allocated_at_end_bytes': torch.cuda.memory_allocated(i)})
                except Exception as error:
                    report['gpu_memory'].append({'logical_device': i, 'measurement_error': str(error)})
        if report['status'] == 'complete' and any('measurement_error' in row for row in report['gpu_memory']):
            report['status'] = 'incomplete'
            code = 1
        report['elapsed_s'] = time.monotonic() - started
        atomic_json(report_path, report)
    print(json.dumps({'status': report['status'], 'stages': args.stages,
                      'fresh_inference': report.get('fresh_inference', False)}), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
