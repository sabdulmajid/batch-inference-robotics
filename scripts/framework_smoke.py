#!/usr/bin/env python3
"""Small actual CUDA numerical check. Run only inside an authorized allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reject_restart():
    raw = os.environ.get('SLURM_RESTART_COUNT', '0')
    try:
        restarts = int(raw)
    except ValueError as error:
        raise RuntimeError('Invalid scheduler restart count') from error
    if restarts != 0:
        raise RuntimeError('Restarted job requires coordinator budget reconciliation before model work')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {'schema_version': 'bir_framework_smoke_v1', 'status': 'started',
              'started_utc': datetime.now(timezone.utc).isoformat(), 'devices': []}
    atomic_json(args.output, report)
    try:
        reject_restart()
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('This probe requires a SLURM allocation')
        import torch
        report.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      visible_device_count=torch.cuda.device_count())
        if not torch.cuda.is_available() or not torch.cuda.device_count():
            raise RuntimeError('CUDA is unavailable in this allocation')
        if not str(torch.__version__).split('+')[0] == '2.11.0':
            raise RuntimeError('Framework version differs from the pinned annotation environment')
        if torch.version.cuda != '12.8':
            raise RuntimeError('CUDA runtime differs from the pinned cu128 annotation environment')
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda', i)
            properties = torch.cuda.get_device_properties(i)
            torch.cuda.reset_peak_memory_stats(i)
            begin = time.monotonic()
            value = torch.full((256, 256), 0.5, dtype=torch.bfloat16, device=device)
            actual = value @ value
            torch.cuda.synchronize(i)
            correct = bool(torch.equal(actual, torch.full_like(actual, 64.0)))
            record = {'logical_device': i, 'name': properties.name,
                      'total_memory_bytes': properties.total_memory,
                      'compute_capability': [properties.major, properties.minor],
                      'result_is_cuda': actual.is_cuda, 'result_dtype': str(actual.dtype),
                      'numerical_result_correct': correct, 'operation_elapsed_s': time.monotonic() - begin,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(i),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(i)}
            report['devices'].append(record)
            atomic_json(args.output, report)
            if not correct or not actual.is_cuda:
                raise RuntimeError('CUDA matrix multiplication failed its numerical check')
            del value, actual
        report['status'] = 'complete'
        return_code = 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return_code = 1
    report['elapsed_s'] = time.monotonic() - started
    atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'devices_tested': len(report['devices'])}), flush=True)
    return return_code


if __name__ == '__main__':
    sys.exit(main())
