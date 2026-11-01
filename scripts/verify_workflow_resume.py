#!/usr/bin/env python3
"""Verify the advertised annotate command resumes an isolated state on CPU.

This helper submits no job. It preserves the fresh GPU run, hides GPUs only in
the resume child, and requires exactly unchanged attempts and generated labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


def run(args):
    code = args.code_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(code / 'src'))
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    from bir.cli import _run_lock, atomic_json, build_execution_contract
    from bir.manifest import canonical_json, file_sha256, load_manifest
    from bir.native import verify_source
    from bir.resource_ledger import reject_restarted_execution
    from native_benchmark import verify_fresh_run, verify_resume
    from verify_native_resume import artifact_hashes, database_snapshot

    reject_restarted_execution()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run workflow reproduction in a coordinator-owned allocation')
    state = args.state_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output == state or output.is_relative_to(state) or state.is_relative_to(output):
        raise ValueError('Use a separate fresh output directory')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(schema_version='bir_workflow_resume_v1', status='started',
                   fresh_model_calls=None, original_evidence_unchanged=False,
                   subprocess_started=False, returncode=None)
    atomic_json(output / 'receipt.json', receipt)
    try:
        metadata = json.loads((state / 'run-metadata.json').read_text())
        contract = build_execution_contract(model_lock=code / 'configs/native-models.json',
                                             condition='accelerated')
        hashes = {p.name: file_sha256(p) for p in sorted((code / 'src/bir').glob('*.py'))}
        if (metadata['contract'] != contract or hashes != contract['engine_source_sha256'] or
                metadata['contract_sha256'] != hashlib.sha256(canonical_json(contract).encode()).hexdigest()):
            raise ValueError('Use the exact engine and runtime that created the fresh run')
        receipt['source_verification'] = verify_source(args.source_root.resolve(strict=True))
        jobs = load_manifest(args.manifest, contract=contract)
        if not 1 <= len(jobs) <= 3:
            raise ValueError('Reproduction uses one to three development episodes')
        fresh = verify_fresh_run(state, len(jobs))
        if set(fresh['outputs']) != {job.id for job in jobs}:
            raise ValueError('Fresh result and reproduction manifest differ')
        identities = {job.request_id for job in jobs}
        names = ['run-metadata.json', 'summary.json', 'inputs.jsonl', 'workflow-inputs.json']
        clone = output / 'resume-state'
        def original_snapshot():
            return dict(database=database_snapshot(state / 'run.sqlite'),
                        annotations=artifact_hashes(state),
                        files={name: file_sha256(state / name) for name in names},
                        inference_log=file_sha256(state / 'inference.log'))
        with _run_lock(state):
            before = original_snapshot()
            if {r['request_id'] for r in before['database']['requests']} != identities:
                raise ValueError('The run contains a different request set')
            expected = {job.request_id: job for job in jobs}
            for request in before['database']['requests']:
                job = expected[request['request_id']]
                if (request['episode_id'] != job.id or
                        hashlib.sha256(request['identity_json'].encode()).hexdigest() != job.request_id or
                        json.loads(request['identity_json']) != job.identity()):
                    raise ValueError('Stored request content differs from the manifest')
            attempts = before['database']['attempts']
            if (len(attempts) != 2 * len(jobs) or
                    {(r['request_id'], r['stage']) for r in attempts} !=
                    {(identity, stage) for identity in identities for stage in ('segment', 'caption')} or
                    any(r['state'] != 'complete' or r['number'] != 1 for r in attempts)):
                raise ValueError('Every original stage must have exactly one complete attempt')
            exports = {r['request_id'] + '.' + r['stage'] + '.json': r['payload_sha256'] for r in attempts}
            if not {identity + '.caption.json' for identity in identities} <= set(before['annotations']):
                raise ValueError('Every original caption must have its saved annotation')
            for name in before['annotations']:
                payload = json.loads((state / 'annotations' / name).read_text())
                if name not in exports or hashlib.sha256(canonical_json(payload).encode()).hexdigest() != exports[name]:
                    raise ValueError('An annotation export differs from its stored checkpoint payload')
            clone.mkdir()
            for name in names:
                shutil.copyfile(state / name, clone / name)
            shutil.copytree(state / 'annotations', clone / 'annotations')
            with sqlite3.connect((state / 'run.sqlite').as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(clone / 'run.sqlite') as destination:
                    source.backup(destination)
        atomic_json(output / 'before.json', before)
        python = args.python.expanduser().absolute()  # Preserve the venv symlink.
        command = [str(python), '-m', 'bir.cli', 'annotate', '--manifest', str(args.manifest.resolve()),
            '--source', str(args.source_root.resolve()), '--run-dir', str(clone),
            '--model-lock', str(code / 'configs/native-models.json'), '--condition', 'accelerated']
        environment = dict(os.environ, PYTHONPATH=str(code / 'src'), CUDA_VISIBLE_DEVICES='',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        receipt.update(command=command, subprocess_started=True)
        atomic_json(output / 'receipt.json', receipt)
        with (output / 'command.log').open('w') as log:
            process = subprocess.run(command, cwd=code, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=240, check=False)
        receipt['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError('The advertised annotate resume command failed; inspect command.log')
        summary = json.loads((clone / 'summary.json').read_text())
        verify_resume(summary, len(jobs), clone / 'inference.log')
        after = dict(database=database_snapshot(clone / 'run.sqlite'), annotations=artifact_hashes(clone))
        if (after['database'] != before['database'] or after['annotations'] != before['annotations'] or
                {(r['request_id'], r['stage']) for r in summary['records']} !=
                {(identity, stage) for identity in identities for stage in ('segment', 'caption')} or
                any(file_sha256(clone / name) != before['files'][name]
                    for name in names if name != 'summary.json')):
            raise ValueError('Resume changed original identities, attempts, inputs, or labels')
        with _run_lock(state):
            if original_snapshot() != before:
                raise ValueError('Original fresh inference evidence changed')
        atomic_json(output / 'after.json', after)
        receipt.update(status='complete', fresh_model_calls=0,
            resumed_stages=summary['resumed_stages'], fresh_stage_attempts=summary['fresh_stage_attempts'],
            original_evidence_unchanged=True, command=command,
            resume_summary_sha256=file_sha256(clone / 'summary.json'))
        result = 0
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        result = 1
    receipt['elapsed_seconds'] = time.monotonic() - started
    atomic_json(output / 'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'code-root', 'source-root', 'manifest', 'state-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
