#!/usr/bin/env python3
"""Verify CPU-only cache resume in an isolated copy of a completed native run.

Run in a coordinator-owned CPU allocation. Keep all output files private. This
utility does not submit jobs, change request identity, or edit original evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bir.cli import _run_lock, atomic_json
from bir.manifest import canonical_json, file_sha256, load_manifest
from bir.native import verify_source
from bir.resource_ledger import reject_restarted_execution
try:
    from .native_benchmark import verify_gate_c, verify_resume
except ImportError:
    from native_benchmark import verify_gate_c, verify_resume


def read(path):
    return json.loads(path.read_text())


def database_snapshot(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        requests = [dict(r) for r in db.execute('SELECT * FROM requests ORDER BY request_id')]
        attempts = []
        for value in db.execute('SELECT * FROM attempts ORDER BY request_id,stage,number'):
            row = dict(value)
            payload = row.pop('payload_json')
            if payload is not None and hashlib.sha256(payload.encode()).hexdigest() != row['payload_sha256']:
                raise ValueError('A stored checkpoint payload checksum changed')
            attempts.append(row)
    return dict(requests=requests, attempts=attempts)


def artifact_hashes(state):
    return {p.name: file_sha256(p) for p in sorted((state / 'annotations').glob('*.json'))}


def run(args):
    output = args.output_dir.resolve()
    state = args.state_dir.resolve()
    if output == state or output.is_relative_to(state) or state.is_relative_to(output):
        raise ValueError('Resume receipts must use a fresh directory outside the original state')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(schema_version='bir_native_resume_verification_v1', status='started',
        started_utc=datetime.now(timezone.utc).isoformat(), fresh_inference=None,
        subprocess_started=False,
        scope='CPU-only checkpoint resume on an isolated exact-state backup')
    atomic_json(output / 'receipt.json', report)
    try:
        state = state.resolve(strict=True)
        reject_restarted_execution()
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Resume verification requires a coordinator-owned CPU allocation')
        if not 0 < args.timeout_seconds <= 240:
            raise ValueError('Resume subprocess timeout must be at most 240 seconds')
        code = args.code_root.resolve(strict=True)
        # Preserve the environment's interpreter path. Resolving a venv symlink
        # would select the base interpreter and can change package availability.
        python = args.python.expanduser().absolute()
        if not python.is_file():
            raise ValueError('The explicit Python interpreter does not exist')
        source = args.source_root.resolve(strict=True)
        manifest = args.manifest.resolve(strict=True)
        model_lock = (args.model_lock or code / 'configs/native-models.json').resolve(strict=True)
        gate = verify_gate_c(args.gate_c_marker)
        marker = read(args.gate_c_marker)
        metadata = read(state / 'run-metadata.json')
        contract = metadata['contract']
        if (metadata['contract_sha256'] != hashlib.sha256(canonical_json(contract).encode()).hexdigest() or
                contract != marker.get('execution_contract') or contract.get('condition') != 'optimized' or
                contract.get('batch_size') != 1 or read(model_lock) != contract.get('native_model_lock')):
            raise ValueError('Resume code/model/condition contract differs from the completed probe')
        hashes = {p.name: file_sha256(p) for p in sorted((code / 'src/bir').glob('*.py'))}
        if hashes != contract['engine_source_sha256']:
            raise ValueError('Use the unchanged engine code that created these checkpoints')
        report['source_verification'] = verify_source(source)
        jobs = load_manifest(manifest, contract=contract)
        ids = {job.request_id for job in jobs}
        if not 1 <= len(jobs) <= 3 or ids != set(marker.get('request_ids', [])):
            raise ValueError('Resume manifest differs from the completed one-to-three-episode probe')
        if {r['episode_id'] for r in gate['annotation_artifacts']} != {j.id for j in jobs}:
            raise ValueError('Gate C does not cover every resume episode')
        for raw in marker['run']['exports']:
            path = Path(raw)
            path = path if path.is_absolute() else args.gate_c_marker.resolve().parent / path
            if path.resolve().parent != (state / 'annotations').resolve():
                raise ValueError('Gate C annotation is outside the supplied original state')
        clone = output / 'resume-state'
        with _run_lock(state):
            before = database_snapshot(state / 'run.sqlite')
            original_hashes = artifact_hashes(state)
            if {r['request_id'] for r in before['requests']} != ids:
                raise ValueError('Stored request identities differ from the resume manifest')
            for request in before['requests']:
                if hashlib.sha256(request['identity_json'].encode()).hexdigest() != request['request_id']:
                    raise ValueError('Stored request identity checksum changed')
            for request in ids:
                for stage in ('segment', 'caption'):
                    attempts = [r for r in before['attempts'] if r['request_id'] == request and r['stage'] == stage]
                    if not attempts or attempts[-1]['state'] != 'complete':
                        raise ValueError('Every original stage must already be complete')
                    if request + '.' + stage + '.json' not in original_hashes:
                        raise ValueError('Every original stage must have its saved annotation')
            clone.mkdir()
            shutil.copytree(state / 'annotations', clone / 'annotations')
            shutil.copyfile(state / 'run-metadata.json', clone / 'run-metadata.json')
            shutil.copyfile(state / 'summary.json', output / 'fresh-summary.json')
            shutil.copyfile(state / 'summary.json', clone / 'summary.json')
            with sqlite3.connect((state / 'run.sqlite').as_uri() + '?mode=ro', uri=True) as src:
                with sqlite3.connect(clone / 'run.sqlite') as dst:
                    src.backup(dst)
            fresh_summary_sha256 = file_sha256(state / 'summary.json')
        atomic_json(output / 'before.json', dict(database=before, annotation_sha256=original_hashes,
                    metadata_sha256=file_sha256(state / 'run-metadata.json'), summary_sha256=fresh_summary_sha256))
        command = [str(python), '-m', 'bir.cli', 'run', '--source', str(source), '--manifest', str(manifest),
            '--model-lock', str(model_lock), '--captioner-headroom-gib', str(contract['captioner_headroom_gib']),
            '--batch-size', '1', '--condition', 'optimized', '--stages', 'all', '--run-dir', str(clone)]
        report.update(gate_c=gate, original_state=str(state), isolated_state=str(clone), command=command,
                      contract_sha256=metadata['contract_sha256'], expected_resumed_stages=2 * len(jobs))
        atomic_json(output / 'receipt.json', report)
        environment = dict(os.environ, PYTHONPATH=str(code / 'src'), CUDA_VISIBLE_DEVICES='',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        log = output / 'resume.log'
        report['subprocess_started'] = True
        atomic_json(output / 'receipt.json', report)
        with log.open('w') as stream:
            completed = subprocess.run(command, cwd=code, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, timeout=args.timeout_seconds, check=False)
        report['returncode'] = completed.returncode
        if completed.returncode:
            raise RuntimeError('The unchanged CLI resume failed; inspect the private log')
        summary = read(clone / 'summary.json')
        verify_resume(summary, len(jobs), log)
        after = database_snapshot(clone / 'run.sqlite')
        resumed_hashes = artifact_hashes(clone)
        atomic_json(output / 'after.json', dict(database=after, annotation_sha256=resumed_hashes))
        if (after != before or resumed_hashes != original_hashes or
                read(clone / 'run-metadata.json') != metadata or
                {(r['request_id'], r['stage']) for r in summary['records']} !=
                {(identity, stage) for identity in ids for stage in ('segment', 'caption')}):
            raise ValueError('Cache resume changed an attempt, artifact, or request/stage identity')
        if (database_snapshot(state / 'run.sqlite') != before or artifact_hashes(state) != original_hashes or
                file_sha256(state / 'summary.json') != fresh_summary_sha256 or read(state / 'run-metadata.json') != metadata):
            raise ValueError('Original inference evidence changed during verification')
        report.update(status='complete', resumed_stages=summary['resumed_stages'], fresh_stage_attempts=0,
                      fresh_inference=False,
                      original_evidence_unchanged=True, cloned_attempts_unchanged=True,
                      annotation_sha256=original_hashes, resume_summary_sha256=file_sha256(clone / 'summary.json'))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(output / 'receipt.json', report)
    print(json.dumps(dict(status=report['status'], resumed_stages=report.get('resumed_stages', 0),
                         fresh_inference=report['fresh_inference'])), flush=True)
    return 0 if report['status'] == 'complete' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'code-root', 'source-root', 'manifest', 'state-dir', 'gate-c-marker', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-lock', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=240.)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
