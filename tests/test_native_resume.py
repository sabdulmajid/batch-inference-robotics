"""Actual CPU CLI resume of synthetic durable checkpoints; no native GPU claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from bir.cli import atomic_json, build_execution_contract
from bir.manifest import canonical_json, file_sha256, load_manifest
from bir.store import RunStore
from scripts import verify_native_resume as resume


class NativeResumeTests(unittest.TestCase):
    def make_fixture(self, root, *, export_segments):
        repository = Path(__file__).resolve().parents[1]
        source = repository / 'third_party/Flora'
        if not (source / 'captioner/batch_utils/models.json').is_file():
            self.skipTest('Requires the pinned local FLORA source; no weights or GPUs are used')
        code, state = root / 'code', root / 'original'
        shutil.copytree(repository / 'src/bir', code / 'src/bir')
        (code / 'configs').mkdir()
        shutil.copyfile(repository / 'configs/native-models.json', code / 'configs/native-models.json')
        lock = code / 'configs/native-models.json'
        contract = build_execution_contract(model_lock=lock)
        contract['engine_source_sha256'] = {p.name: file_sha256(p) for p in sorted((code / 'src/bir').glob('*.py'))}
        manifest = root / 'jobs.jsonl'
        entries = []
        for index in range(3):
            video = root / ('input-%d.mp4' % index)
            video.write_bytes(b'identity fixture ' + bytes([index]))
            entries.append(dict(id='episode-%d' % index, video=str(video), profile='human'))
        manifest.write_text('\n'.join(json.dumps(r) for r in entries) + '\n')
        jobs = load_manifest(manifest, contract=contract)
        state.mkdir()
        store = RunStore(state / 'run.sqlite')
        with store.exclusive():
            for job in jobs:
                store.register(job)
                for stage in ('segment', 'caption'):
                    attempt = store.begin(job.request_id, stage)
                    payload = dict(episode_id=job.id, request_id=job.request_id,
                        segments=[dict(start_s=0., end_s=1., status='complete', caption='fixture')])
                    store.complete(job.request_id, stage, attempt, payload, 1.)
                    if stage == 'caption' or export_segments:
                        store.export(job.request_id, state / 'annotations', stage)
        store.close()
        metadata = dict(schema_version='bir_run_contract_v1', contract=contract,
                        contract_sha256=hashlib.sha256(canonical_json(contract).encode()).hexdigest())
        atomic_json(state / 'run-metadata.json', metadata)
        atomic_json(state / 'summary.json', dict(complete=True, fresh_stage_attempts=6, fixture=True))
        marker = root / 'gate.json'
        atomic_json(marker, dict(execution_contract=contract, request_ids=[j.request_id for j in jobs],
            run={'exports': [str(state / 'annotations' / (j.request_id + '.caption.json')) for j in jobs]}))
        gate = dict(annotation_artifacts=[dict(episode_id=j.id, sha256=file_sha256(state / 'annotations' /
            (j.request_id + '.caption.json'))) for j in jobs])
        args = argparse.Namespace(output_dir=root / 'verification', state_dir=state,
            python=Path(sys.executable), code_root=code, source_root=source, manifest=manifest,
            model_lock=lock, gate_c_marker=marker, timeout_seconds=10.)
        return args, gate, jobs

    def test_real_cli_resumes_six_checkpoints_with_caption_only_or_both_exports(self):
        for export_segments in (False, True):
            with self.subTest(export_segments=export_segments), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, gate, jobs = self.make_fixture(root, export_segments=export_segments)
                state, code = args.state_dir, args.code_root
                before = {p.relative_to(state): p.read_bytes() for p in state.rglob('*') if p.is_file()}
                with patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '0'}), \
                        patch.object(resume, 'verify_gate_c', return_value=gate):
                    result = resume.run(args)
                    self.assertEqual(result, 0, (args.output_dir / 'receipt.json').read_text())
                    receipt = json.loads((args.output_dir / 'receipt.json').read_text())
                    self.assertEqual(receipt['resumed_stages'], 6)
                    self.assertEqual(receipt['fresh_stage_attempts'], 0)
                    self.assertTrue(receipt['original_evidence_unchanged'])
                    self.assertEqual(receipt['command'][-1], str((args.output_dir / 'resume-state').resolve()))
                    self.assertEqual(json.loads((args.output_dir / 'resume-state/summary.json').read_text())['attempts'], [])
                    self.assertEqual(len(receipt['annotation_sha256']), 6 if export_segments else 3)
                    self.assertEqual(len(list((args.output_dir / 'resume-state/annotations').glob('*.segment.json'))),
                                     3 if export_segments else 0)
                    for path, contents in before.items():
                        self.assertEqual((state / path).read_bytes(), contents, str(path))
                    # A verifier from another release can run the exact saved
                    # engine code. A changed engine itself must be rejected.
                    (code / 'src/bir/__init__.py').write_text('# changed code\n')
                    args.output_dir = root / 'rejected'
                    self.assertEqual(resume.run(args), 1)
                    failure = json.loads((args.output_dir / 'receipt.json').read_text())
                    self.assertIn('unchanged engine code', failure['error'])
                    self.assertFalse((args.output_dir / 'resume-state').exists())
                    for path, contents in before.items():
                        self.assertEqual((state / path).read_bytes(), contents, str(path))

    def test_missing_caption_and_changed_existing_exports_fail_before_subprocess(self):
        for change in ('missing_caption', 'caption_payload', 'segment_payload'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                args, gate, jobs = self.make_fixture(Path(directory), export_segments=True)
                stage = 'segment' if change == 'segment_payload' else 'caption'
                path = args.state_dir / 'annotations' / (jobs[0].request_id + '.' + stage + '.json')
                if change == 'missing_caption':
                    path.unlink()
                else:
                    payload = json.loads(path.read_text())
                    payload['segments'][0]['caption'] = 'Changed exported content'
                    atomic_json(path, payload)
                before = {p.relative_to(args.state_dir): p.read_bytes() for p in args.state_dir.rglob('*') if p.is_file()}
                with patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '0'}), \
                        patch.object(resume, 'verify_gate_c', return_value=gate):
                    self.assertEqual(resume.run(args), 1)
                failure = json.loads((args.output_dir / 'receipt.json').read_text())
                expected = 'Every original caption' if change == 'missing_caption' else 'stored checkpoint payload'
                self.assertIn(expected, failure['error'])
                self.assertFalse(failure['subprocess_started'])
                self.assertIsNone(failure['fresh_inference'])
                self.assertFalse((args.output_dir / 'resume-state').exists())
                for path, contents in before.items():
                    self.assertEqual((args.state_dir / path).read_bytes(), contents, str(path))

    def test_optional_segment_export_still_requires_valid_complete_stored_payload(self):
        for change in ('checksum', 'missing_payload', 'non_object'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                args, gate, jobs = self.make_fixture(Path(directory), export_segments=False)
                payload = None if change == 'missing_payload' else '[]'
                checksum = '0' * 64 if change == 'checksum' else (
                    hashlib.sha256(payload.encode()).hexdigest() if payload is not None else None)
                with sqlite3.connect(args.state_dir / 'run.sqlite') as db:
                    db.execute("UPDATE attempts SET payload_json=?,payload_sha256=? WHERE request_id=? AND stage='segment'",
                               (payload, checksum, jobs[0].request_id))
                with patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '0'}), \
                        patch.object(resume, 'verify_gate_c', return_value=gate):
                    self.assertEqual(resume.run(args), 1)
                failure = json.loads((args.output_dir / 'receipt.json').read_text())
                self.assertIn('checksum changed' if change == 'checksum' else 'result object', failure['error'])
                self.assertFalse(failure['subprocess_started'])
                self.assertIsNone(failure['fresh_inference'])


if __name__ == '__main__':
    unittest.main()
