"""Actual annotate CLI resume of synthetic checkpoints; no native GPU claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bir.cli import atomic_json, build_execution_contract
from bir.manifest import canonical_json, load_manifest
from bir.store import RunStore
from bir.workflow import prepare_inputs
from scripts import verify_workflow_resume as resume


class WorkflowResumeTests(unittest.TestCase):
    def fixture(self, root):
        repository = Path(__file__).resolve().parents[1]
        source = repository / 'third_party/Flora'
        if not (source / 'captioner/batch_utils/models.json').is_file():
            self.skipTest('Requires pinned local FLORA source; no weights or GPU used')
        code, state = root / 'code', root / 'fresh'
        shutil.copytree(repository / 'src/bir', code / 'src/bir')
        (code / 'configs').mkdir()
        shutil.copyfile(repository / 'configs/native-models.json', code / 'configs/native-models.json')
        contract = build_execution_contract(model_lock=code / 'configs/native-models.json', condition='accelerated')
        video = root / 'source.mp4'
        video.write_bytes(b'identity fixture; never decoded')
        manifest = root / 'jobs.jsonl'
        manifest.write_text(canonical_json(dict(id='fixture', video=str(video), profile='human')) + '\n')
        frozen, _ = prepare_inputs(state, manifest=manifest)
        job = load_manifest(frozen, contract=contract)[0]
        store = RunStore(state / 'run.sqlite')
        attempts, records = [], []
        with store.exclusive():
            store.register(job)
            for stage in ('segment', 'caption'):
                number = store.begin(job.request_id, stage)
                payload = dict(episode_id=job.id, request_id=job.request_id, duration_s=2.,
                    segments=[dict(start_s=0., end_s=1., status='complete', caption='Fixture label.')])
                store.complete(job.request_id, stage, number, payload, 1.)
                if stage == 'caption':
                    store.export(job.request_id, state / 'annotations', stage)
                attempts.append(dict(request_id=job.request_id, stage=stage, number=1, state='complete'))
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, attempt=1,
                                    status='complete', fresh_execution=True))
        store.close()
        atomic_json(state / 'run-metadata.json', dict(schema_version='bir_run_contract_v1', contract=contract,
            contract_sha256=hashlib.sha256(canonical_json(contract).encode()).hexdigest()))
        atomic_json(state / 'summary.json', dict(complete=True, fresh_stage_attempts=2, attempts=attempts,
                                                records=records, fixture=True))
        (state / 'inference.log').write_text('Synthetic original evidence; no model ran.\n')
        return argparse.Namespace(python=Path(sys.executable), code_root=code, source_root=source,
            manifest=manifest, state_dir=state, output_dir=root / 'verification')

    def test_advertised_command_resumes_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            before = {p.relative_to(args.state_dir): p.read_bytes()
                      for p in args.state_dir.rglob('*') if p.is_file()}
            with patch.dict(os.environ, SLURM_JOB_ID='fixture', SLURM_RESTART_COUNT='0'):
                self.assertEqual(resume.run(args), 0, (args.output_dir / 'receipt.json').read_text())
            receipt = json.loads((args.output_dir / 'receipt.json').read_text())
            self.assertEqual(receipt['fresh_model_calls'], 0)
            self.assertEqual(receipt['fresh_stage_attempts'], 0)
            self.assertEqual(receipt['resumed_stages'], 2)
            self.assertTrue(receipt['original_evidence_unchanged'])
            self.assertEqual(receipt['command'][1:4], ['-m', 'bir.cli', 'annotate'])
            for path, value in before.items():
                self.assertEqual((args.state_dir / path).read_bytes(), value, str(path))

    def test_changed_identity_contract_or_export_fails_before_child(self):
        for change in ('code', 'metadata_checksum', 'export', 'stored_identity', 'stored_payload'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                if change == 'code':
                    (args.code_root / 'src/bir/__init__.py').write_text('# changed engine\n')
                elif change == 'metadata_checksum':
                    path = args.state_dir / 'run-metadata.json'
                    value = json.loads(path.read_text()); value['contract_sha256'] = '0' * 64
                    atomic_json(path, value)
                elif change == 'export':
                    path = next((args.state_dir / 'annotations').glob('*.caption.json'))
                    value = json.loads(path.read_text()); value['segments'][0]['caption'] = 'Changed'
                    atomic_json(path, value)
                elif change == 'stored_identity':
                    with sqlite3.connect(args.state_dir / 'run.sqlite') as db:
                        db.execute("UPDATE requests SET identity_json='{}'")
                else:
                    with sqlite3.connect(args.state_dir / 'run.sqlite') as db:
                        db.execute("UPDATE attempts SET payload_json='{}' WHERE stage='segment'")
                with patch.dict(os.environ, SLURM_JOB_ID='fixture', SLURM_RESTART_COUNT='0'):
                    self.assertEqual(resume.run(args), 1)
                receipt = json.loads((args.output_dir / 'receipt.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                self.assertIsNone(receipt['fresh_model_calls'])
                self.assertFalse((args.output_dir / 'resume-state').exists())
                self.assertFalse((args.output_dir / 'command.log').exists())

    def test_child_failure_retains_receipt_and_preserves_original_bytes(self):
        original_run = subprocess.run
        for mode in ('exit', 'timeout'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                before = {p.relative_to(args.state_dir): p.read_bytes()
                          for p in args.state_dir.rglob('*') if p.is_file()}
                def child(command, *positional, **keywords):
                    if command[1:4] == ['-m', 'bir.cli', 'annotate']:
                        if mode == 'timeout':
                            raise subprocess.TimeoutExpired(command, 240)
                        return subprocess.CompletedProcess(command, 3)
                    return original_run(command, *positional, **keywords)
                with patch.dict(os.environ, SLURM_JOB_ID='fixture', SLURM_RESTART_COUNT='0'), \
                        patch('subprocess.run', side_effect=child):
                    self.assertEqual(resume.run(args), 1)
                receipt = json.loads((args.output_dir / 'receipt.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                self.assertTrue(receipt['subprocess_started'])
                self.assertEqual(receipt['command'][1:4], ['-m', 'bir.cli', 'annotate'])
                self.assertEqual(receipt['returncode'], 3 if mode == 'exit' else None)
                self.assertIsNone(receipt['fresh_model_calls'])
                for path, value in before.items():
                    self.assertEqual((args.state_dir / path).read_bytes(), value, str(path))


if __name__ == '__main__':
    unittest.main()
