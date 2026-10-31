"""Actual CPU CLI resume of synthetic durable checkpoints; no native GPU claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from bir.cli import atomic_json, build_execution_contract
from bir.manifest import canonical_json, file_sha256, load_manifest
from bir.store import RunStore
from scripts import verify_native_resume as resume


class NativeResumeTests(unittest.TestCase):
    def test_real_cli_resumes_six_checkpoints_without_changing_fresh_evidence(self):
        repository = Path(__file__).resolve().parents[1]
        source = repository / 'third_party/Flora'
        if not (source / 'captioner/batch_utils/models.json').is_file():
            self.skipTest('Requires the pinned local FLORA source; no weights or GPUs are used')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                        store.export(job.request_id, state / 'annotations', stage)
            store.close()
            import hashlib
            metadata = dict(schema_version='bir_run_contract_v1', contract=contract,
                            contract_sha256=hashlib.sha256(canonical_json(contract).encode()).hexdigest())
            atomic_json(state / 'run-metadata.json', metadata)
            atomic_json(state / 'summary.json', dict(complete=True, fresh_stage_attempts=3, fixture=True))
            marker = root / 'gate.json'
            atomic_json(marker, dict(execution_contract=contract, request_ids=[j.request_id for j in jobs],
                run={'exports': [str(state / 'annotations' / (j.request_id + '.caption.json')) for j in jobs]}))
            gate = dict(annotation_artifacts=[dict(episode_id=j.id, sha256=file_sha256(state / 'annotations' /
                (j.request_id + '.caption.json'))) for j in jobs])
            args = argparse.Namespace(output_dir=root / 'verification', state_dir=state,
                python=Path(sys.executable), code_root=code, source_root=source, manifest=manifest,
                model_lock=lock, gate_c_marker=marker, timeout_seconds=10.)
            before = {p.relative_to(state): p.read_bytes() for p in state.rglob('*') if p.is_file()}
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '0'}), \
                    patch.object(resume, 'verify_gate_c', return_value=gate):
                result = resume.run(args)
                self.assertEqual(result, 0, (args.output_dir / 'resume.log').read_text())
                receipt = json.loads((args.output_dir / 'receipt.json').read_text())
                self.assertEqual(receipt['resumed_stages'], 6)
                self.assertEqual(receipt['fresh_stage_attempts'], 0)
                self.assertTrue(receipt['original_evidence_unchanged'])
                self.assertEqual(receipt['command'][-1], str((args.output_dir / 'resume-state').resolve()))
                self.assertEqual(json.loads((args.output_dir / 'resume-state/summary.json').read_text())['attempts'], [])
                for path, contents in before.items():
                    self.assertEqual((state / path).read_bytes(), contents, str(path))
                # A code change must fail before another CLI run. The separate
                # failed receipt must preserve the earlier successful evidence.
                (code / 'src/bir/__init__.py').write_text('# changed code\n')
                args.output_dir = root / 'rejected'
                self.assertEqual(resume.run(args), 1)
                failure = json.loads((args.output_dir / 'receipt.json').read_text())
                self.assertEqual(failure['status'], 'failed')
                self.assertIn('unchanged engine code', failure['error'])
                self.assertFalse((args.output_dir / 'resume-state').exists())
                for path, contents in before.items():
                    self.assertEqual((state / path).read_bytes(), contents, str(path))


if __name__ == '__main__':
    unittest.main()
