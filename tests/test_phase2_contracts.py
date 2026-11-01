from dataclasses import dataclass
from fractions import Fraction
import errno
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from bir.cli import build_execution_contract
from bir.engine import Engine
from bir.manifest import load_manifest
from bir.metadata_cache import PreparedMetadataSpool
from bir.native import PreparedEpisode
from bir.runtime_identity import runtime_identity
from bir.store import RunStore


@dataclass(frozen=True)
class Video:
    path: Path
    sha256: str
    rate: Fraction
    pts_us: tuple
    duration_us: int
    size: tuple
    cfr: bool


class PhaseTwoContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'video.mp4').write_bytes(b'source identity fixture')
        self.manifest = self.root / 'jobs.jsonl'
        self.manifest.write_text('\n'.join(json.dumps(dict(id=f'episode{i}', video='video.mp4')) for i in range(12)))
        self.jobs = load_manifest(self.manifest, contract={'fixture': 1})

    def prepared(self, job):
        return PreparedEpisode(job.as_native_dict(), {'main': Video(job.video,
            job.source_sha256['main'], Fraction(30000, 1001),
            (Fraction(0), Fraction(100100, 3), Fraction(205000, 3)), 100001,
            (1920, 1080), False)}, 'human', {'input_sha256': dict(job.source_sha256)})

    def test_twelve_episode_second_stage_uses_metadata_without_lru_cliff(self):
        case = self
        class Backend:
            media = SimpleNamespace(Video=Video)
            def prepare(self, row):
                calls.append(row['id'])
                return case.prepared(next(job for job in case.jobs if job.id == row['id']))
            def segment(self, prepared):
                return {'segments': [], 'id': prepared.job['id']}
            def caption(self, prepared, segment):
                expected = case.prepared(next(job for job in case.jobs if job.id == prepared.job['id']))
                case.assertEqual(prepared, expected)
                case.assertEqual(segment['id'], prepared.job['id'])
                return {'segments': []}
            def close(self):
                pass
        for spill, expected_calls in [(False, 24), (True, 12)]:
            calls = []
            store = RunStore(self.root / f'run-{spill}.sqlite')
            try:
                engine = Engine(Backend, store, prefetch=1, prepared_cache_size=8, spill_metadata=spill)
                result = engine.run(self.jobs)
                self.assertTrue(result['complete'])
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(result['telemetry']['cpu_preparation']['source_verification_calls'], 24)
                if spill:
                    stats = result['telemetry']['prepared_metadata_spool']
                    self.assertEqual(stats['writes'], 12)
                    self.assertGreater(stats['hits'], 0)
                    self.assertLessEqual(stats['written_bytes'], stats['max_bytes'])
                    self.assertIsNone(engine._metadata_spool)
                resumed = engine.run(self.jobs)
                self.assertEqual(resumed['fresh_stage_attempts'], 0)
                self.assertEqual(len(calls), expected_calls)
            finally:
                store.close()

    def test_spool_refuses_oversized_records_and_rejects_corruption(self):
        spool = PreparedMetadataSpool(max_bytes=1, max_entry_bytes=1)
        self.addCleanup(spool.close)
        self.assertFalse(spool.put(self.jobs[0], self.prepared(self.jobs[0])))
        self.assertIsNone(spool.get(self.jobs[0], SimpleNamespace(media=SimpleNamespace(Video=Video))))
        self.assertEqual(spool.stats()['refused'], 1)
        spool2 = PreparedMetadataSpool()
        self.addCleanup(spool2.close)
        spool2.put(self.jobs[0], self.prepared(self.jobs[0]))
        path, _ = spool2._entries[self.jobs[0].request_id]
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed inside'):
            spool2.get(self.jobs[0], SimpleNamespace(media=SimpleNamespace(Video=Video)))

    def test_source_change_is_rejected_even_when_metadata_exists(self):
        case = self
        class Backend:
            media = SimpleNamespace(Video=Video)
            def prepare(self, row):
                return case.prepared(case.jobs[0])
            def segment(self, prepared):
                case.jobs[0].video.write_bytes(b'changed')
                return {'segments': []}
            def caption(self, *args):
                case.fail('Changed source must not reach captioning')
            def close(self):
                pass
        store = RunStore(self.root / 'changed.sqlite')
        try:
            result = Engine(Backend, store, prefetch=0, spill_metadata=True).run(self.jobs[:1])
            self.assertFalse(result['complete'])
            self.assertIn('Source changed', result['records'][-1]['error'])
        finally:
            store.close()

    def test_interrupted_spill_writes_do_not_leave_unaccounted_files(self):
        spool = PreparedMetadataSpool(max_bytes=1024)
        self.addCleanup(spool.close)
        original = Path.write_bytes
        def interrupted(path, data):
            original(path, data[:100])
            raise OSError(errno.EIO, 'interrupted metadata write')
        with patch.object(Path, 'write_bytes', interrupted):
            for job in self.jobs:
                with self.assertRaises(OSError):
                    spool.put(job, self.prepared(job))
        self.assertEqual(list(Path(spool._directory.name).iterdir()), [])
        self.assertEqual(spool.stats()['written_bytes'], 0)
        self.assertEqual(spool.stats()['write_errors'], 12)
        with patch.object(Path, 'write_bytes', interrupted), patch.object(Path, 'unlink', side_effect=PermissionError):
            with self.assertRaisesRegex(RuntimeError, 'failed cleanup'):
                spool.put(self.jobs[0], self.prepared(self.jobs[0]))
        self.assertTrue(spool.stats()['poisoned'])
        self.assertLessEqual(spool.stats()['unremoved_partial_bound_bytes'], spool.max_bytes)
        with self.assertRaisesRegex(RuntimeError, 'no further writes'):
            spool.put(self.jobs[1], self.prepared(self.jobs[1]))

    def test_dependency_version_changes_request_identity(self):
        with patch('bir.runtime_identity.importlib.metadata.version', return_value='1.0'):
            before = build_execution_contract()
        with patch('bir.runtime_identity.importlib.metadata.version', return_value='1.1'):
            after = build_execution_contract()
        self.assertNotEqual(load_manifest(self.manifest, contract=before)[0].request_id,
                            load_manifest(self.manifest, contract=after)[0].request_id)
        self.assertEqual(before['engine_source_sha256'], after['engine_source_sha256'])

    def test_missing_optional_package_and_gpu_visibility_have_stable_identity(self):
        with patch('bir.runtime_identity.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError), \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2'}):
            before = runtime_identity()
        with patch('bir.runtime_identity.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError), \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
            after = runtime_identity()
        self.assertEqual(before, after)
        self.assertTrue(all(value is None for value in before['packages'].values()))


if __name__ == '__main__':
    unittest.main()
