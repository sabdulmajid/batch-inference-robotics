import json
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from bir.engine import Engine, TransientError
from bir.manifest import load_manifest
from bir.store import RunStore


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "video.mp4").write_bytes(b"fixture")
        (self.root / "jobs.jsonl").write_text("\n".join(json.dumps(dict(id=str(i), video="video.mp4")) for i in range(3)))
        self.jobs = load_manifest(self.root / "jobs.jsonl", contract={"source": "test"})
        self.store = RunStore(self.root / "run.sqlite")
        self.addCleanup(self.store.close)
        self.calls = []

    def factory(self, fail=None):
        calls = self.calls
        class Backend:
            def prepare(self, job):
                calls.append(("prepare", job["id"]))
                return job
            def segment(self, prepared):
                calls.append(("segment", prepared["id"]))
                if fail is not None:
                    fail(prepared)
                return {"segments": [], "duration_s": 1, "episode_id": prepared["id"]}
            def caption(self, prepared, segmentation):
                assert segmentation["episode_id"] == prepared["id"]
                calls.append(("caption", prepared["id"]))
                return {"segments": [], "duration_s": 1}
            def close(self):
                calls.append(("close",))
        return Backend

    def test_staged_order_identity_and_resume_without_inference(self):
        engine = Engine(self.factory(), self.store)
        first = engine.run(self.jobs, output_dir=self.root / "out")
        self.assertTrue(first["complete"])
        self.assertEqual(first["fresh_stage_attempts"], 6)
        stages = [c for c in self.calls if c[0] in {"segment", "caption"}]
        self.assertEqual(stages, [(s, str(i)) for s in ("segment", "caption") for i in range(3)])
        self.assertEqual(len([c for c in self.calls if c[0] == "prepare"]), 3)
        self.calls.clear()
        second = engine.run(self.jobs)
        self.assertTrue(second["complete"])
        self.assertEqual(second["fresh_stage_attempts"], 0)
        self.assertEqual(second["resumed_stages"], 6)
        self.assertEqual(self.calls, [("close",)])

    def test_reference_closes_after_each_episode(self):
        Engine(self.factory(), self.store, mode="reference").run(self.jobs)
        stages = [c[0] for c in self.calls if c[0] != "prepare"]
        self.assertEqual(stages, ["segment", "caption", "close"] * 3)
        self.assertEqual(len([c for c in self.calls if c[0] == "prepare"]), 3)

    def test_transient_attempts_are_counted_and_deterministic_failures_block_caption(self):
        attempts = {}
        def fail(prepared):
            identity = prepared["id"]
            attempts[identity] = attempts.get(identity, 0) + 1
            if identity == "0" and attempts[identity] == 1:
                raise TransientError("temporary I/O")
            if identity == "1":
                raise ValueError("invalid native input")
        result = Engine(self.factory(fail), self.store).run(self.jobs)
        self.assertFalse(result["complete"])
        self.assertEqual(result["fresh_stage_attempts"], 6)
        self.assertEqual(attempts, {"0": 2, "1": 1, "2": 1})
        self.assertNotIn(("caption", "1"), self.calls)
        later = Engine(self.factory(fail), self.store).run(self.jobs)
        self.assertEqual(later["fresh_stage_attempts"], 0)

    def test_preparation_overlaps_stage_without_reordering(self):
        prepared_next = threading.Event()
        class Backend:
            def prepare(self, job):
                if job["id"] == "1":
                    prepared_next.set()
                return job
            def segment(self, job):
                if job["id"] == "0":
                    if not prepared_next.wait(2):
                        raise AssertionError("No CPU preparation overlap")
                return {"segments": []}
            def close(self):
                pass
        result = Engine(Backend, self.store, prefetch=1).run(self.jobs, stages="segment")
        self.assertTrue(result["complete"])

    def test_signed_segmentation_is_unchanged_and_partial_output_stays_visible(self):
        from bir.manifest import canonical_json
        import hashlib
        class Backend:
            def prepare(self, job):
                return job
            def segment(self, job):
                result = {"segments": [{"start_s": 0, "end_s": 1}]}
                result["digest"] = hashlib.sha256(canonical_json(result).encode()).hexdigest()
                return result
            def caption(self, job, segmentation):
                digest = segmentation.pop("digest")
                if digest != hashlib.sha256(canonical_json(segmentation).encode()).hexdigest():
                    raise ValueError("Changed signed segmentation")
                return {"segments": [{"start_s": 0, "end_s": 1, "status": "unsupported_target_frames"}]}
            def close(self):
                pass
        engine = Engine(Backend, self.store)
        first = engine.run(self.jobs[:1])
        self.assertTrue(first["execution_complete"])
        self.assertFalse(first["complete"])
        self.assertEqual(first["records"][-1]["status"], "partial")
        second = engine.run(self.jobs[:1])
        self.assertFalse(second["complete"])
        self.assertEqual(second["fresh_stage_attempts"], 0)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'requires optional Pillow CPU dependency')
    def test_preparation_and_real_cache_telemetry_survive_cleanup_and_reset_on_resume(self):
        from PIL import Image
        from bir.frame_cache import DecodedFrameCache
        instances = []
        video = SimpleNamespace(sha256='shared-test-source', pts_us=(0, 1, 2))
        class Backend:
            def __init__(self):
                self.frame_cache = DecodedFrameCache(24, max_entries=2)
                instances.append(self)
            def prepare(self, job):
                return job
            def decode(self, indices):
                def original(source, wanted, canonical=False):
                    return {index: Image.new('RGB', (2, 2), (index, 1, 2)) for index in wanted}
                return self.frame_cache.decode(video, indices, decoder=original)
            def segment(self, job):
                self.decode({0, 1})
                return {'segments': []}
            def caption(self, job, segmentation):
                self.decode({1, 2})
                return {'segments': []}
            def close(self):
                self.frame_cache.clear()
        engine = Engine(Backend, self.store, prefetch=1)
        fresh = engine.run(self.jobs)
        self.assertTrue(fresh['complete'])
        work = fresh['telemetry']['cpu_preparation']
        self.assertEqual(work['preparation_requests'], 6)
        self.assertEqual(work['source_verification_calls'], 6)
        self.assertEqual(work['backend_preparation_calls'], 3)
        self.assertEqual(work['metadata_cache_hits'], 3)
        self.assertEqual(work['source_verification_failures'] + work['backend_preparation_failures'], 0)
        self.assertGreaterEqual(work['source_verification_work_seconds'], 0)
        self.assertGreaterEqual(work['backend_preparation_work_seconds'], 0)
        saved = fresh['telemetry']['backends'][0]['decoded_frame_cache']
        self.assertEqual({key: saved[key] for key in ('hits', 'misses', 'evictions')},
                         dict(hits=9, misses=3, evictions=1))
        self.assertEqual(saved['retained_bytes'], 24)
        self.assertEqual(saved['peak_retained_bytes'], 24)
        self.assertEqual((saved['max_bytes'], saved['max_entries']), (24, 2))
        self.assertEqual(instances[0].frame_cache.stats()['retained_bytes'], 0)
        self.assertEqual(instances[0].frame_cache.stats()['peak_retained_bytes'], 24)
        self.assertIn('not end-to-end wall time', fresh['telemetry']['cpu_timing_scope'])
        resumed = engine.run(self.jobs)
        self.assertEqual(resumed['telemetry']['cpu_preparation']['preparation_requests'], 0)
        self.assertEqual(resumed['telemetry']['backends'][0]['decoded_frame_cache']['hits'], 0)
        self.assertEqual(fresh['telemetry']['cpu_preparation']['preparation_requests'], 6)
        json.dumps(fresh['telemetry'], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
