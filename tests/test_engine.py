import json
from pathlib import Path
import tempfile
import threading
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


if __name__ == "__main__":
    unittest.main()
