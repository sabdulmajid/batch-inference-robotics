"""Regression for loader diagnostics crossing real adapter/engine/store APIs.

Inference is stubbed. This checks persistence and recovery, not GPU execution.
When installed, the loading-info fixture uses the actual Transformers class.
"""
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bir.engine import Engine
from bir.manifest import file_sha256, load_manifest
from bir.native import MODEL_REVISIONS, NativeBackend, PersistentSession, PreparedEpisode
from bir.store import RunStore


class NativeEngineSerializationTests(unittest.TestCase):
    def test_separate_stage_workers_keep_signed_results_and_raw_loader_sets(self):
        if importlib.util.find_spec("transformers") is not None:
            from transformers.utils.loading_report import LoadStateDictInfo
            def make_info():
                return LoadStateDictInfo(set(), set(), set(), [], {}).to_dict()
        else:
            def make_info():
                return dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=set(), error_msgs=[])
        calls, sessions = [], []
        class UpstreamSession:
            def __init__(self):
                self.model = self.proc = None
                self.audit = {}
                sessions.append(self)
            def load(self, kind):
                self.model, self.proc = object(), object()
                info = make_info()
                self.audit[kind] = ({"MODEL_LOADED.json": {"base_loading_info": info}}
                                    if kind == "tempseg" else {"loading_info": info})
                return self.model, self.proc
            def close(self):
                self.model = self.proc = None
        def factory():
            backend = NativeBackend.__new__(NativeBackend)
            backend.frame_cache = None
            backend.progress = lambda *args, **kwargs: None
            backend.config = SimpleNamespace(PROFILE_POOLS={"Human": "human"}, MODELS=MODEL_REVISIONS)
            backend.session = PersistentSession(SimpleNamespace(Session=UpstreamSession))
            def prepare(job):
                metadata = dict(input_sha256={"main": file_sha256(Path(job["video"]))}, profile="Human",
                                yam32=False, instruction_sha256=hashlib.sha256(job["instruction"].encode()).hexdigest(),
                                duration_s=1.0)
                return PreparedEpisode(job, {"main": SimpleNamespace(episode_id=job["id"])}, "Human", metadata)
            def segment(videos, pool, session, instruction, progress):
                session.load("tempseg")
                calls.append(("segment", videos["main"].episode_id))
                return ([dict(start_s=0.0, end_s=1.0, metadata={"segmentation_provenance": "fine"})],
                        [dict(frame_plan={"source_frame_indices": [0, 4], "actual_pts_us": [0.0, 500000.0]},
                              output='{"segments":[]}')])
            def caption(videos, segments, profile, session, instruction, progress):
                session.load("captioner")
                calls.append(("caption", videos["main"].episode_id))
                return ([dict(start_s=0.0, end_s=1.0, status="complete", caption="Move the cup.")],
                        [dict(event_id="event_0000", blocks=[])])
            backend.prepare = prepare
            backend.pipeline = SimpleNamespace(segment=segment, caption_single=caption)
            return backend
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": fake_torch}), \
                patch("bir.native.verify_residency", return_value={"all_parameters_and_nonempty_buffers_on_cuda": True}), \
                patch("bir.native.captioner_placement", return_value=nullcontext()):
            root = Path(directory)
            (root / "identity.mp4").write_bytes(b"identity fixture; not decoded")
            manifest = root / "jobs.jsonl"
            manifest.write_text("\n".join(json.dumps(dict(id=f"episode-{i}", video="identity.mp4")) for i in range(2)))
            jobs = load_manifest(manifest, contract={"test": "separate native stage workers"})
            store = RunStore(root / "run.sqlite")
            try:
                segment_result = Engine(factory, store, prefetch=0).run(jobs, stages="segment")
                self.assertTrue(segment_result["complete"], segment_result)
                before = {job.request_id: store.result(job.request_id, "segment") for job in jobs}
                caption_result = Engine(factory, store, prefetch=0).run(jobs, stages="caption", output_dir=root / "out")
                self.assertTrue(caption_result["complete"], caption_result)
                self.assertEqual(caption_result["fresh_stage_attempts"], 2)
                self.assertEqual(calls, [(stage, f"episode-{i}") for stage in ("segment", "caption") for i in range(2)])
                for job in jobs:
                    self.assertEqual(store.result(job.request_id, "segment"), before[job.request_id])
                    output = store.result(job.request_id, "caption")
                    audit = output["engine"]["model_verification"]
                    self.assertEqual(audit["tempseg"]["MODEL_LOADED.json"]["base_loading_info"]["missing_keys"], [])
                    self.assertEqual(audit["captioner"]["loading_info"]["unexpected_keys"], [])
                    self.assertEqual(output["request_id"], job.request_id)
                    self.assertEqual(output["episode_id"], job.id)
                resumed = Engine(factory, store, prefetch=0).run(jobs)
                self.assertTrue(resumed["complete"])
                self.assertEqual(resumed["fresh_stage_attempts"], 0)
                self.assertEqual(resumed["resumed_stages"], 4)
                self.assertEqual(len(calls), 4)
                self.assertIsInstance(sessions[0].audit["tempseg"]["MODEL_LOADED.json"]["base_loading_info"]["missing_keys"], set)
                self.assertIsInstance(sessions[1].audit["captioner"]["loading_info"]["unexpected_keys"], set)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
