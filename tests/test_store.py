import json
from pathlib import Path
import tempfile
import unittest

from bir.manifest import load_manifest
from bir.store import RunStore, StoreBusy, RetryExhausted


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "video.mp4").write_bytes(b"fixture")
        (self.root / "jobs.jsonl").write_text(json.dumps(dict(id="ep", video="video.mp4")))
        self.job = load_manifest(self.root / "jobs.jsonl", contract={"source": "test"})[0]
        self.store = RunStore(self.root / "run.sqlite")
        self.addCleanup(self.store.close)

    def test_interrupted_attempt_consumes_retry(self):
        with self.store.exclusive():
            self.store.register(self.job)
            self.store.begin(self.job.request_id, "segment")
        with self.store.exclusive():
            first = self.store.latest(self.job.request_id, "segment")
            self.assertEqual(first["error_kind"], "interrupted")
            second = self.store.begin(self.job.request_id, "segment")
            self.assertEqual(second, 2)
            self.store.fail(self.job.request_id, "segment", second, "I/O", kind="transient", elapsed_s=1)
            with self.assertRaises(RetryExhausted):
                self.store.begin(self.job.request_id, "segment")
        self.assertEqual(len(self.store.attempts()), 2)

    def test_complete_is_atomic_and_export_is_identity_safe(self):
        with self.store.exclusive():
            self.store.register(self.job)
            attempt = self.store.begin(self.job.request_id, "segment")
            with self.assertRaises(ValueError):
                self.store.complete(self.job.request_id, "segment", attempt, {"bad": float("nan")}, 1)
            self.assertIsNone(self.store.result(self.job.request_id, "segment"))
            self.store.complete(self.job.request_id, "segment", attempt, {"segments": []}, 1)
            with self.assertRaises(ValueError):
                self.store.complete(self.job.request_id, "segment", attempt, {"wrong": 1}, 2)
            result = self.store.export(self.job.request_id, self.root / "out", "segment")
            self.assertEqual(json.loads(result.read_text()), {"segments": []})

    def test_process_lock_blocks_another_store(self):
        other = RunStore(self.root / "run.sqlite")
        try:
            with self.store.exclusive():
                with self.assertRaises(StoreBusy):
                    with other.exclusive():
                        pass
        finally:
            other.close()

    def test_caption_requires_segment_and_deterministic_error_is_not_retried(self):
        with self.store.exclusive():
            self.store.register(self.job)
            with self.assertRaises(ValueError):
                self.store.begin(self.job.request_id, "caption")
            attempt = self.store.begin(self.job.request_id, "segment")
            self.store.fail(self.job.request_id, "segment", attempt, "bad input", kind="deterministic", elapsed_s=0)
            with self.assertRaises(RetryExhausted):
                self.store.begin(self.job.request_id, "segment")

    def test_changed_request_contract_cannot_reuse_identity(self):
        self.job.contract["source"] = "changed"
        with self.store.exclusive():
            with self.assertRaisesRegex(ValueError, "identity changed"):
                self.store.register(self.job)


if __name__ == "__main__":
    unittest.main()
