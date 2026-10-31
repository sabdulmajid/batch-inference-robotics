import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bir.cli import project_bytes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/Flora"
HAS_VIDEO = all(importlib.util.find_spec(name) is not None for name in ("av", "numpy", "PIL"))


class CliTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-m", "bir.cli", *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)

    def test_batch_two_is_rejected_before_source_access(self):
        result = self.invoke("run", "--source", "missing", "--manifest", "missing", "--run-dir", "unused",
                             "--batch-size", "2")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Batch sizes above one", result.stderr)

    def test_storage_uses_current_policy_and_counts_hardlinks_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "one").write_bytes(b"123")
            (path / "two").hardlink_to(path / "one")
            self.assertEqual(project_bytes(path), 3)
            result = self.invoke("storage-check", "--project-path", path, "--planned-gib", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["allowed"])

    @unittest.skipUnless(HAS_VIDEO and SOURCE.is_dir(), "requires pinned source and optional CPU video dependencies")
    def test_real_native_inspection_and_failed_run_are_reported(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            video = path / "synthetic.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=8)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for i in range(16):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            manifest = path / "jobs.jsonl"
            manifest.write_text(json.dumps(dict(id="synthetic", video=video.name, profile="human")))
            inspect = self.invoke("inspect", "--source", SOURCE, "--manifest", manifest)
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            result = json.loads(inspect.stdout)
            self.assertEqual(result["new_model_calls"], 0)
            self.assertEqual(result["episodes"][0]["cameras"]["main"]["frames"], 16)
            # This task's local test environment deliberately has no GPU stack.
            # Do not start a model download if the test is later run on a GPU.
            if importlib.util.find_spec("torch") is None:
                run_dir = path / "run"
                failed = self.invoke("run", "--source", SOURCE, "--manifest", manifest, "--run-dir", run_dir,
                                     "--stages", "segment", "--condition", "optimized")
                self.assertEqual(failed.returncode, 1, failed.stdout)
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertFalse(summary["complete"])
                self.assertEqual(summary["attempts"][0]["state"], "failed")
                self.assertIn("total_cli_workflow_s", summary)
                changed = self.invoke("run", "--source", SOURCE, "--manifest", manifest, "--run-dir", run_dir,
                                      "--condition", "persistent")
                self.assertEqual(changed.returncode, 1)
                self.assertIn("different code/configuration/condition", changed.stderr)


if __name__ == "__main__":
    unittest.main()
