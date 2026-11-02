"""CPU CLI integration. Test fixtures never claim native predictions or actions."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import h5py
except ImportError:
    h5py = None

from bir.bundle import make_bundle, save_bundle, load_bundle
from bir.cli import main, _policy
from bir.assets import asset_path
from bir.manifest import file_sha256
from bir.resource_ledger import ResourceLedger
from bir.trajectory import SOURCE_MASK
from bir.workflow import digest


class RolloutCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "fixture.mp4"
        self.video.write_bytes(b"CLI fixture; not a real video or model output")
        self.media = [dict(camera="main", filename="fixture.mp4", source_sha256=file_sha256(self.video), duration_s=.05)]
        self.task = "Lift the cube with the Panda robot."
        t = dict(schema_version="bir_trajectory_v1", episode_id="demo_0",
            source=dict(dataset="synthetic-cli-test", revision="test", sha256="a" * 64, demo_id="demo_0"),
            actions=[[.5]], observations={"state": [[1.]]}, next_observations={"state": [[1.25]]},
            observation_times_s=[0.], action_times_s=[0.], next_observation_times_s=[.05],
            terminated=[None], truncated=[None], environment_rewards=[1.], source_bootstrap_stop=[1],
            array_dtypes={"actions": "<f8", "rewards": "<f8", "dones": "<i8", "obs/state": "<f8", "next_obs/state": "<f8"},
            boundary=dict(kind="recording_end", source_mask_semantics=SOURCE_MASK),
            action_spec=dict(representation="normalized test command", units=["normalized"], bounds=[-1, 1]),
            environment_metadata=dict(env_name="Lift", type=1, env_kwargs={"test_only": True}),
            clock_mappings=[dict(camera="main", video_sha256=self.media[0]["source_sha256"], scale=1., offset_s=0., source="explicit test clock")])
        t["transition_ids"] = [digest(["a" * 64, "demo_0", 0, 0., .05])]
        t["trajectory_id"] = digest(t)
        self.trajectory = t
        self.episode = dict(episode_id="demo_0", task=self.task, media=self.media, trajectory=t, records=[], outcome=None)

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([str(x) for x in args])
        payload = json.loads(out.getvalue() or err.getvalue())
        return code, payload

    def save(self, value, name):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def test_bundle_pack_and_attach_keep_saved_prediction_bytes_unchanged(self):
        row = dict(schema_version="bir_label_v1", episode_id="demo_0", request_id="fixture",
            source_sha256={"main": self.media[0]["source_sha256"]}, profile="robot", interval_index=0,
            start_s=0., end_s=.05, generated_caption="Synthetic test caption.", inference_status="complete",
            failure=None, generation={"precomputed": True, "test_fixture": True})
        row["record_id"] = digest(row)
        run = self.root / "saved-run"
        run.mkdir()
        source = run / "precomputed.json"
        source.write_text(json.dumps({"records": [row]}))
        before = source.read_bytes()
        info = self.save({"demo_0": {"task": self.task, "media": self.media}}, "media-info.json")
        output = self.root / "packed.json"
        code, _ = self.invoke("bundle-pack", "--run-dir", run, "--media-info", info, "--output", output)
        self.assertEqual(code, 0)
        trajectory = self.save(self.trajectory, "trajectory.json")
        attached = self.root / "attached.json"
        code, _ = self.invoke("bundle-attach", "--bundle", output, "--episode-id", "demo_0", "--trajectory", trajectory, "--output", attached)
        self.assertEqual(code, 0)
        code, summary = self.invoke("bundle-check", "--bundle", attached)
        self.assertEqual(code, 0)
        self.assertEqual(summary["action_episodes"], 1)
        self.assertEqual(summary["annotation_records"], 1)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(json.loads(load_bundle(output)["payload_json"])["episodes"][0]["trajectory"], None)

    def test_bundle_check_reports_corruption_without_creating_outputs(self):
        bundle = make_bundle([self.episode])
        bundle["payload_json"] += " "
        path = self.save(bundle, "corrupt.json")
        code, error = self.invoke("bundle-check", "--bundle", path)
        self.assertEqual(code, 1)
        self.assertEqual(error["status"], "failed")
        self.assertIn("checksum", error["error"])

    def test_import_dispatch_preserves_selected_demos_and_rejects_wrong_source(self):
        source = self.root / "fixture.hdf5"
        source.write_bytes(b"not the pinned real dataset")
        output = self.root / "imported.json"
        with patch("bir.trajectory.import_robomimic", return_value=[self.episode]) as importer:
            code, _ = self.invoke("trajectory-import", "--dataset", source, "--demos", "demo_0", "--output", output)
        self.assertEqual(code, 0)
        importer.assert_called_once_with(source, demos=["demo_0"], media_receipts=None)
        # The genuine importer rejects the bytes; the dispatch fixture above is not real-data evidence.
        code, error = self.invoke("trajectory-import", "--dataset", source, "--demos", "demo_0", "--output", self.root / "bad.json")
        self.assertEqual(code, 1)
        self.assertIn("checksum", error["error"])
        self.assertFalse((self.root / "bad.json").exists())

    @unittest.skipUnless(h5py is not None, "Training export CLI needs the data extra")
    def test_training_export_cli_loads_real_hdf5_and_prevents_duplicate_destination(self):
        path = self.root / "bundle.json"
        save_bundle(make_bundle([self.episode]), path)
        output = self.root / "trainer.hdf5"
        args = ("training-export", "--bundle", path, "--reward-source", "environment", "--boundary-mode", "source-mask", "--output", output)
        code, receipt = self.invoke(*args)
        self.assertEqual(code, 0, receipt)
        self.assertEqual(receipt["transitions"], 1)
        with h5py.File(output, "r") as f:
            self.assertEqual(f["data/demo_0/actions"][0, 0], .5)
            self.assertEqual(f["data/demo_0/rewards"][0], 1.)
            self.assertEqual(f["data/demo_0/bir/terminated"][0], -1)
        before = file_sha256(output)
        code, error = self.invoke(*args)
        self.assertEqual(code, 1)
        self.assertIn("already exists", error["error"])
        self.assertEqual(file_sha256(output), before)

    def test_reward_prepare_is_cpu_only_and_score_is_disabled_by_default(self):
        output = self.root / "request.private.json"
        args = ("reward-prepare", "--episode-id", "demo_0", "--task", self.task, "--video", self.video,
                "--camera", "main", "--clock-offset", "0", "--whole-episode", "--output", output)
        code, receipt = self.invoke(*args)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "prepared_not_executed")
        code, error = self.invoke("reward-score", "--request", output, "--output-dir", self.root / "scores")
        self.assertEqual(code, 1)
        self.assertIn("GPU work remains paused", error["error"])
        self.assertFalse((self.root / "scores").exists())
        code, _ = self.invoke(*args)
        self.assertEqual(code, 1)

    def test_worker_failure_and_interruption_have_nonzero_cli_exit(self):
        request = self.save({"test_only": True}, "request.json")
        for status in ["failed", "interrupted"]:
            with self.subTest(status=status), patch("bir.rewarder.run_terminal", return_value={"status": status, "parse_valid": False}):
                code, result = self.invoke("reward-score", "--request", request, "--output-dir", self.root / "never-created")
            self.assertNotEqual(code, 0)
            self.assertEqual(result["status"], status)

    def test_resource_extraction_is_additive_and_refuses_existing_directory(self):
        # A small packaged-resource fixture checks extraction semantics, not wheel contents.
        package = self.root / "installed-bir"
        data = package / "data" / "configs"
        data.mkdir(parents=True)
        (data / "fixture.json").write_text('{"test_only":true}')
        output = self.root / "resources"
        with patch("bir.assets.__file__", str(package / "assets.py")):
            code, receipt = self.invoke("resources", "--output-dir", output)
            self.assertEqual(code, 0)
            self.assertEqual(receipt["new_model_calls"], 0)
            self.assertEqual((output / "configs/fixture.json").read_text(), '{"test_only":true}')
            code, error = self.invoke("resources", "--output-dir", output)
            self.assertEqual(code, 1)
            self.assertIn("already exists", error["error"])

    def test_existing_storage_and_ledger_commands_still_work_without_submission(self):
        policy = asset_path("configs/resource_policy.yaml")
        code, result = self.invoke("storage-check", "--policy", policy, "--project-path", self.root, "--planned-gib", "0")
        self.assertEqual(code, 0)
        self.assertTrue(result["allowed"])
        ledger_path = self.root / "test-ledger.sqlite"
        ledger = ResourceLedger(ledger_path, _policy(policy), "test-coordinator")
        before = ledger.snapshot()
        code, result = self.invoke("ledger-status", "--policy", policy, "--ledger", ledger_path, "--coordinator", "test-coordinator")
        self.assertEqual(code, 0)
        self.assertEqual(result, before)
        self.assertEqual(result["gpu_submissions_and_reservations"], 0)

    def test_cli_module_help_is_available_without_gpu_or_heavy_model_setup(self):
        result = subprocess.run([sys.executable, "-m", "bir.cli", "bundle-check", "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--bundle", result.stdout)


if __name__ == "__main__":
    unittest.main()
