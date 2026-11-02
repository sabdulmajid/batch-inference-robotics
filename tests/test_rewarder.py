"""CPU contract tests. No fixture is a native Rewarder prediction."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bir.rewarder import (DEFAULT_CONFIG, _atomic, _digest, load_config, load_result,
                         native_command, parse_terminal, prepare_request, run_terminal,
                         terminal_reward, validate_binding, validate_result, _native_executor)


class RewarderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "original.mp4"
        self.video.write_bytes(b"CPU fixture only; not a decodable video or a native prediction")
        self.request = self.prepare()
        self.raw = dict(id="ep-1", task="Place the block on the table.", score=4,
            expected_score=4.0, progress=0.75, probabilities={str(x): float(x == 4) for x in range(1, 6)},
            rationale=None, duration_s=10.0, sampled_frames=64, output="Score: 4")
        self.calls = 0

    def prepare(self, **changes):
        values = dict(episode_id="ep-1", task="Place the block on the table.", videos=[self.video],
                      camera_order=["front"], whole_episode=True, clock_offsets_s=[0])
        values.update(changes)
        return prepare_request(**values)

    def fixture(self, request, attempt):
        self.calls += 1
        _atomic(attempt / "native.jsonl", self.raw)
        return {"fixture": True}

    def run_fixture(self, **kwargs):
        return run_terminal(self.request, self.root / "rewards", _test_executor=self.fixture, **kwargs)

    def test_config_is_pinned_and_installed_default_is_self_contained(self):
        self.assertEqual(load_config(), DEFAULT_CONFIG)
        wrong = self.root / "changed.json"
        value = copy.deepcopy(DEFAULT_CONFIG)
        value["native"]["dtype"] = "float16"
        wrong.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_config(wrong)

    def test_content_identity_changes_only_with_source_task_or_contract(self):
        other = self.root / "renamed.mp4"
        other.write_bytes(self.video.read_bytes())
        self.assertEqual(self.request["request_id"], self.prepare(videos=[other])["request_id"])
        self.assertNotEqual(self.request["request_id"], self.prepare(task="Lift the block.")["request_id"])
        self.assertNotEqual(self.request["request_id"], self.prepare(camera_order=["wrist"])["request_id"])
        self.video.write_bytes(b"changed source")
        self.assertNotEqual(self.request["request_id"], self.prepare()["request_id"])

    def test_incomplete_clock_hash_camera_and_missing_inputs_rejected(self):
        for change in [dict(whole_episode=False), dict(clock_offsets_s=None), dict(clock_offsets_s=[1]),
                       dict(source_sha256=["0" * 64]), dict(task=" "), dict(videos=[]),
                       dict(camera_order=["../bad"]), dict(videos=[self.video, self.video], camera_order=["right", "left"], clock_offsets_s=[0, 0])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(**change)
        with self.assertRaises(FileNotFoundError):
            self.prepare(videos=[self.root / "missing.mp4"])

    def test_default_never_launches_native_work(self):
        with self.assertRaisesRegex(RuntimeError, "GPU work remains paused"):
            run_terminal(self.request, self.root / "rewards")
        self.assertFalse((self.root / "rewards").exists())

    def test_real_upstream_command_has_fixed_native_options_and_local_model(self):
        args = native_command("/example/python", "/example/source", "/example/model", self.root / "jobs", self.root / "raw")
        self.assertEqual(args[1:3], ["-m", "rewarder.batch_utils.terminal_reward"])
        self.assertEqual(args[args.index("--max-model-len") + 1], "16384")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "1")
        self.assertNotIn("--dry-run", args)
        self.assertNotIn("--with-rationale", args)

    def test_subprocess_contract_roundtrip_is_explicitly_a_fixture(self):
        def process_fixture(request, attempt):
            source = self.root / "fixture.json"
            source.write_text(json.dumps(self.raw))
            subprocess.run([sys.executable, "-c",
                "import pathlib,sys;pathlib.Path(sys.argv[2]).write_text(pathlib.Path(sys.argv[1]).read_text()+'\\n')",
                str(source), str(attempt / "native.jsonl")], check=True, timeout=10)
            return {"fixture": "CPU subprocess; no model loaded"}
        result = run_terminal(self.request, self.root / "rewards", _test_executor=process_fixture)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["parse_valid"])
        self.assertEqual(result["execution"]["kind"], "test_fixture")
        with self.assertRaisesRegex(ValueError, "test-fixture"):
            terminal_reward(result)

    def test_resume_is_independent_and_does_not_repeat_generation(self):
        first = self.run_fixture()
        second = self.run_fixture()
        self.assertEqual(first, second)
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.root / "rewards" / "run.sqlite").exists())
        result_path = self.root / "rewards" / self.request["request_id"] / "result.json"
        self.assertEqual(load_result(result_path), first)
        self.assertNotIn(str(self.video), result_path.read_text())

    def test_invalid_output_unknown_until_explicit_retry_and_history_is_kept(self):
        self.raw["output"] = "No valid answer"
        failed = self.run_fixture()
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["parse_valid"])
        self.assertIsNone(failed["score"])
        self.assertIsNone(failed["progress"])
        self.assertEqual(failed["raw_record"]["output"], "No valid answer")
        self.assertEqual(self.run_fixture(), failed)
        self.assertEqual(self.calls, 1)
        self.raw["output"] = "Score: 4"
        completed = self.run_fixture(retry=True)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["attempt"], 2)
        first = self.root / "rewards" / self.request["request_id"] / "attempt-0001" / "result.json"
        self.assertEqual(load_result(first), failed)

    def test_parser_rejects_wrong_identity_nonfinite_and_inconsistent_fields(self):
        changes = [dict(id="other"), dict(task="other task"), dict(score=True), dict(score=0),
            dict(expected_score=float("nan")), dict(progress=float("inf")), dict(progress=0),
            dict(probabilities={"4": 1}), dict(probabilities={str(i): 0.3 for i in range(1, 6)}),
            dict(sampled_frames=63), dict(sampled_frames=0), dict(duration_s=0), dict(rationale="extra"),
            dict(output="Score: 5"), dict(extra="unrecognized output contract")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_terminal(dict(self.raw, **change), episode_id="ep-1", task=self.raw["task"])

    def test_native_probability_rounding_is_accepted(self):
        raw = dict(self.raw, probabilities={"1": .0333, "2": .0333, "3": .0333, "4": .8667, "5": .0333},
                   expected_score=3.8333, progress=.7083)
        self.assertEqual(parse_terminal(raw, episode_id="ep-1", task=raw["task"]), raw)

    def test_nan_and_partial_native_json_are_failed_not_crashed_or_zero(self):
        for index, payload in enumerate(['{"score":NaN}', '{"score":', json.dumps(self.raw) + "\n" + json.dumps(self.raw)]):
            def invalid(request, attempt):
                (attempt / "native.jsonl").write_text(payload)
                return {"fixture": True}
            result = run_terminal(self.request, self.root / f"run-{index}", _test_executor=invalid)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["parse_valid"])
            self.assertIsNone(result["progress"])

    def test_overflowing_json_number_is_failed_with_raw_bytes_preserved(self):
        for index, overflow in enumerate(("1e999", "-1e999")):
            payload = json.dumps(self.raw).replace('"expected_score": 4.0', '"expected_score": ' + overflow)
            def invalid(request, attempt):
                (attempt / "native.jsonl").write_text(payload + "\n")
                return {"fixture": True}
            result = run_terminal(self.request, self.root / f"overflow-{index}", _test_executor=invalid)
            directory = self.root / f"overflow-{index}" / self.request["request_id"]
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["parse_valid"])
            self.assertIsNone(result["score"])
            self.assertIsNone(result["progress"])
            self.assertIsNone(result["raw_record"])
            self.assertIsNotNone(result["raw_output_sha256"])
            self.assertEqual(json.loads((directory / "state.json").read_text())["status"], "failed")
            self.assertEqual((directory / "attempt-0001" / "native.jsonl").read_text(), payload + "\n")
            self.assertEqual(load_result(directory / "result.json"), result)

    def test_source_change_and_result_tampering_block_resume(self):
        self.run_fixture()
        self.video.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_fixture()
        self.video.write_bytes(b"CPU fixture only; not a decodable video or a native prediction")
        path = self.root / "rewards" / self.request["request_id"] / "result.json"
        raw = json.loads(path.read_text())
        raw["score"] = 5
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_fixture()

    def test_input_changed_during_execution_is_unknown(self):
        def changed(request, attempt):
            _atomic(attempt / "native.jsonl", self.raw)
            self.video.write_bytes(b"changed during inference")
            return {"fixture": True}
        result = run_terminal(self.request, self.root / "rewards", _test_executor=changed)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["score"])

    def test_interrupted_receipt_cannot_promote_a_partial_file(self):
        directory = self.root / "rewards" / self.request["request_id"]
        attempt = directory / "attempt-0001"
        attempt.mkdir(parents=True)
        _atomic(attempt / "native.jsonl", self.raw)
        _atomic(directory / "state.json", dict(attempt=1, status="running", execution={"kind": "test_fixture"}))
        result = self.run_fixture()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(self.calls, 0)
        self.assertIsNone(result["score"])
        self.assertEqual(self.run_fixture(retry=True)["attempt"], 2)

    def test_live_process_receipt_blocks_ambiguous_retry_without_cancelling(self):
        directory = self.root / "rewards" / self.request["request_id"]
        attempt = directory / "attempt-0001"
        attempt.mkdir(parents=True)
        _atomic(attempt / "process.private.json", dict(pid=os.getpid()))
        _atomic(directory / "state.json", dict(attempt=1, status="running", execution={"kind": "native"}))
        with patch('bir.rewarder.os.kill') as probe, patch('bir.rewarder.os.killpg') as cancel:
            with self.assertRaisesRegex(RuntimeError, "allocation reconciliation"):
                self.run_fixture(retry=True)
        probe.assert_not_called()
        cancel.assert_not_called()

    def test_native_worker_without_process_receipt_cannot_start_a_duplicate_attempt(self):
        directory = self.root / "rewards" / self.request["request_id"]
        attempt = directory / "attempt-0001"
        attempt.mkdir(parents=True)
        _atomic(attempt / "native.jsonl", self.raw)
        _atomic(directory / "state.json", dict(attempt=1, status="running", execution={"kind": "native"}))
        before = (directory / "state.json").read_bytes()
        for retry in (False, True):
            with self.subTest(retry=retry), self.assertRaisesRegex(RuntimeError, "allocation reconciliation"):
                self.run_fixture(retry=retry)
        self.assertEqual(self.calls, 0)
        self.assertFalse((directory / "attempt-0002").exists())
        self.assertFalse((directory / "result.json").exists())
        self.assertEqual((directory / "state.json").read_bytes(), before)

    def test_process_receipt_write_failure_still_cleans_up_started_process(self):
        # Entire native boundary is mocked. No runtime probe, process, or GPU is run.
        runtime = dict(python="3.12.1", packages={
            "vllm": "0.24.0", "transformers": "5.8.1", "qwen-vl-utils": "0.0.14", "decord": "0.6.0",
            "numpy": "2.0.0", "Pillow": "11.0.0", "av": "16.0.1",
        }, gpus=[dict(name="H100 fixture only", memory_bytes=80 * 1024 ** 3)])
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.return_value = 0
        attempt = self.root / "receipt-failure"
        attempt.mkdir()
        def write(path, value):
            if Path(path).name == "process.private.json":
                raise OSError("fixture disk-full receipt failure")
            return _atomic(path, value)
        with patch('bir.rewarder.subprocess.check_output', return_value=json.dumps(runtime)), \
                patch('bir.rewarder._verify_assets', return_value={"fixture": True}), \
                patch('bir.rewarder.subprocess.Popen', return_value=process), \
                patch('bir.rewarder._atomic', side_effect=write), \
                patch('bir.rewarder.os.killpg') as terminate:
            with self.assertRaisesRegex(OSError, "disk-full"):
                _native_executor(self.request, attempt, python="fixture-python", flora_source=self.root,
                                 model_dir=self.root, timeout_s=1)
        import signal
        terminate.assert_called_once_with(process.pid, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=5)

    def test_keyboard_interrupt_records_interruption_and_preserves_attempt(self):
        def stopped(request, attempt):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_terminal(self.request, self.root / "rewards", _test_executor=stopped)
        result = self.run_fixture()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(self.calls, 0)

    def test_separate_mapping_arithmetic_and_version_checks(self):
        # Arithmetic fixture only. This explicit constructed record is not execution evidence.
        result = self.run_fixture()
        result["execution"]["kind"] = "native"
        result["execution"]["receipt"] = dict(runtime={"test_only": "arithmetic fixture"},
            assets=dict(source_commit=DEFAULT_CONFIG["source_commit"], model_revision=DEFAULT_CONFIG["model_revision"],
                        verified_files={"test_only": "not model weights"}))
        result["result_digest"] = _digest({k: v for k, v in result.items() if k != "result_digest"})
        mapped = terminal_reward(result)
        self.assertEqual(mapped["value"], .75)
        self.assertEqual(mapped["placement"], "actual_episode_end_only")
        self.assertEqual(result["raw_record"]["score"], 4)
        with self.assertRaisesRegex(ValueError, "mapping"):
            terminal_reward(result, mapping="changed-v2")
        result["status"] = "failed"
        result["parse_valid"] = False
        result.update(score=None, expected_score=None, progress=None)
        result["result_digest"] = _digest({k: v for k, v in result.items() if k != "result_digest"})
        with self.assertRaisesRegex(ValueError, "valid native"):
            terminal_reward(result)

    def test_portable_binding_requires_matching_task_camera_source_and_config(self):
        result = self.run_fixture()
        c = self.request["contract"]
        cameras = [{"camera": x["camera"], "source_sha256": x["sha256"]} for x in c["sources"]]
        self.assertEqual(validate_binding(result, episode_id=c["episode_id"], task=c["task"], cameras=cameras), result)
        for change in [dict(episode_id="different"), dict(task="different"), dict(cameras=[dict(camera="front", source_sha256="0" * 64)])]:
            args = dict(episode_id=c["episode_id"], task=c["task"], cameras=cameras)
            args.update(change)
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_binding(result, **args)
        result["request"]["config"]["model_revision"] = "0" * 40
        result["request_id"] = _digest(result["request"])
        result["result_digest"] = _digest({k: v for k, v in result.items() if k != "result_digest"})
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_result(result)

    def test_unknown_artifact_is_displayable_but_has_no_training_value(self):
        result = self.run_fixture()
        result.update(status="unavailable", parse_valid=False, score=None, expected_score=None, progress=None, raw_record=None)
        result["execution"] = {"kind": "not_executed", "native_validation": "awaiting_validation"}
        result["result_digest"] = _digest({k: v for k, v in result.items() if k != "result_digest"})
        self.assertEqual(validate_result(result), result)
        with self.assertRaisesRegex(ValueError, "valid native"):
            terminal_reward(result)


if __name__ == "__main__":
    unittest.main()
