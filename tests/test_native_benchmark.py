"""CPU checks for benchmark gates and evidence accounting, not GPU benchmarks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bir.native import MODEL_REVISIONS, UPSTREAM_COMMIT
from scripts.native_benchmark import placement_signature, runtime_timings, verify_gate_c, verify_resume


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_gate_requires_both_native_stages_and_saved_complete_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "annotation.json"
            info = dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
            gdn = dict(all_values_equal_saved_checkpoint=True, all_runtime_fp32=True, tensors=2)
            output = {"episode_id": "synthetic", "request_id": "synthetic-request",
                      "model_revisions": {kind: {"revision": revision} for kind, revision in MODEL_REVISIONS.items()},
                      "segments": [{"status": "complete", "caption": "fixture"}],
                      "pipeline": {"temporal_windows": [1], "caption_inputs": [1]},
                      "engine": {"source_commit": UPSTREAM_COMMIT, "mode": "placement_adapted_native", "request_batch_size": 1,
                          "residency": {stage: {"all_parameters_and_nonempty_buffers_on_cuda": True}
                                        for stage in ("tempseg", "captioner")},
                          "model_verification": {"tempseg": {
                              "ADAPTER_VALUE_VERIFICATION.json": dict(all_saved_values_exact_before_and_after_placement=True,
                                  holders_fp32_before_assignment=True, merged=False),
                              "GDN_VALUE_VERIFICATION.json": gdn,
                              "MODEL_LOADED.json": dict(base_loading_info=info, gdn_values_exact=True,
                                                          adapter_persistent_dtype="torch.float32")},
                              "captioner": dict(gdn=gdn, loading_info=info)}}}
            artifact.write_text(json.dumps(output))
            marker = root / "probe.json"
            marker_data = {"status": "complete", "native_caption_gate": True, "fresh_inference": True,
                          "run": {"complete": True, "exports": [str(artifact)], "records": [dict(stage="caption", status="complete",
                                  fresh_execution=True, request_id="synthetic-request")]}}
            marker.write_text(json.dumps(marker_data))
            self.assertEqual(verify_gate_c(marker)["annotation_artifacts"][0]["episode_id"], "synthetic")
            marker_data["fresh_inference"] = False
            marker.write_text(json.dumps(marker_data))
            with self.assertRaisesRegex(ValueError, "native caption"):
                verify_gate_c(marker)
            marker_data["fresh_inference"] = True
            marker.write_text(json.dumps(marker_data))
            original_audit = output["engine"]["model_verification"]
            output["engine"]["model_verification"] = {"tempseg": {}, "captioner": {}}
            artifact.write_text(json.dumps(output))
            with self.assertRaisesRegex(ValueError, "FP32 adapter"):
                verify_gate_c(marker)
            output["engine"]["model_verification"] = original_audit
            del output["engine"]["residency"]["tempseg"]
            artifact.write_text(json.dumps(output))
            with self.assertRaisesRegex(ValueError, "both stages"):
                verify_gate_c(marker)
            marker.write_text(json.dumps({"status": "complete", "native_tempseg_gate": True,
                                          "run": {"complete": True, "exports": [str(artifact)]}}))
            with self.assertRaisesRegex(ValueError, "native caption"):
                verify_gate_c(marker)

    def test_resume_rejects_hidden_generation_and_new_attempts(self):
        summary = {"complete": True, "fresh_stage_attempts": 0, "attempts": [], "resumed_stages": 2,
                   "records": [{"status": "resumed", "fresh_execution": False} for _ in range(2)]}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "resume.log"
            log.write_text("No model operations\n")
            verify_resume(summary, 1, log)
            log.write_text('FLORA_RUNTIME {"event":"generation_start"}\n')
            with self.assertRaisesRegex(ValueError, "executed generation"):
                verify_resume(summary, 1, log)
            with self.assertRaisesRegex(ValueError, "fresh or incomplete"):
                verify_resume(dict(summary, fresh_stage_attempts=1), 1, log)

    def test_actual_placement_must_match_every_episode(self):
        record = {"all_parameters_and_nonempty_buffers_on_cuda": True,
                  "parameter_bytes_by_device": {"cuda:0": 100}, "parameter_dtypes": {"torch.bfloat16": 10},
                  "buffer_devices": {"cuda:0": 1}, "device_map": {"": "0"}}
        output = {"engine": {"residency": {stage: copy.deepcopy(record) for stage in ("tempseg", "captioner")}}}
        first = placement_signature({"one": output, "two": copy.deepcopy(output)})[0]
        changed = copy.deepcopy(output)
        changed["engine"]["residency"]["captioner"]["device_map"] = {"": "1"}
        self.assertNotEqual(first, placement_signature({"one": changed})[0])
        with self.assertRaisesRegex(ValueError, "within one"):
            placement_signature({"one": output, "two": changed})

    def test_runtime_load_and_generation_times_keep_stage_bindings(self):
        # Sanitized fields from the actual TempSeg trace. Upstream model_ready
        # deliberately has no elapsed_s; parameter verification precedes it.
        events = [dict(event="model_load_start", kind="tempseg"),
                  dict(event="model_parameters_verified", kind="tempseg", elapsed_s=67.285),
                  dict(event="model_ready", kind="tempseg"),
                  dict(event="generation_end", input_tokens=1811, output_limit=1024,
                       generated_tokens=84, complete=True, elapsed_s=7.942),
                  dict(event="generation_end", input_tokens=1810, output_limit=1024,
                       generated_tokens=81, complete=True, elapsed_s=7.223)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text("\n".join("FLORA_RUNTIME " + json.dumps(event) for event in events))
            result = runtime_timings(path)["stages"]
            self.assertEqual(result["tempseg"]["model_parameter_load_verification_event_seconds"], 67.285)
            self.assertEqual(result["tempseg"]["model_ready_count"], 1)
            self.assertIsNone(result["tempseg"]["session_load_event_seconds"])
            self.assertAlmostEqual(result["tempseg"]["generation_event_seconds"], 15.165)
            self.assertEqual(result["tempseg"]["generation_call_count"], 2)
            self.assertEqual(result["captioner"]["model_load_count"], 0)

    def test_adapter_full_load_events_are_separate_and_failed_work_is_counted(self):
        # Test the separately versioned adapter event, not an upstream field.
        events = [dict(event='model_load_end', kind='captioner', cold_load=True,
                       status='complete', elapsed_s=12., synchronization_complete=True),
                  dict(event='model_load_end', kind='captioner', cold_load=True,
                       status='failed', elapsed_s=3., synchronization_complete=False),
                  dict(event='model_load_blocked', kind='captioner', cold_load=False, status='blocked')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.log'
            path.write_text('\n'.join('BIR_RUNTIME ' + json.dumps(event) for event in events))
            result = runtime_timings(path)['stages']['captioner']
            self.assertEqual(result['session_load_event_count'], 2)
            self.assertEqual(result['session_load_failed_event_count'], 1)
            self.assertEqual(result['session_load_event_seconds'], 15.)
            self.assertIsNone(result['model_parameter_load_verification_event_seconds'])
            events[0]['elapsed_s'] = -1
            path.write_text('BIR_RUNTIME ' + json.dumps(events[0]))
            with self.assertRaisesRegex(ValueError, 'invalid stage/time/status'):
                runtime_timings(path)

    def test_no_allocation_or_restart_stops_before_gpu_import(self):
        script = Path(__file__).resolve().parents[1] / "scripts/native_benchmark.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, str(script), "--source-root", str(root), "--manifest", str(root / "none"),
                       "--output-dir", str(root / "output"), "--gate-c-marker", str(root / "gate")]
            environment = dict(os.environ, SLURM_RESTART_COUNT="0")
            environment.pop("SLURM_JOB_ID", None)
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("authorized SLURM allocation", result.stderr)
            self.assertFalse((root / "output").exists())
            environment.update(SLURM_JOB_ID="synthetic", SLURM_RESTART_COUNT="1")
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Restarted allocation blocked", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
