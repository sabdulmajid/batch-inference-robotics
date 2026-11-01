"""CPU tests for phase-two denominators, admission, and identity instrumentation."""
import copy
import io
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import phase2_benchmark as bench


def selected():
    return bench.read(Path(__file__).resolve().parents[1] / "manifests/phase2-native.json")


def parity():
    return dict(environment={"torch": "2.11.0+cu128"}, placement_sha256="p"*64,
        hardware=dict(cuda_runtime="12.8", devices=[dict(name="RTX 3090")]*3,
                      allocation_identity_sha256="private-allocation-hash"))


def pilot(baseline=600., candidate=300.):
    return dict(status="complete", phase="pilot", source_seconds=100.,
        comparison=dict(compatible=True, tensor_input_agreement_verified=True),
        conditions={version: dict(status="fresh_complete", gpu_count=3, profile_enabled=False,
            subprocess_wall_seconds=seconds, parity=parity())
            for version, seconds in (("baseline", baseline), ("candidate", candidate))})


def plan_and_pairs():
    selection = selected()
    plan = dict(phase="final", episodes=selection["episodes"], blocks=selection["blocks"],
                target_speedup=1.5, pilot_execution_parity=bench.stable_parity(parity()))
    plan["plan_sha256"] = bench.digest(plan)
    pairs = []
    for block in plan["blocks"]:
        row = pilot()
        row.update(phase="final", plan_sha256=plan["plan_sha256"], block=block["block"],
            episode_ids=block["episode_ids"], order=block["order"], request_input_identity_agreement=True)
        pairs.append(row)
    return plan, pairs


class PhaseTwoTests(unittest.TestCase):
    def test_source_only_selection_is_order_independent_and_preserves_invalid_references(self):
        frozen = selected()
        records = [dict(id=e["episode_id"], source_sha256=e["source_sha256"], duration_s=e["duration_s"],
            size=e["source_size"], profile="human", native_supported=True, references_valid=e["independent_references_valid"],
            instruction="private source instruction") for e in frozen["episodes"]]
        records += [dict(id=name, native_supported=True) for name in bench.DEVELOPMENT]
        index = dict(dataset=frozen["dataset"], records=records)
        result = bench.select_workload(index)
        random.Random(3).shuffle(records)
        self.assertEqual(result, bench.select_workload(index))
        self.assertEqual(result, frozen)
        self.assertEqual([len(b["episode_ids"]) for b in result["blocks"]], [5, 6])
        self.assertEqual(len({identity for b in result["blocks"] for identity in b["episode_ids"]}), 11)
        self.assertFalse(next(e for e in result["episodes"] if e["episode_id"] == "homer_39")["independent_references_valid"])
        self.assertNotIn("private source instruction", json.dumps(result))
        self.assertTrue(all(e["instruction"] == "" for e in result["episodes"]))
        records.append(dict(id="unexpected", native_supported=True))
        with self.assertRaisesRegex(ValueError, "pool changed"):
            bench.select_workload(index)

    def test_source_partition_reaches_global_duration_minimum_without_dropping_a_source(self):
        frozen = selected()
        rows = frozen['episodes']
        by_id = {row['episode_id']: row for row in rows}
        microseconds = {key: round(row['duration_s']*1_000_000) for key,row in by_id.items()}
        first = set(frozen['blocks'][0]['episode_ids'])
        second = set(frozen['blocks'][1]['episode_ids'])
        self.assertEqual(first | second, set(by_id))
        self.assertFalse(first & second)
        self.assertEqual(first, {'homer_29','homer_38','homer_48','homer_59','homer_60'})
        self.assertEqual(sum(microseconds[k] for k in first)-sum(microseconds[k] for k in second),336067)
        possible=[]
        for subset in itertools.combinations(sorted(by_id),5):
            other=set(by_id)-set(subset)
            if min(len({by_id[k]['task'] for k in subset}),len({by_id[k]['task'] for k in other}))>=3:
                possible.append(abs(sum(microseconds[k] for k in subset)-sum(microseconds[k] for k in other)))
        self.assertEqual(min(possible),336067)
        self.assertEqual(frozen['previous_selection_sha256'],'3a283c389e765afe4cf0abe232a36c0f47dc205f8f1af86a5e884c3418278eb8')
        self.assertEqual(frozen['block_assignment_revision'],2)

    def test_equal_duration_partitions_use_the_declared_hash_tie_break(self):
        rows=[dict(episode_id=f'episode-{i}',duration_s=1.,task=f'task-{i}') for i in range(11)]
        first,second=bench.balanced_members(rows)
        expected=min(itertools.combinations(sorted(row['episode_id'] for row in rows),5),
            key=lambda ids:bench.digest(['phase2-balance-v2',list(ids)]))
        self.assertEqual(sorted(row['episode_id'] for row in first),list(expected))
        self.assertEqual((first,second),bench.balanced_members(list(reversed(rows))))

    def test_admission_protects_both_blocks_reproduction_and_slots(self):
        result = bench.projection(selected(), pilot(), available_gpu_hours=15., available_slots=3)
        self.assertTrue(result["admitted"])
        self.assertEqual(result["protected_reproduction_gpu_hours"], 1.5)
        for block, measured in zip(selected()["blocks"], result["blocks"]):
            self.assertAlmostEqual(measured["estimated_pair_seconds"], block["source_seconds"]*9*1.2+120)
        self.assertFalse(bench.projection(selected(), pilot(), available_gpu_hours=2., available_slots=3)["admitted"])
        self.assertFalse(bench.projection(selected(), pilot(2000, 2000), available_gpu_hours=23., available_slots=3)["admitted"])
        with self.assertRaisesRegex(ValueError, "Protect"):
            bench.projection(selected(), pilot(), available_gpu_hours=15., available_slots=2)
        profiled = pilot();profiled["conditions"]["candidate"]["profile_enabled"] = True
        with self.assertRaisesRegex(ValueError, "profiled"):
            bench.projection(selected(), profiled, available_gpu_hours=15., available_slots=3)

    def test_freeze_command_accepts_historical_baseline_candidate_only_cost_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/"models.json";model.write_text('{}')
            selection = root/"selection.json";bench.write(selection, selected())
            costs = pilot()
            costs.update(schema_version="bir_phase2_cost_evidence_v1",
                engine_conditions={"baseline": "optimized", "candidate": "accelerated"},
                code_sha256={version: {"fixture.py": "a"*64} for version in bench.VERSIONS},
                model_lock_sha256=bench.sha(model), baseline_measurement="historical_unprofiled_v1_optimized",
                candidate_measurement="fresh_unprofiled_accelerated_pilot", receipt_sha256={"historical": "b"*64, "fresh": "c"*64})
            evidence=root/"costs.json";bench.write(evidence,costs)
            output=root/"plan.json"
            command=["freeze", "--selection", str(selection), "--pilot", str(evidence),
                "--baseline-root", str(root), "--candidate-root", str(root), "--model-lock", str(model),
                "--available-gpu-hours", "15", "--available-slots", "3", "--release-gpu-hours", "1.5",
                "--target-speedup", "1.5", "--output", str(output)]
            with patch.object(bench,"code_snapshot",return_value={"fixture.py":"a"*64}):
                self.assertEqual(bench.main(command),0)
                plan=bench.read(output)
                bench.verify_digest(plan,"plan_sha256")
                self.assertTrue(plan['admission']['admitted'])
                self.assertEqual(plan['conditions'],costs['engine_conditions'])
                self.assertEqual(plan['pilot_receipt_sha256'],bench.sha(evidence))
                costs['engine_conditions']['candidate']='optimized';bench.write(evidence,costs)
                command[-1]=str(root/'changed.json')
                with self.assertRaisesRegex(ValueError,'binding differs'):
                    bench.main(command)
                costs['engine_conditions']['candidate']='accelerated';bench.write(evidence,costs)
                command[command.index('--target-speedup')+1]='2.0'
                with self.assertRaisesRegex(ValueError,'fixed at 1.5'):
                    bench.main(command)

    def test_counterbalanced_complete_summary_and_single_campaign_cost(self):
        plan, pairs = plan_and_pairs()
        accounting = [dict(allocation_key="one", gpus=3, elapsed_seconds=1000, final_state="COMPLETED"),
                      dict(allocation_key="two", gpus=3, elapsed_seconds=1000, final_state="COMPLETED")]
        result = bench.summarize(plan, pairs, accounting)
        self.assertEqual(result["baseline_over_candidate_wall_ratio"], 2.)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["conditions"]["baseline"]["verified_complete_block_episodes"], 11)
        self.assertAlmostEqual(result["allocated_gpu_hours"], 6000/3600)
        with self.assertRaisesRegex(ValueError, "unique"):
            bench.summarize(plan, pairs, accounting + [accounting[0]])
        with self.assertRaisesRegex(ValueError, "duplicate block"):
            bench.summarize(plan, pairs + [pairs[0]])

    def test_failed_or_missing_block_never_reduces_denominator_or_claims_speedup(self):
        plan, pairs = plan_and_pairs()
        pairs[1]["status"] = "failed"
        pairs[1]["conditions"] = {"candidate": dict(status="timeout", subprocess_wall_seconds=500.)}
        result = bench.summarize(plan, pairs)
        self.assertFalse(result["performance_claim_admitted"])
        self.assertNotIn("baseline_over_candidate_wall_ratio", result)
        for version in bench.VERSIONS:
            self.assertEqual(result["conditions"][version]["planned_episodes"], 11)
            self.assertEqual(result["conditions"][version]["verified_complete_block_episodes"], 5)
            self.assertEqual(result["conditions"][version]["episodes_outside_fully_verified_blocks"], 6)
        self.assertEqual(result["conditions"]["candidate"]["observed_subprocess_wall_seconds_including_failed_calls"], 800.)
        missing = bench.summarize(plan, pairs[:1])
        self.assertEqual(missing["conditions"]["candidate"]["episodes_outside_fully_verified_blocks"], 6)

    def test_actual_placement_and_order_cannot_change_after_freeze(self):
        plan, pairs = plan_and_pairs()
        changed = copy.deepcopy(pairs)
        changed[1]["conditions"]["candidate"]["parity"]["placement_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "placement differs"):
            bench.summarize(plan, changed)
        changed = copy.deepcopy(pairs);changed[1]["order"].reverse()
        with self.assertRaisesRegex(ValueError, "order changed"):
            bench.summarize(plan, changed)

    def test_baseline_markers_preserve_calls_results_failures_and_restore_method(self):
        calls = []
        def generate(value, *, setting):
            calls.append((value, setting))
            if value == "fail": raise RuntimeError("fixture failure")
            return (value, setting)
        class Backend:
            pipeline = SimpleNamespace(generate=generate)
            def segment(self, prepared): return self.pipeline.generate("segment", setting=7)
            def caption(self, prepared, segmentation): return self.pipeline.generate(segmentation, setting=9)
        original = Backend.segment
        prepared = SimpleNamespace(job={"id": "one"}, metadata={"input_sha256": {"main": "a"*64}})
        stream = io.StringIO()
        with redirect_stdout(stream), bench.baseline_request_markers(Backend):
            backend = Backend()
            self.assertEqual(backend.segment(prepared), ("segment", 7))
            self.assertEqual(backend.caption(prepared, "caption"), ("caption", 9))
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                backend.caption(prepared, "fail")
        self.assertIs(Backend.segment, original)
        self.assertIs(Backend.pipeline.generate, generate)
        self.assertEqual(calls, [("segment", 7), ("caption", 9), ("fail", 9)])
        events = [json.loads(line.removeprefix("BIR_RUNTIME ")) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(events), 6)
        self.assertEqual(events[-1]["status"], "failed")
        self.assertTrue(all(e["episode_id"] == "one" for e in events))

    def test_request_digest_requires_exact_identity_and_completed_markers(self):
        episodes = [dict(episode_id="one", source_sha256="a"*64)]
        identity = dict(episode_id="one", stage="caption", source_sha256={"main": "a"*64}, request_ordinal=0)
        events = [("BIR_RUNTIME ", dict(event="native_request_start", **identity)),
                  ("FLORA_RUNTIME ", dict(event="generation_start", input_tensor_digest="c"*64, input_tokens=5, output_limit=512)),
                  ("BIR_RUNTIME ", dict(event="native_request_end", status="complete", **identity))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"runtime.log"
            path.write_text("\n".join(prefix+json.dumps(event) for prefix,event in events))
            result = bench.request_inputs(path, episodes)
            self.assertEqual(list(result), [("one", "caption", 0)])
            path.write_text("\n".join(prefix+json.dumps(event) for prefix,event in events[:-1]))
            with self.assertRaisesRegex(ValueError, "cover"):
                bench.request_inputs(path, episodes)
            path.write_text("BIR_PROFILE {}\n")
            with self.assertRaisesRegex(ValueError, "profiling"):
                bench.request_inputs(path, episodes)
            events[0][1]["source_sha256"] = {"main": "b"*64}
            path.write_text("\n".join(prefix+json.dumps(event) for prefix,event in events))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                bench.request_inputs(path, episodes)

    def test_missing_complete_load_timer_is_not_invented(self):
        loads = dict(session_load_event_count=1, session_load_failed_event_count=0, session_load_event_seconds=10.)
        runtime = {"stages": {kind: dict(loads) for kind in ("tempseg", "captioner")}}
        with patch("scripts.compare_native.performance", return_value={"source_hours": 1.}):
            result = bench.metrics({}, runtime, 100., 30., 3)
            self.assertEqual(result["full_cold_model_load_seconds"], 20.)
            self.assertEqual(result["load_excluded_workflow_seconds"], 80.)
            self.assertEqual(result["condition_window_gpu_seconds_per_source_minute"], 600.)
            runtime["stages"]["captioner"]["session_load_event_seconds"] = None
            result = bench.metrics({}, runtime, 100., 30., 3)
            self.assertIsNone(result["full_cold_model_load_seconds"])
            self.assertIsNone(result["load_excluded_workflow_seconds"])

    def test_worker_without_allocation_preserves_failure_receipt_without_inference(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            env = dict(os.environ, SLURM_RESTART_COUNT="0")
            env.pop("SLURM_JOB_ID", None)
            result = subprocess.run([sys.executable, str(root/"scripts/phase2_benchmark.py"), "worker",
                "--plan", str(temp/"missing.json"), "--block", "0", "--version", "candidate",
                "--code-root", str(root), "--source", str(temp), "--manifest", str(temp/"missing.jsonl"),
                "--model-lock", str(root/"configs/native-models.json"), "--output", str(temp/"state")],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            receipt = bench.read(temp/"worker.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("authorized SLURM", receipt["error"])
            self.assertFalse((temp/"state").exists())


if __name__ == "__main__":
    unittest.main()
