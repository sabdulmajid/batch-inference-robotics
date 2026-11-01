#!/usr/bin/env python3
"""Finite phase-two selection, admission, and fresh paired native measurements.

This tool does not submit jobs. Keep run directories and worker logs private.
The same worker file instruments both explicitly selected engine checkouts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "d2701b08a08b88d18887abed81f71c8373478d7b"
DATASET_REVISION = "9cfb7bb98d60a58319f4b3d7b41374eedfb7980c"
DEVELOPMENT = {"homer_41", "homer_50", "homer_52"}
TASKS = {"homer_29": "chopping", "homer_33": "chopping", "homer_37": "chopping",
         "homer_38": "table_setting", "homer_39": "surface_cleaning",
         "homer_40": "surface_cleaning", "homer_48": "container_stacking",
         "homer_53": "shoelaces", "homer_56": "cloth_handling",
         "homer_59": "shelf_retrieval", "homer_60": "shelf_placement"}
VERSIONS = ("baseline", "candidate")
TARGET_SPEEDUP = 1.5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Expected a finite nonnegative number")
    return float(value)


def balanced_members(episodes):
    """Choose the exact source-duration minimum among the 462 five/six splits."""
    if len(episodes) != 11 or len({row["episode_id"] for row in episodes}) != 11:
        raise ValueError("Balance requires eleven unique episodes")
    rows = sorted(episodes, key=lambda row: row["episode_id"])
    durations = {row["episode_id"]: Decimal(str(row["duration_s"])) for row in rows}
    total = sum(durations.values())
    candidates = []
    for first in itertools.combinations(rows, 5):
        identities = {row["episode_id"] for row in first}
        second = [row for row in rows if row["episode_id"] not in identities]
        if min(len({row["task"] for row in first}), len({row["task"] for row in second})) < 3:
            continue
        difference = abs(total - 2 * sum(durations[identity] for identity in identities))
        tie = digest(["phase2-balance-v2", sorted(identities)])
        candidates.append(((difference, tie), (list(first), second)))
    if not candidates:
        raise ValueError("No five/six partition has three task labels in both blocks")
    return min(candidates, key=lambda row: row[0])[1]


def select_workload(index):
    """Use already validated metadata only; never inspect generated predictions."""
    if index["dataset"]["revision"] != DATASET_REVISION:
        raise ValueError("Selection requires the pinned WGO dataset")
    supported = [row for row in index["records"] if row.get("native_supported") is True]
    selected = [row for row in supported if row["id"] not in DEVELOPMENT]
    if len(supported) != 14 or {row["id"] for row in selected} != set(TASKS):
        raise ValueError("Supported source pool changed; review the protocol before selecting")
    episodes = []
    for row in sorted(selected, key=lambda row: row["id"]):
        if row.get("profile") != "human":
            raise ValueError("Frozen source eligibility changed")
        episodes.append(dict(episode_id=row["id"], source_sha256=row["source_sha256"],
            duration_s=number(row["duration_s"], positive=True), source_size=row["size"],
            profile="human", instruction="", instruction_sha256=hashlib.sha256(b"").hexdigest(),
            task=TASKS[row["id"]], independent_references_valid=row.get("references_valid") is True))
    total = sum(row["duration_s"] for row in episodes)
    members = balanced_members(episodes)
    blocks = []
    for block_index, rows in enumerate(members):
        if len({row["task"] for row in rows}) < 3:
            raise ValueError("Duration balancing produced insufficient task diversity")
        chosen = sorted(rows, key=lambda row: digest(["phase2-order-v1", row["episode_id"]]))
        blocks.append(dict(block=block_index, order=list(VERSIONS if block_index == 0 else reversed(VERSIONS)),
            episode_ids=[row["episode_id"] for row in chosen], source_seconds=sum(row["duration_s"] for row in chosen)))
    result = dict(schema_version="bir_phase2_selection_v1", dataset=index["dataset"],
        known_supported_episodes=14, excluded_development_episode_ids=sorted(DEVELOPMENT),
        selected_episodes=11, source_seconds=total, episodes=episodes, blocks=blocks,
        block_assignment_revision=2,
        previous_selection_sha256="3a283c389e765afe4cf0abe232a36c0f47dc205f8f1af86a5e884c3418278eb8",
        scheduling_decision="Before any phase-two GPU profile, pilot, or final prediction, replace the initial greedy assignment's 46.514999-second duration imbalance with an exhaustive source-only duration minimum. Preserve every source, both condition orders, five/six counts, cost headroom and resource limits. Previously extracted original videos remain valid.",
        selection_rule="All eleven previously unpredicted supported episodes. Enumerate all 462 five-episode subsets and their six-episode complements. Require at least three source task labels in each block. Minimize absolute source-duration difference with exact decimal arithmetic; break ties by SHA256 of ['phase2-balance-v2', sorted five-episode IDs]. Use the unchanged fixed hash order inside each block. Task labels come from source instructions and source-video metadata, not predictions. Both engines receive the same explicit empty instruction.",
        scope_limit="Eleven unseen episodes fall below the 12–20 target because the pinned validated pool has fourteen supported episodes, three used for development. Each block stays below the old eight-job metadata capacity; a separate CPU regression covers that limit.")
    result["selection_sha256"] = digest(result)
    return result


def verify_digest(value, field):
    body = {key: item for key, item in value.items() if key != field}
    if value.get(field) != digest(body):
        raise ValueError(field + " mismatch")


def code_snapshot(root, *, baseline=False):
    root = Path(root).resolve(strict=True)
    paths = sorted((root / "src/bir").glob("*.py"))
    if len(paths) < 10:
        raise ValueError("Engine checkout is incomplete")
    result = {path.name: sha(path) for path in paths}
    if baseline:
        # Freeze locally against immutable Git objects. Remote workers need only hashes.
        names = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "--name-only",
            BASELINE_COMMIT + ":src/bir"], text=True).splitlines()
        expected = {name: hashlib.sha256(subprocess.check_output(["git", "-C", str(ROOT), "show",
            BASELINE_COMMIT + ":src/bir/" + name])).hexdigest() for name in names if name.endswith(".py")}
        if result != expected:
            raise ValueError("Baseline engine files differ from immutable d2701b0")
    return result


def harness_hashes():
    return {name: sha(ROOT / "scripts" / name) for name in
            ("phase2_benchmark.py", "native_benchmark.py", "compare_native.py")}


def projection(selection, pilot, *, available_gpu_hours, available_slots,
               release_gpu_hours=1.5, headroom=1.2):
    """Conservative whole-workflow scaling; estimates are never measurements."""
    verify_digest(selection, "selection_sha256")
    if (pilot.get("status") != "complete" or pilot.get("phase") != "pilot" or
            not pilot.get("comparison", {}).get("compatible") or
            not pilot.get("comparison", {}).get("tensor_input_agreement_verified")):
        raise ValueError("Admission requires a complete fresh paired pilot")
    if (type(available_slots) is not int or not 3 <= available_slots <= 16 or
            number(release_gpu_hours) < 1.5 or number(headroom) < 1.2 or number(available_gpu_hours) > 24):
        raise ValueError("Protect two comparison slots and at least one reproduction slot / 1.5 GPU-hours")
    rates = {}
    for version in VERSIONS:
        row = pilot["conditions"][version]
        if row["status"] != "fresh_complete" or row["gpu_count"] != 3 or row["profile_enabled"]:
            raise ValueError("Pilot is incomplete, differently sized, or profiled")
        rates[version] = number(row["subprocess_wall_seconds"], positive=True) / number(pilot["source_seconds"], positive=True)
    blocks = []
    for block in selection["blocks"]:
        estimate = headroom * block["source_seconds"] * sum(rates.values()) + 120
        minutes = math.ceil(estimate / 60)
        blocks.append(dict(block=block["block"], estimated_pair_seconds=estimate,
                           requested_minutes=minutes, reserved_gpu_hours=minutes * 3 / 60))
    requested = sum(row["reserved_gpu_hours"] for row in blocks)
    return dict(estimate_not_measurement=True, headroom_factor=headroom,
        controller_allowance_seconds_per_block=120, rates_seconds_per_source_second=rates,
        blocks=blocks, comparison_reserved_gpu_hours=requested,
        protected_reproduction_gpu_hours=release_gpu_hours, protected_reproduction_slots=1,
        available_gpu_hours=number(available_gpu_hours), available_submission_slots=available_slots,
        admitted=all(row["requested_minutes"] <= 120 for row in blocks) and requested + release_gpu_hours <= available_gpu_hours,
        rule="Both frozen blocks must fit their 120-minute limits and the remaining ledger budget with reproduction protected. No shortening or removal after final predictions.")


def stable_parity(parity):
    return dict(environment=parity["environment"], placement_sha256=parity["placement_sha256"],
        hardware={key: value for key, value in parity["hardware"].items() if key != "allocation_identity_sha256"})


@contextmanager
def baseline_request_markers(backend_type):
    """Add candidate-equivalent identity logging without changing baseline files."""
    def instrument(method, stage):
        def call(backend, prepared, *args, **kwargs):
            original_generate = backend.pipeline.generate
            ordinal = 0
            def generate(*values, **options):
                nonlocal ordinal
                identity = dict(episode_id=prepared.job.get("id", prepared.job.get("episode_id", "unknown")),
                    stage=stage, request_ordinal=ordinal, source_sha256=prepared.metadata.get("input_sha256", {}))
                ordinal += 1
                def emit(event, **extra):
                    print("BIR_RUNTIME " + json.dumps(dict(event=event, utc=datetime.now(timezone.utc).isoformat(),
                        **extra, **identity), sort_keys=True, allow_nan=False), flush=True)
                emit("native_request_start")
                try:
                    result = original_generate(*values, **options)
                except BaseException as error:
                    emit("native_request_end", status="failed", error_type=type(error).__name__)
                    raise
                emit("native_request_end", status="complete")
                return result
            with patch.object(backend.pipeline, "generate", side_effect=generate):
                return method(backend, prepared, *args, **kwargs)
        return call
    with ExitStack() as stack:
        for stage in ("segment", "caption"):
            stack.enter_context(patch.object(backend_type, stage, instrument(getattr(backend_type, stage), stage)))
        yield


def request_inputs(log, episodes):
    """Bind each input tensor to episode/stage/ordinal, not just a multiset."""
    expected = {r["episode_id"]: {"main": r["source_sha256"]} for r in episodes}
    current = None
    result = {}
    for line in Path(log).read_text().splitlines():
        if line.startswith("BIR_PROFILE "):
            raise ValueError("Detailed profiling must be off in matched measurements")
        if line.startswith("BIR_RUNTIME "):
            event = json.loads(line[len("BIR_RUNTIME "):])
            if event.get("event") not in {"native_request_start", "native_request_end"}:
                continue
            key = (event["episode_id"], event["stage"], event["request_ordinal"])
            if expected.get(key[0]) != event["source_sha256"] or key[1] not in {"segment", "caption"} or type(key[2]) is not int or key[2] < 0:
                raise ValueError("Native request source/stage identity differs from the frozen workload")
            if event["event"] == "native_request_start":
                if current is not None or key in result:
                    raise ValueError("Duplicate or nested native request marker")
                current = key
                result[key] = []
            else:
                if current != key or event.get("status") != "complete" or not result[key]:
                    raise ValueError("Incomplete native request trace")
                current = None
        elif line.startswith("FLORA_RUNTIME "):
            event = json.loads(line[len("FLORA_RUNTIME "):])
            if event.get("event") == "generation_start":
                if current is None:
                    raise ValueError("Generation has no native request identity")
                result[current].append({key: event[key] for key in ("input_tensor_digest", "input_tokens", "output_limit")})
    if current is not None or not result or {key[0] for key in result} != set(expected):
        raise ValueError("Request trace does not cover every frozen episode")
    return result


def summarize(plan, pairs, accounting=None):
    """Keep all frozen episodes in coverage even after a missing/failed condition."""
    verify_digest(plan, "plan_sha256")
    by_block = {}
    for pair in pairs:
        if pair["plan_sha256"] != plan["plan_sha256"] or pair["block"] in by_block:
            raise ValueError("Changed plan or duplicate block report")
        block = plan["blocks"][pair["block"]]
        if pair["episode_ids"] != block["episode_ids"] or pair["order"] != block["order"]:
            raise ValueError("Counterbalanced block identity/order changed")
        by_block[pair["block"]] = pair
    total_seconds = sum(row["duration_s"] for row in plan["episodes"])
    report = dict(schema_version="bir_phase2_comparison_v1", plan_sha256=plan["plan_sha256"],
        planned_unique_episodes=len(plan["episodes"]), planned_unique_source_seconds=total_seconds,
        conditions={}, blocks=[], performance_claim_admitted=False, status="incomplete",
        replication="Two counterbalanced disjoint blocks; one paired trial per episode. Report both block ratios and pooled wall ratio. No repeat-based confidence interval or population generalization.",
        compatibility_is_annotation_quality=False, allocated_gpu_hours=None)
    complete = len(by_block) == len(plan["blocks"])
    for version in VERSIONS:
        rows = [by_block.get(b["block"], {}).get("conditions", {}).get(version, {}) for b in plan["blocks"]]
        finished = sum(len(b["episode_ids"]) for b, row in zip(plan["blocks"], rows) if row.get("status") == "fresh_complete")
        condition_complete = all(row.get("status") == "fresh_complete" for row in rows)
        wall = sum(number(row.get("subprocess_wall_seconds", 0)) for row in rows)
        report["conditions"][version] = dict(planned_episodes=len(plan["episodes"]),
            verified_complete_block_episodes=finished, episodes_outside_fully_verified_blocks=len(plan["episodes"])-finished,
            coverage_limit="Counts are whole verified blocks. Failed blocks can retain successful partial stages in their private state; these are not silently treated as a complete performance sample.",
            observed_subprocess_wall_seconds_including_failed_calls=wall,
            whole_workflow_seconds=wall if condition_complete else None,
            condition_window_gpu_seconds_per_source_minute=wall*3/(total_seconds/60) if condition_complete else None)
        complete = complete and condition_complete
    for block in plan["blocks"]:
        pair = by_block.get(block["block"])
        ok = bool(pair and pair["status"] == "complete" and pair.get("comparison", {}).get("compatible")
                  and pair.get("comparison", {}).get("tensor_input_agreement_verified")
                  and pair.get("request_input_identity_agreement") is True)
        if ok:
            for version in VERSIONS:
                if stable_parity(pair["conditions"][version]["parity"]) != plan["pilot_execution_parity"]:
                    raise ValueError("Final hardware/environment/placement differs from the admitted pilot")
        complete = complete and ok
        report["blocks"].append(dict(block=block["block"], order=block["order"], complete=ok,
            conditions={version: {key: row[key] for key in
                ("status", "subprocess_wall_seconds", "metrics", "gpu_memory", "request_input_identity_sha256") if key in row}
                for version, row in (pair or {}).get("conditions", {}).items()},
            baseline_over_candidate_wall_ratio=(pair["conditions"]["baseline"]["subprocess_wall_seconds"] /
                pair["conditions"]["candidate"]["subprocess_wall_seconds"]) if ok else None))
    if complete:
        baseline, candidate = (report["conditions"][v]["whole_workflow_seconds"] for v in VERSIONS)
        report.update(status="complete", performance_claim_admitted=True,
            baseline_over_candidate_wall_ratio=baseline/candidate,
            candidate_wall_reduction_percent=100*(1-candidate/baseline),
            frozen_target_speedup=plan["target_speedup"], target_met=baseline/candidate >= plan["target_speedup"])
    if accounting is not None:
        try:
            from .compare_native import allocation_gpu_hours
        except ImportError:
            from compare_native import allocation_gpu_hours
        report["allocated_gpu_hours"] = allocation_gpu_hours(accounting)
        report["allocation_count"] = len(accounting)
        report["campaign_allocated_gpu_seconds_per_unique_source_minute"] = report["allocated_gpu_hours"]*3600/(total_seconds/60)
        report["allocation_boundary"] = "Both versions and all controller, failed, idle and cleanup time, charged once per complete parent attempt; not a per-version bill."
    return report


def metrics(run, runtime, elapsed, source_seconds, gpu_count):
    try:
        from .compare_native import performance
    except ImportError:
        from compare_native import performance
    result = performance(run)
    loads = [runtime["stages"][kind] for kind in ("tempseg", "captioner")]
    complete_loads = all(row["session_load_event_count"] == 1 and row["session_load_failed_event_count"] == 0
                         and row["session_load_event_seconds"] is not None for row in loads)
    cold = sum(row["session_load_event_seconds"] for row in loads) if complete_loads else None
    if cold is not None and not 0 <= cold < elapsed:
        raise ValueError("Cold-load time exceeds the enclosing process boundary")
    result.update(subprocess_wall_seconds=elapsed, planned_source_seconds=source_seconds,
        source_minutes_per_wall_minute=source_seconds/elapsed,
        condition_window_gpu_seconds_per_source_minute=elapsed*gpu_count/(source_seconds/60),
        full_cold_model_load_seconds=cold,
        load_excluded_workflow_seconds=elapsed-cold if cold is not None else None,
        load_excluded_source_minutes_per_wall_minute=source_seconds/(elapsed-cold) if cold is not None else None,
        sustained_measurement_limit="Load-excluded workflow is a derived diagnostic, not an independently timed steady-state run. It retains process/CLI and other overhead. Cold-load work can overlap CPU preparation. Primary comparison uses full subprocess wall time.",
        gpu_cost_limit="Condition-window GPU seconds multiply visible GPU count by subprocess wall. They measure held-device capacity, not GPU utilization or separately billed allocations. Coordinator reconciles full parent allocations once.",
        runtime=runtime)
    telemetry = run.get("summary", {}).get("telemetry", {})
    result["engine_work_telemetry"] = {key: telemetry[key] for key in
        ("cpu_preparation", "backends", "prepared_metadata_spool", "prefetch", "prepared_metadata_cache_capacity",
         "cpu_timing_scope", "decoded_cache_scope") if key in telemetry}
    identities = {row["request_id"]: row["id"] for row in run.get("summary", {}).get("records", [])}
    result["stage_attempts"] = [dict(episode_id=identities[row["request_id"]], stage=row["stage"],
        elapsed_s=row.get("elapsed_s"), state=row["state"], number=row["number"])
        for row in run.get("summary", {}).get("attempts", [])]
    return result


def _imports(code_root):
    # Must run before any bir import in a fresh worker process.
    sys.path.insert(0, str(Path(code_root).resolve() / "src"))
    sys.path.insert(1, str(ROOT / "scripts"))


def _jobs(manifest, expected=None):
    from bir.manifest import load_manifest
    jobs = load_manifest(manifest, contract={"phase2_source_validation": 1})
    if expected is not None:
        actual = [(j.id, dict(j.source_sha256), j.profile, j.yam32,
                   hashlib.sha256(j.instruction.encode()).hexdigest()) for j in jobs]
        wanted = [(r["episode_id"], {"main": r["source_sha256"]}, "human", False, r["instruction_sha256"]) for r in expected]
        if actual != wanted:
            raise ValueError("Manifest IDs, order, source bytes, profiles, or instructions differ from the freeze")
    return jobs


def worker(args):
    started = time.monotonic()
    _imports(args.code_root)
    from bir.cli import execute_run
    from bir.native import NativeBackend
    from native_benchmark import _guard_allocation, _hardware, EXPECTED
    report = dict(status="started", gpu_memory=[])
    torch = None
    try:
        _guard_allocation()
        plan = read(args.plan)
        verify_digest(plan, "plan_sha256")
        if harness_hashes() != plan["harness_sha256"] or code_snapshot(args.code_root) != plan["code_sha256"][args.version]:
            raise ValueError("Frozen code or common worker instrumentation changed")
        if sha(args.model_lock) != plan["model_lock_sha256"]:
            raise ValueError("Model lock changed")
        expected = [next(r for r in plan["episodes"] if r["episode_id"] == identity) for identity in plan["blocks"][args.block]["episode_ids"]]
        _jobs(args.manifest, expected)
        environment = {name: importlib.metadata.version(name) for name in EXPECTED}
        if any(environment[name].split("+")[0] != value for name, value in EXPECTED.items()):
            raise ValueError("Worker environment differs from pinned native dependencies")
        import torch as framework
        torch = framework
        if not torch.cuda.is_available() or torch.cuda.device_count() != 3 or torch.version.cuda != "12.8" or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 4:
            raise ValueError("Paired measurements require three GPUs, four CPUs, and CUDA 12.8")
        report.update(environment=environment, hardware=_hardware(torch), profile_enabled=False)
        for device in range(3):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if args.output.exists():
            raise ValueError("Worker state must be fresh")
        invocation = argparse.Namespace(source=args.source, manifest=args.manifest,
            model_lock=args.model_lock, captioner_headroom_gib=4., batch_size=1,
            condition=plan["conditions"][args.version], run_dir=args.output, stages="all")
        with ExitStack() as hooks:
            if args.version == "baseline":
                hooks.enter_context(baseline_request_markers(NativeBackend))
            summary, code = execute_run(invocation, time.monotonic())
        report.update(status="complete" if code == 0 else "failed", summary=summary)
    except Exception as error:
        code = 1
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        if torch is not None and torch.cuda.is_available():
            for device in range(3):
                try:
                    torch.cuda.synchronize(device)
                    report["gpu_memory"].append(dict(logical_device=device,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
                except Exception as error:
                    code = 1
                    report.update(status="failed", memory_measurement_error=type(error).__name__)
        report["worker_wall_seconds"] = time.monotonic()-started
        write(args.output.parent / "worker.json", report)
    return code


def run_pair(args, plan):
    _imports(args.candidate_root)
    from native_benchmark import _guard_allocation, verify_fresh_run, runtime_timings, placement_signature
    from compare_native import compare
    _guard_allocation()
    verify_digest(plan, "plan_sha256")
    if plan["phase"] == "final" and not plan["admission"]["admitted"]:
        raise ValueError("Frozen final plan was not admitted")
    if harness_hashes() != plan["harness_sha256"]:
        raise ValueError("Common worker instrumentation changed after freeze")
    for version in VERSIONS:
        if code_snapshot(getattr(args, version + "_root")) != plan["code_sha256"][version]:
            raise ValueError("Engine changed after freeze")
    if args.output.exists():
        raise ValueError("Use a fresh pair output directory")
    args.output.mkdir(parents=True)
    write(args.output / "plan.json", plan)
    block = plan["blocks"][args.block]
    expected = [next(row for row in plan["episodes"] if row["episode_id"] == identity) for identity in block["episode_ids"]]
    _jobs(args.manifest, expected)
    started = time.monotonic()
    limit = plan["admission"]["blocks"][args.block]["requested_minutes"] * 60
    report = dict(schema_version="bir_phase2_pair_v1", phase=plan["phase"], status="started",
        plan_sha256=plan["plan_sha256"], block=args.block, order=block["order"],
        episode_ids=block["episode_ids"], source_seconds=block["source_seconds"],
        planned_episode_count=len(expected), conditions={}, allocated_gpu_hours=None,
        accounting="Coordinator charges the complete shared allocation once, including both conditions, failures and idle time.")
    write(args.output / "pair-summary.json", report)
    try:
        parity = None
        for version in block["order"]:
            directory = args.output / version
            directory.mkdir()
            log = directory / "runtime.log"
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan", str(args.output / "plan.json"),
                "--block", str(args.block), "--version", version, "--code-root", str(getattr(args, version + "_root")),
                "--source", str(args.source), "--manifest", str(args.manifest), "--model-lock", str(args.model_lock),
                "--output", str(directory / "state")]
            remaining = limit - (time.monotonic()-started) - 30
            if remaining <= 0:
                raise TimeoutError("No remaining bounded worker time")
            begin = time.monotonic()
            row = report["conditions"][version] = dict(status="started", gpu_count=3, profile_enabled=False)
            write(args.output / "pair-summary.json", report)
            try:
                with log.open("w") as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=remaining, check=False)
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
                raise
            finally:
                row["subprocess_wall_seconds"] = time.monotonic()-begin
                write(args.output / "pair-summary.json", report)
            if process.returncode:
                row["status"] = "failed"
                raise RuntimeError(version + " failed; retain its full denominator and private evidence")
            if (harness_hashes() != plan["harness_sha256"] or
                    code_snapshot(getattr(args, version + "_root")) != plan["code_sha256"][version]):
                raise ValueError("Engine or harness changed during worker execution")
            measured = read(directory / "worker.json")
            if measured["status"] != "complete" or measured["profile_enabled"]:
                raise ValueError("Worker was incomplete or had unmatched profiling")
            run = verify_fresh_run(directory / "state", len(expected))
            if set(run["outputs"]) != set(block["episode_ids"]):
                raise ValueError("Worker completed the wrong episode set")
            for episode in expected:
                if abs(run["outputs"][episode["episode_id"]]["duration_s"]-episode["duration_s"]) > 1e-5:
                    raise ValueError("Measured source duration differs from frozen source metadata")
            signature, details = placement_signature(run["outputs"])
            current = dict(hardware=measured["hardware"], environment=measured["environment"], placement_sha256=signature)
            if plan["phase"] == "final" and stable_parity(current) != plan["pilot_execution_parity"]:
                raise ValueError("Hardware, environment, or placement changed from measured pilot")
            if parity is not None and current != parity:
                raise ValueError("Hardware, environment, or placement differ within the pair")
            parity = current
            row.update(status="fresh_complete", metrics=metrics(run, runtime_timings(log), row["subprocess_wall_seconds"],
                block["source_seconds"], 3), gpu_memory=measured["gpu_memory"], parity=current,
                placement=details, runtime_sha256=sha(log), worker_sha256=sha(directory / "worker.json"))
            row["request_input_identity_sha256"] = digest([
                [list(key), values] for key, values in sorted(request_inputs(log, expected).items())])
            write(args.output / "pair-summary.json", report)
        if plan["phase"] == "pilot" and block["order"] == ["candidate"]:
            report["status"] = "complete_candidate_pilot"
            report["compatibility_pending"] = "Compare native outputs and input tensors with the accepted original development baseline before final admission."
        else:
            comparison = compare(args.output / "baseline/state", args.output / "candidate/state", stage="all",
                left_logs=[args.output / "baseline/runtime.log"], right_logs=[args.output / "candidate/runtime.log"])
            report["comparison"] = comparison
            if not comparison["compatible"] or not comparison["tensor_input_agreement_verified"]:
                raise ValueError("Native outputs or input tensors changed; inspect video before any quality claim")
            if report["conditions"]["baseline"]["request_input_identity_sha256"] != report["conditions"]["candidate"]["request_input_identity_sha256"]:
                raise ValueError("Native tensors differ by episode/stage/request identity")
            report["request_input_identity_agreement"] = True
            report["status"] = "complete"
        code = 0
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        code = 1
    report["pair_wall_seconds"] = time.monotonic()-started
    write(args.output / "pair-summary.json", report)
    return code


def base_plan(args, selection, phase):
    return dict(schema_version="bir_phase2_plan_v1", phase=phase, selection_sha256=selection.get("selection_sha256"),
        episodes=selection["episodes"], blocks=selection["blocks"],
        baseline_commit=BASELINE_COMMIT, conditions=dict(baseline="optimized", candidate="accelerated"),
        baseline_git_verified_at_freeze=phase == "final",
        code_sha256=dict(baseline=code_snapshot(args.baseline_root, baseline=phase == "final"), candidate=code_snapshot(args.candidate_root)),
        model_lock_sha256=sha(args.model_lock), harness_sha256=harness_hashes(),
        instrumentation="Same external worker; full native input digests and value checks on; optional profiler off. Baseline receives candidate-equivalent episode/stage/source/ordinal markers from the external harness; immutable engine files remain unchanged.",
        hardware=dict(gpus=3, cpus=4, host_ram_gib=128, captioner_headroom_gib=4, batch_size=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select")
    selection.add_argument("--index", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--pilot", type=Path, required=True)
    freeze.add_argument("--available-gpu-hours", type=float, required=True)
    freeze.add_argument("--available-slots", type=int, required=True)
    freeze.add_argument("--release-gpu-hours", type=float, default=1.5)
    freeze.add_argument("--target-speedup", type=float, required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--episodes", type=Path, required=True, help="Metadata-only development episode rows")
    pilot.add_argument("--minutes", type=int, default=30)
    pilot.add_argument("--candidate-only", action="store_true", help="Use accepted historical baseline for later cost estimation; does not claim paired compatibility")
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--block", type=int, choices=(0, 1), required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    summary.add_argument("--pairs", type=Path, nargs="+", required=True)
    summary.add_argument("--accounting", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    for item in (freeze, pilot, run):
        item.add_argument("--baseline-root", type=Path, required=True)
        item.add_argument("--candidate-root", type=Path, required=True)
        item.add_argument("--model-lock", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    for item in (pilot, run):
        item.add_argument("--source", type=Path, required=True)
        item.add_argument("--manifest", type=Path, required=True)
    internal = commands.add_parser("worker", help=argparse.SUPPRESS)
    for name in ("plan", "code-root", "source", "manifest", "model-lock", "output"):
        internal.add_argument("--" + name, type=Path, required=True)
    internal.add_argument("--block", type=int, required=True)
    internal.add_argument("--version", choices=VERSIONS, required=True)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.command == "worker":
        return worker(args)
    if args.command == "run":
        return run_pair(args, read(args.plan))
    if args.output.exists():
        raise ValueError("Preserve the existing artifact; select a fresh output")
    if args.command == "select":
        write(args.output, select_workload(read(args.index)))
    elif args.command == "summarize":
        write(args.output, summarize(read(args.plan), [read(path) for path in args.pairs], read(args.accounting) if args.accounting else None))
    elif args.command == "freeze":
        selected, measured = read(args.selection), read(args.pilot)
        plan = base_plan(args, selected, "final")
        if measured.get("schema_version") == "bir_phase2_cost_evidence_v1":
            if (any(plan[key] != measured[key] for key in ("code_sha256", "model_lock_sha256")) or
                    plan["conditions"] != measured.get("engine_conditions")):
                raise ValueError("Final engine/model binding differs from measured pilot cost evidence")
            if measured.get("baseline_measurement") != "historical_unprofiled_v1_optimized" or measured.get("candidate_measurement") != "fresh_unprofiled_accelerated_pilot" or not measured.get("receipt_sha256"):
                raise ValueError("Combined pilot cost evidence lacks explicit measurement provenance")
            plan["cost_evidence_limit"] = "Historical unprofiled v1 and a fresh unprofiled candidate pilot size the allocation. Their wrapper boundaries can differ; the 1.2 headroom and controller allowance are estimates. Final performance uses the identical common paired worker only. The separate cProfile run is excluded from these rates."
        else:
            pilot_plan = read(args.pilot.parent / "plan.json")
            verify_digest(pilot_plan, "plan_sha256")
            if any(plan[key] != pilot_plan[key] for key in ("code_sha256", "conditions", "model_lock_sha256", "harness_sha256")) or measured["plan_sha256"] != pilot_plan["plan_sha256"]:
                raise ValueError("Final code/instrumentation differs from measured paired pilot")
        if stable_parity(measured["conditions"]["baseline"]["parity"]) != stable_parity(measured["conditions"]["candidate"]["parity"]):
            raise ValueError("Measured pilot code uses unmatched placement/environment/hardware")
        if args.target_speedup != TARGET_SPEEDUP:
            raise ValueError("The pre-final phase-two target is fixed at 1.5x full-workflow throughput")
        plan.update(target_speedup=args.target_speedup, pilot_receipt_sha256=sha(args.pilot),
            pilot_execution_parity=stable_parity(measured["conditions"]["baseline"]["parity"]),
            admission=projection(selected, measured, available_gpu_hours=args.available_gpu_hours,
                available_slots=args.available_slots, release_gpu_hours=args.release_gpu_hours))
        plan["plan_sha256"] = digest(plan)
        write(args.output, plan)
        if not plan["admission"]["admitted"]:
            return 2
    elif args.command == "pilot":
        episodes = read(args.episodes)
        if not 1 <= len(episodes) <= 3 or not {row["episode_id"] for row in episodes} <= DEVELOPMENT or not 1 <= args.minutes <= 120:
            raise ValueError("Pilot uses one to three existing development episodes within the job cap")
        selected = dict(episodes=episodes, blocks=[dict(block=0, order=["candidate"] if args.candidate_only else list(VERSIONS),
            episode_ids=[row["episode_id"] for row in episodes], source_seconds=sum(number(row["duration_s"], positive=True) for row in episodes))])
        plan = base_plan(args, selected, "pilot")
        plan["admission"] = dict(admitted=True, blocks=[dict(requested_minutes=args.minutes)])
        plan["plan_sha256"] = digest(plan)
        args.block = 0
        return run_pair(args, plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(status="failed", error_type=type(error).__name__, error=str(error))), file=sys.stderr)
        raise SystemExit(1)
