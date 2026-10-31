#!/usr/bin/env python3
"""Bounded fresh native A/B/C runs inside one coordinator-owned allocation.

This script never submits jobs. Gate C evidence must exist first. Raw outputs,
logs, and error messages belong in private project storage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from bir.cli import atomic_json, build_execution_contract, execute_run
from bir.manifest import canonical_json, file_sha256, load_manifest
from bir.native import MODEL_REVISIONS, UPSTREAM_COMMIT, verify_source
from bir.resource_ledger import reject_restarted_execution

try:
    from .compare_native import compare, load_run, performance, read_json
except ImportError:
    from compare_native import compare, load_run, performance, read_json

CONDITIONS = ("reference", "persistent", "optimized")
EXPECTED = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.8.1",
            "huggingface-hub": "1.33.0", "peft": "0.18.1", "accelerate": "1.12.0",
            "safetensors": "0.7.0", "av": "16.0.1"}


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def workload_signature(jobs):
    return digest([dict(id=job.id, source_sha256=dict(job.source_sha256), profile=job.profile,
                        yam32=job.yam32, instruction=job.instruction) for job in jobs])


def runner_source_hashes():
    return {path.name: file_sha256(path) for path in
            (Path(__file__).resolve(), Path(__file__).resolve().with_name("compare_native.py"))}


def runtime_timings(path):
    """Keep native load and generation event boundaries separate from wall time."""
    stages = {stage: dict(model_load_count=0, model_ready_count=0, model_load_event_seconds=0.0,
                          generation_call_count=0, generation_event_seconds=0.0)
              for stage in ("tempseg", "captioner")}
    current = None
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith("FLORA_RUNTIME "):
                continue
            event = json.loads(line[len("FLORA_RUNTIME "):])
            kind = event.get("event")
            if kind == "model_load_start":
                current = event.get("kind")
                if current not in stages:
                    raise ValueError("Unexpected model stage in native runtime trace")
                stages[current]["model_load_count"] += 1
            elif kind in {"model_ready", "generation_end"}:
                selected = event.get("kind") if kind == "model_ready" else current
                elapsed = event.get("elapsed_s")
                if (selected not in stages or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
                        not math.isfinite(elapsed) or elapsed < 0):
                    raise ValueError("Native runtime event has no valid stage/time binding")
                if kind == "model_ready":
                    stages[selected]["model_ready_count"] += 1
                    stages[selected]["model_load_event_seconds"] += elapsed
                else:
                    stages[selected]["generation_call_count"] += 1
                    stages[selected]["generation_event_seconds"] += elapsed
    return dict(stages=stages,
                timing_boundary="Native rounded event times: load includes asset checks and parameter verification; generation excludes preparation, input hashing, and device transfer; do not add them again to total wall time")


def verify_gate_c(path):
    """Require saved full annotations, not a framework or TempSeg-only marker."""
    path = Path(path).resolve(strict=True)
    marker = read_json(path)
    if (marker.get("status") != "complete" or marker.get("native_caption_gate") is not True or
            marker.get("fresh_inference") is not True or marker.get("run", {}).get("complete") is not True):
        raise ValueError("Gate C requires a complete native caption probe and complete run")
    fresh_captions = {row.get("request_id") for row in marker["run"].get("records", [])
                      if row.get("stage") == "caption" and row.get("status") == "complete" and
                      row.get("fresh_execution") is True}
    artifacts = marker["run"].get("exports", [])
    if not artifacts:
        raise ValueError("Gate C has no serialized annotation artifacts")
    reviewed = []
    for value in artifacts:
        candidate = Path(value)
        artifact = candidate if candidate.is_absolute() else path.parent / candidate
        output = read_json(artifact)
        engine = output.get("engine", {})
        residency, audits = engine.get("residency", {}), engine.get("model_verification", {})
        if engine.get("source_commit") != UPSTREAM_COMMIT:
            raise ValueError("Gate C used a different source revision")
        if output.get("request_id") not in fresh_captions or output.get("request_id") is None:
            raise ValueError("Gate C artifact has no successful fresh caption attempt")
        if engine.get("mode") != "placement_adapted_native" or engine.get("request_batch_size") != 1:
            raise ValueError("Gate C used an experimental or unverified input/generation mode")
        if any(output.get("model_revisions", {}).get(kind, {}).get("revision") != revision
               for kind, revision in MODEL_REVISIONS.items()):
            raise ValueError("Gate C used different model or processor revisions")
        if not all(stage in audits and residency.get(stage, {}).get("all_parameters_and_nonempty_buffers_on_cuda") is True
                   for stage in ("tempseg", "captioner")):
            raise ValueError("Gate C lacks native verification and GPU residency for both stages")
        tempseg = audits["tempseg"]
        adapter = tempseg.get("ADAPTER_VALUE_VERIFICATION.json", {})
        loaded = tempseg.get("MODEL_LOADED.json", {})
        if (adapter.get("all_saved_values_exact_before_and_after_placement") is not True or
                adapter.get("holders_fp32_before_assignment") is not True or adapter.get("merged") is not False or
                loaded.get("gdn_values_exact") is not True or loaded.get("adapter_persistent_dtype") != "torch.float32"):
            raise ValueError("Gate C lacks exact unmerged FP32 adapter verification")
        for receipt in (tempseg.get("GDN_VALUE_VERIFICATION.json", {}), audits["captioner"].get("gdn", {})):
            if (receipt.get("all_values_equal_saved_checkpoint") is not True or
                    receipt.get("all_runtime_fp32") is not True or not isinstance(receipt.get("tensors"), int) or
                    receipt["tensors"] < 1):
                raise ValueError("Gate C lacks exact native Gated DeltaNet parameter verification")
        for stage, info in (("tempseg", loaded.get("base_loading_info", {})),
                            ("captioner", audits["captioner"].get("loading_info", {}))):
            required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
            if not required <= set(info) or any(info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("Gate C has incomplete checkpoint loading evidence")
            if info["unexpected_keys"] and (stage == "tempseg" or any("_extra_state" not in key for key in info["unexpected_keys"])):
                raise ValueError("Gate C checkpoint loading has unexpected parameter keys")
        rows = output.get("segments", [])
        if not rows or any(row.get("status") != "complete" or not row.get("caption", "").strip() for row in rows):
            raise ValueError("Gate C has no complete nonempty native annotation")
        if not output.get("pipeline", {}).get("temporal_windows") or not output.get("pipeline", {}).get("caption_inputs"):
            raise ValueError("Gate C lacks temporal and caption input evidence")
        reviewed.append(dict(episode_id=output.get("episode_id"), sha256=file_sha256(artifact)))
    return dict(marker_sha256=file_sha256(path), annotation_artifacts=reviewed,
                source_commit=UPSTREAM_COMMIT, scheduler_final_status="must_be_confirmed_by_coordinator")


def verify_fresh_run(directory, episode_count):
    run = load_run(directory, "all")
    if run["errors"]:
        raise ValueError("A condition is not a complete fresh native run: " + canonical_json(run["errors"]))
    summary = run["summary"]
    if len(run["outputs"]) != episode_count or summary.get("fresh_stage_attempts") != episode_count * 2:
        raise ValueError("Fresh condition coverage or attempt count differs from the manifest")
    if any(row.get("state") != "complete" or row.get("number") != 1 for row in summary.get("attempts", [])):
        raise ValueError("A benchmark condition had a failed or repeated attempt; stop and diagnose")
    return run


def verify_resume(summary, episode_count, log_path):
    if (summary.get("complete") is not True or summary.get("fresh_stage_attempts") != 0 or
            summary.get("attempts") or summary.get("resumed_stages") != episode_count * 2):
        raise ValueError("Cache-only resume unexpectedly performed fresh or incomplete work")
    records = summary.get("records", [])
    if len(records) != episode_count * 2 or any(row.get("status") != "resumed" or row.get("fresh_execution") is not False
                                               for row in records):
        raise ValueError("Resume did not preserve every episode/stage identity")
    with Path(log_path).open() as stream:
        for line in stream:
            if line.startswith("FLORA_RUNTIME "):
                event = json.loads(line[len("FLORA_RUNTIME "):])
                if event.get("event") in {"model_load_start", "generation_start"}:
                    raise ValueError("Cache-only resume loaded a model or executed generation")


def placement_signature(outputs):
    """Compare actual final device placement and dtype bytes, not free-memory estimates."""
    descriptions = []
    for output in outputs.values():
        residency = output.get("engine", {}).get("residency", {})
        description = {}
        for stage in ("tempseg", "captioner"):
            record = residency.get(stage, {})
            if record.get("all_parameters_and_nonempty_buffers_on_cuda") is not True:
                raise ValueError("A condition lacks complete GPU residency evidence")
            description[stage] = {key: record.get(key) for key in
                                  ("parameter_bytes_by_device", "parameter_dtypes", "buffer_devices", "device_map")}
            if not description[stage]["parameter_bytes_by_device"]:
                raise ValueError("A condition lacks per-device parameter byte counts")
        descriptions.append(description)
    if not descriptions or any(value != descriptions[0] for value in descriptions[1:]):
        raise ValueError("Model placement changed within one benchmark condition")
    return digest(descriptions[0]), descriptions[0]


def _guard_allocation():
    reject_restarted_execution()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Native benchmarks require an authorized SLURM allocation")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned native environment requires Python 3.12")
    for setting in ("HF_HOME", "FLORA_MODEL_CACHE"):
        if not os.environ.get(setting):
            raise RuntimeError(setting + " must use verified project storage")
    # Inference must use already prepared assets. A fresh GPU benchmark must not
    # spend its allocation downloading models or silently changing revisions.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _hardware(torch):
    return dict(cuda_runtime=torch.version.cuda, devices=[dict(logical_device=i,
                name=torch.cuda.get_device_name(i), total_memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())],
                allocation_identity_sha256=digest({key: os.environ.get(key) for key in
                    ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")}))


def _worker(args):
    started = time.monotonic()
    condition_dir = args.output_dir / args.worker_condition
    report_path = condition_dir / ("resume-probe.json" if args.worker_resume else "probe-summary.json")
    report = dict(schema_version="bir_native_benchmark_condition_v1", status="started",
                  condition=args.worker_condition, cache_only=args.worker_resume, gpu_memory=[])
    atomic_json(report_path, report)
    torch = None
    try:
        _guard_allocation()
        report["gate_c"] = verify_gate_c(args.gate_c_marker)
        environment = {name: importlib.metadata.version(name) for name in EXPECTED}
        if any(environment[name].split("+")[0] != expected for name, expected in EXPECTED.items()):
            raise RuntimeError("Benchmark environment differs from the pinned native packages")
        import torch as framework
        torch = framework
        if not torch.cuda.is_available() or not 1 <= torch.cuda.device_count() <= 4 or torch.version.cuda != "12.8":
            raise RuntimeError("Benchmark requires one to four granted GPUs and the pinned CUDA 12.8 runtime")
        report.update(environment=environment, hardware=_hardware(torch))
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
        contract = build_execution_contract(model_lock=args.model_lock, condition=args.worker_condition,
                                             captioner_headroom_gib=args.captioner_headroom_gib)
        report["execution_contract"] = contract
        jobs = load_manifest(args.manifest, contract=contract)
        if not 1 <= len(jobs) <= 12:
            raise ValueError("A native benchmark must contain one to twelve episodes")
        plan = read_json(args.output_dir / "benchmark-plan.json")
        if (report["gate_c"] != plan["gate_c_evidence"] or runner_source_hashes() != plan["runner_sources_sha256"] or
                file_sha256(args.manifest) != plan["manifest_sha256"] or
                workload_signature(jobs) != plan["workload_signature"] or
                file_sha256(args.model_lock) != plan["model_lock_sha256"] or
                contract["engine_source_sha256"] != plan["engine_source_sha256"] or
                args.captioner_headroom_gib != plan["captioner_headroom_gib"]):
            raise ValueError("Benchmark workload, code, or placement settings changed before worker inference")
        state = condition_dir / "state"
        if not args.worker_resume and state.exists():
            raise ValueError("Fresh benchmark condition already has a state directory")
        if args.worker_resume and not (state / "summary.json").exists():
            raise ValueError("Resume requires a completed first condition invocation")
        invocation = argparse.Namespace(source=args.source_root, manifest=args.manifest, model_lock=args.model_lock,
                captioner_headroom_gib=args.captioner_headroom_gib, batch_size=1,
                condition=args.worker_condition, run_dir=state, stages="all")
        summary, code = execute_run(invocation, time.monotonic())
        report.update(run=summary, status="complete" if code == 0 else "incomplete")
        atomic_json(condition_dir / ("resume-summary.json" if args.worker_resume else "fresh-summary.json"), summary)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        code = 1
    finally:
        if torch is not None and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)
                    report["gpu_memory"].append(dict(logical_device=i,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(i)))
                except Exception as error:
                    report["gpu_memory"].append(dict(logical_device=i, measurement_error=str(error)))
                    report["status"], code = "incomplete", 1
        report["worker_workflow_seconds"] = time.monotonic() - started
        atomic_json(report_path, report)
    return code


def _subprocess(args, condition, *, resume=False):
    directory = args.output_dir / condition
    directory.mkdir(exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--source-root", str(args.source_root),
               "--manifest", str(args.manifest), "--output-dir", str(args.output_dir),
               "--gate-c-marker", str(args.gate_c_marker), "--model-lock", str(args.model_lock),
               "--captioner-headroom-gib", str(args.captioner_headroom_gib), "--worker-condition", condition]
    if resume:
        command.append("--worker-resume")
    log = directory / ("resume-runtime.log" if resume else "runtime.log")
    begin = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - begin
    if process.returncode != 0:
        raise RuntimeError(f"{condition} {'resume' if resume else 'fresh'} worker failed; inspect its private log")
    return elapsed, log


def run_benchmark(args):
    started = time.monotonic()
    _guard_allocation()
    gate = verify_gate_c(args.gate_c_marker)
    verify_source(args.source_root)
    contract = build_execution_contract(model_lock=args.model_lock, captioner_headroom_gib=args.captioner_headroom_gib)
    jobs = load_manifest(args.manifest, contract=contract)
    if not 1 <= len(jobs) <= 12:
        raise ValueError("A native benchmark must contain one to twelve episodes")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Benchmark output directory must be fresh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_hashes = runner_source_hashes()
    atomic_json(args.output_dir / "benchmark-plan.json", dict(gate_c_evidence=gate, runner_sources_sha256=runner_hashes,
                manifest_sha256=file_sha256(args.manifest),
                workload_signature=workload_signature(jobs), model_lock_sha256=file_sha256(args.model_lock),
                engine_source_sha256=contract["engine_source_sha256"], captioner_headroom_gib=args.captioner_headroom_gib))
    report = dict(schema_version="bir_native_benchmark_v1", status="started", gate_c=gate,
                  started_utc=datetime.now(timezone.utc).isoformat(), condition_order=list(CONDITIONS),
                  episode_ids=[job.id for job in jobs], source_commit=UPSTREAM_COMMIT,
                  conditions={}, comparisons={}, allocated_gpu_hours=None,
                  accounting_note="One shared allocation; coordinator must reconcile its full elapsed GPU time once, including setup, resume, failures, and idle time",
                  timing_note="Condition subprocess wall includes process startup; CLI time excludes worker framework/environment checks; stage attempt time excludes preparation",
                  order_limitation="Fixed A/B/C order can have OS page-cache and machine-state effects; no speedup claim is independent of this order")
    path = args.output_dir / "benchmark-summary.json"
    atomic_json(path, report)
    expected_hardware = expected_environment = expected_placement = None
    try:
        for condition in CONDITIONS:
            if runner_source_hashes() != runner_hashes:
                raise ValueError("Benchmark or comparison script changed during execution")
            directory = args.output_dir / condition
            subprocess_wall, log = _subprocess(args, condition)
            if runner_source_hashes() != runner_hashes:
                raise ValueError("Benchmark or comparison script changed during execution")
            probe = read_json(directory / "probe-summary.json")
            if probe.get("status") != "complete":
                raise ValueError("Worker did not finish with complete evidence")
            measured_contract = probe["execution_contract"]
            if measured_contract["engine_source_sha256"] != contract["engine_source_sha256"]:
                raise ValueError("Engine code changed during the benchmark")
            expected = build_execution_contract(model_lock=args.model_lock, condition=condition,
                                                  captioner_headroom_gib=args.captioner_headroom_gib)
            if measured_contract != expected:
                raise ValueError("Worker changed the model/configuration contract")
            run = verify_fresh_run(directory / "state", len(jobs))
            placement, placement_details = placement_signature(run["outputs"])
            if expected_hardware is None:
                expected_hardware, expected_environment, expected_placement = probe["hardware"], probe["environment"], placement
            elif (probe["hardware"] != expected_hardware or probe["environment"] != expected_environment or
                  placement != expected_placement):
                raise ValueError("Hardware, environment, or actual placement changed across conditions")
            fresh = directory / "fresh"
            (fresh / "annotations").mkdir(parents=True)
            atomic_json(fresh / "summary.json", run["summary"])
            for artifact in (directory / "state" / "annotations").glob("*.caption.json"):
                shutil.copyfile(artifact, fresh / "annotations" / artifact.name)
            metrics = performance(run)
            attempts = run["summary"]["attempts"]
            identities = {row["request_id"]: row["id"] for row in run["summary"]["records"]}
            metrics.update(subprocess_wall_seconds=subprocess_wall,
                           native_runtime_events=runtime_timings(log),
                           source_hours_per_subprocess_hour=metrics["source_hours"] / (subprocess_wall / 3600),
                           per_stage_attempt_work_seconds={stage: sum(row["elapsed_s"] for row in attempts if row["stage"] == stage)
                                                           for stage in ("segment", "caption")},
                           per_stage_episode_attempt_seconds={stage: [dict(episode_id=identities[row["request_id"]], elapsed_s=row["elapsed_s"])
                                                                     for row in attempts if row["stage"] == stage]
                                                              for stage in ("segment", "caption")})
            report["conditions"][condition] = dict(status="fresh_complete", metrics=metrics,
                    hardware=probe["hardware"], environment=probe["environment"], placement_sha256=placement,
                    placement=placement_details, gpu_memory=probe["gpu_memory"], runtime_trace_sha256=file_sha256(log))
            atomic_json(path, report)
            if condition != "reference":
                comparison = compare(args.output_dir / "reference" / "fresh", fresh, stage="all",
                    left_logs=[args.output_dir / "reference" / "runtime.log"], right_logs=[log])
                atomic_json(args.output_dir / f"reference-vs-{condition}.json", comparison)
                report["comparisons"][condition] = comparison
                if not comparison["compatible"] or not comparison["tensor_input_agreement_verified"]:
                    raise ValueError("Native inputs or outputs differ across benchmark conditions")
            resume_wall, resume_log = _subprocess(args, condition, resume=True)
            resume = read_json(directory / "resume-summary.json")
            verify_resume(resume, len(jobs), resume_log)
            report["conditions"][condition]["resume"] = dict(status="cache_only_verified", fresh_stage_attempts=0,
                subprocess_wall_seconds=resume_wall, cli_workflow_seconds=resume["total_cli_workflow_s"],
                resumed_stages=resume["resumed_stages"])
            atomic_json(path, report)
        report["status"] = "complete"
        return_code = 0
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return_code = 1
    report["total_benchmark_workflow_seconds"] = time.monotonic() - started
    atomic_json(path, report)
    return return_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-c-marker", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, default=Path("configs/native-models.json"))
    parser.add_argument("--captioner-headroom-gib", type=float, default=4.0)
    parser.add_argument("--worker-condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_resume and not args.worker_condition:
        parser.error("--worker-resume is an internal condition-worker option")
    for key in ("source_root", "manifest", "output_dir", "gate_c_marker", "model_lock"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    try:
        return _worker(args) if args.worker_condition else run_benchmark(args)
    except Exception as error:
        print(canonical_json(dict(status="failed", error_type=type(error).__name__, error=str(error))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
