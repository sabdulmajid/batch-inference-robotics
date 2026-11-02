"""Independent terminal FLORA scoring. Importing this module never loads a model.

The companion worker calls the pinned upstream program in a separate environment.
Native execution is opt-in and is not validated by this release's CPU tests.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile


DEFAULT_CONFIG = {
    "schema_version": "bir_rewarder_config_v1",
    "profile": "terminal_native_h100_v1",
    "native_validation": "awaiting_validation",
    "source_commit": "129a2ada74c526ee4692f3d1b111bea6ca19d5b7",
    "model_repo": "nvidia/Flora-Rewarder",
    "model_revision": "031299e46e452fc0cedb4e0975eb9fa95fc37783",
    "model_checksums_sha256": "b20a073e6d792365248aeb29f3a1b339a434ee1309ef26a5cd2df683941ebe56",
    "runtime": {"python": "3.12", "vllm": "0.24.0", "transformers": "5.8.1",
                "qwen-vl-utils": "0.0.14", "decord": "0.6.0"},
    "native": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.88,
               "max_num_seqs": 8, "max_model_len": 16384, "seed": 0,
               "with_rationale": False, "dtype": "bfloat16"},
    "input": {"mode": "terminal", "whole_episode": True, "sampling": "upstream_uniform_64_10fps",
              "camera_clock": "synchronized_start", "preprocessing": "upstream_unchanged"},
}
_CAMERA_ORDERS = {1: None, 2: ["left", "right"], 3: ["left", "center", "right"]}
_RESULT_SCHEMA = "bir_rewarder_result_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".rewarder-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _lock(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This reward request already has an active worker") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_config(path=None):
    """Only the pinned first profile is supported. No approximate configuration."""
    value = json.loads(Path(path).read_text()) if path else json.loads(_json(DEFAULT_CONFIG))
    if value != DEFAULT_CONFIG:
        raise ValueError("Unsupported Rewarder configuration; use the pinned terminal profile")
    return value


def prepare_request(episode_id, task, videos, camera_order, *, whole_episode,
                    source_sha256=None, clock_offsets_s=None, config=None):
    """Bind original videos to one task, in upstream camera order.

    ``whole_episode`` and zero clock offsets are explicit caller assertions.
    Equal stream lengths alone cannot establish synchronized capture. Native
    decoding also checks the upstream frame-count and frame-rate tolerances.
    Local paths are separate from the portable, hashed request identity.
    """
    if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", episode_id):
        raise ValueError("episode_id must use the upstream safe identifier format")
    if not isinstance(task, str) or not task.strip() or len(task) > 10000:
        raise ValueError("An explicit nonempty episode task is required")
    if whole_episode is not True:
        raise ValueError("Terminal scoring requires the complete original episode, not a video cut")
    if not isinstance(videos, (list, tuple)) or not 1 <= len(videos) <= 4:
        raise ValueError("Terminal scoring supports one to four original camera videos")
    if not isinstance(camera_order, (list, tuple)) or len(camera_order) != len(videos):
        raise ValueError("One explicit camera name is required for each video")
    cameras = list(camera_order)
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", x) for x in cameras):
        raise ValueError("Camera names must be nonempty safe identifiers")
    if len(set(cameras)) != len(cameras):
        raise ValueError("Camera names must be unique")
    expected = _CAMERA_ORDERS.get(len(cameras))
    if expected and cameras != expected:
        raise ValueError("Unsupported camera order for the native profile")
    if len(cameras) == 4 and cameras not in (["top", "front", "left", "right"], ["top", "bottom", "left", "right"]):
        raise ValueError("Four cameras must be top, front/bottom, left, right")
    offsets = list(clock_offsets_s) if clock_offsets_s is not None else None
    if offsets is None or len(offsets) != len(videos) or any(type(x) not in (int, float) or x != 0 for x in offsets):
        raise ValueError("Explicit zero clock_offsets_s are required for synchronized whole-episode cameras")
    pinned = load_config()
    if config is not None and config != pinned:
        raise ValueError("Unsupported Rewarder configuration")
    paths = [Path(x).expanduser().resolve(strict=True) for x in videos]
    if len(set(paths)) != len(paths) or any(not x.is_file() or x.stat().st_size == 0 for x in paths):
        raise ValueError("Camera sources must be distinct nonempty regular files")
    hashes = [_hash_file(x) for x in paths]
    if source_sha256 is not None and list(source_sha256) != hashes:
        raise ValueError("Video source hash does not match the declared original source")
    contract = dict(schema_version="bir_rewarder_request_v1", episode_id=episode_id, task=task,
                    whole_episode=True, camera_order=cameras, clock_offsets_s=offsets,
                    sources=[dict(camera=c, sha256=h, size_bytes=p.stat().st_size)
                             for c, h, p in zip(cameras, hashes, paths)], config=pinned)
    return dict(request_id=_digest(contract), contract=contract, local_sources=[str(x) for x in paths])


def _verify_request(request):
    if set(request) != {"request_id", "contract", "local_sources"} or request["request_id"] != _digest(request["contract"]):
        raise ValueError("Reward request identity is invalid")
    c = request["contract"]
    rebuilt = prepare_request(c["episode_id"], c["task"], request["local_sources"], c["camera_order"],
        whole_episode=c["whole_episode"], source_sha256=[x["sha256"] for x in c["sources"]],
        clock_offsets_s=c["clock_offsets_s"], config=c["config"])
    if rebuilt["request_id"] != request["request_id"]:
        raise ValueError("Reward request contract changed")


def parse_terminal(raw, *, episode_id, task):
    """Check the exact pinned upstream terminal JSON contract, including rounding."""
    fields = {"id", "task", "score", "expected_score", "progress", "probabilities", "rationale",
              "duration_s", "sampled_frames", "output"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Unexpected native terminal output fields")
    if raw["id"] != episode_id or raw["task"] != task:
        raise ValueError("Native output belongs to a different episode or task")
    if type(raw["score"]) is not int or raw["score"] not in range(1, 6):
        raise ValueError("Native terminal score must be an integer from one to five")
    if not isinstance(raw["output"], str) or not re.fullmatch(r"Score:\s*([1-5])\s*", raw["output"].strip()):
        raise ValueError("Invalid native terminal parse")
    if int(raw["output"].strip()[-1]) != raw["score"] or raw["rationale"] is not None:
        raise ValueError("Native score or rationale disagrees with the pinned answer format")
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    p = raw["probabilities"]
    if not isinstance(p, dict) or set(p) != set("12345") or any(not number(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError("Invalid score probabilities")
    # Upstream independently rounds five probabilities and the mean to 4 places.
    if abs(sum(p.values()) - 1) > 0.000251:
        raise ValueError("Score probabilities do not sum to one within native rounding")
    expected = raw["expected_score"]
    if not number(expected) or not 1 <= expected <= 5 or abs(expected - sum(int(k) * v for k, v in p.items())) > 0.000801:
        raise ValueError("Expected score is inconsistent with native probabilities")
    if not number(raw["progress"]) or not 0 <= raw["progress"] <= 1 or abs(raw["progress"] - (expected - 1) / 4) > 0.000064:
        raise ValueError("Progress is inconsistent with the expected score")
    if not number(raw["duration_s"]) or raw["duration_s"] <= 0:
        raise ValueError("Invalid scored duration")
    if type(raw["sampled_frames"]) is not int or not 2 <= raw["sampled_frames"] <= 64 or raw["sampled_frames"] % 2:
        raise ValueError("Invalid native terminal frame count")
    return json.loads(_json(raw))


def _verify_assets(source, model):
    source, model = Path(source).resolve(strict=True), Path(model).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    if git("rev-parse", "HEAD") != DEFAULT_CONFIG["source_commit"] or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Rewarder source is not the clean pinned upstream revision")
    checksums = model / "SHA256SUMS"
    if _hash_file(checksums) != DEFAULT_CONFIG["model_checksums_sha256"]:
        raise ValueError("Rewarder checkpoint checksum manifest differs from the pinned revision")
    checked = {}
    for line in checksums.read_text().splitlines():
        expected, name = line.split()
        if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid checkpoint checksum manifest")
        if _hash_file(model / name) != expected:
            raise ValueError("Rewarder model or processor checksum mismatch: " + name)
        checked[name] = expected
    return dict(source_commit=DEFAULT_CONFIG["source_commit"], model_revision=DEFAULT_CONFIG["model_revision"],
                verified_files=checked)


def native_command(python, flora_source, model_dir, jobs, output):
    """Build the real upstream CLI call, without executing or downloading anything."""
    args = [str(python), "-m", "rewarder.batch_utils.terminal_reward", "--jobs", str(Path(jobs).resolve()),
            "--output", str(Path(output).resolve()), "--model", str(Path(model_dir).resolve())]
    for key in ("tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs", "max_model_len", "seed"):
        args += ["--" + key.replace("_", "-"), str(DEFAULT_CONFIG["native"][key])]
    return args


def _environment():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "FLORA_REWARDER_REPO"):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def _native_executor(request, attempt_dir, *, python, flora_source, model_dir, timeout_s):
    # This function is reachable only through explicit native opt-in.
    if not all((python, flora_source, model_dir)):
        raise ValueError("Native scoring needs a separate Python executable, pinned source, and local model directory")
    check = """import importlib.metadata as m,json,sys,torch
names=['vllm','transformers','qwen-vl-utils','decord','av','numpy','Pillow','huggingface-hub','torch','triton']
print(json.dumps(dict(python='.'.join(map(str,sys.version_info[:3])),packages={n:m.version(n) for n in names},gpus=[dict(name=torch.cuda.get_device_name(i),memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])))"""
    runtime = json.loads(subprocess.check_output([str(python), "-c", check], env=_environment(), text=True, timeout=60))
    if not runtime["python"].startswith("3.12.") or any(runtime["packages"][k] != v for k, v in DEFAULT_CONFIG["runtime"].items() if k != "python"):
        raise ValueError("Rewarder runtime does not match the pinned separate environment")
    packages = runtime["packages"]
    if int(packages["numpy"].split(".")[0]) >= 3 or not 11 <= int(packages["Pillow"].split(".")[0]) < 13 or int(packages["av"].split(".")[0]) < 15:
        raise ValueError("Rewarder video runtime is outside the upstream dependency limits")
    if len(runtime["gpus"]) != 1 or "H100" not in runtime["gpus"][0]["name"] or runtime["gpus"][0]["memory_bytes"] < 79 * 1024**3:
        raise ValueError("Unsupported hardware profile: first native profile requires one H100 80 GB")
    assets = _verify_assets(flora_source, model_dir)
    _atomic(attempt_dir / "runtime.json", dict(runtime=runtime, assets=assets))
    jobs, output = attempt_dir / "jobs.jsonl", attempt_dir / "native.jsonl"
    c = request["contract"]
    _atomic(jobs, dict(id=c["episode_id"], task=c["task"], videos=request["local_sources"]))
    command = native_command(python, flora_source, model_dir, jobs, output)
    with (attempt_dir / "native.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=flora_source, env=_environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _atomic(attempt_dir / "process.private.json", dict(pid=process.pid, started_at=_now()))
            code = process.wait(timeout=timeout_s)
            if code:
                raise RuntimeError("Native Rewarder subprocess failed; inspect the private attempt log")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return dict(runtime=runtime, assets=assets)


def _save_result(path, result):
    result = dict(result)
    result["result_digest"] = _digest(result)
    _atomic(path, result)
    return result


def validate_result(result):
    """Check a portable in-memory artifact without requiring original local paths."""
    result = json.loads(_json(result))
    digest = result.pop("result_digest", None)
    if digest != _digest(result) or result.get("schema_version") != _RESULT_SCHEMA:
        raise ValueError("Reward result integrity check failed")
    c = result.get("request", {})
    if result.get("request_id") != _digest(c):
        raise ValueError("Reward result request identity mismatch")
    keys = {"schema_version", "episode_id", "task", "whole_episode", "camera_order", "clock_offsets_s", "sources", "config"}
    if set(c) != keys or c["schema_version"] != "bir_rewarder_request_v1" or c["config"] != DEFAULT_CONFIG or c["whole_episode"] is not True:
        raise ValueError("Unsupported reward request contract or configuration")
    if not isinstance(c["episode_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", c["episode_id"]):
        raise ValueError("Invalid reward episode identity")
    if not isinstance(c["task"], str) or not c["task"].strip() or len(c["task"]) > 10000:
        raise ValueError("Invalid reward task")
    sources = c["sources"]
    cameras = c["camera_order"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 4 or not isinstance(cameras, list) or len(cameras) != len(sources):
        raise ValueError("Invalid reward camera sources")
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", x) for x in cameras) or len(set(cameras)) != len(cameras):
        raise ValueError("Invalid reward camera names")
    if len(cameras) in (2, 3) and cameras != _CAMERA_ORDERS[len(cameras)]:
        raise ValueError("Unsupported reward camera order")
    if len(cameras) == 4 and cameras not in (["top", "front", "left", "right"], ["top", "bottom", "left", "right"]):
        raise ValueError("Unsupported reward camera order")
    if not isinstance(c["clock_offsets_s"], list) or len(c["clock_offsets_s"]) != len(sources) or any(type(x) not in (int, float) or x != 0 for x in c["clock_offsets_s"]):
        raise ValueError("Invalid reward camera clock alignment")
    for camera, source in zip(cameras, sources):
        if not isinstance(source, dict) or set(source) != {"camera", "sha256", "size_bytes"} or source["camera"] != camera or not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) or type(source["size_bytes"]) is not int or source["size_bytes"] <= 0:
            raise ValueError("Invalid reward source identity")
    if result.get("status") not in {"complete", "failed", "interrupted", "pending", "unknown", "unavailable"} or type(result.get("parse_valid")) is not bool:
        raise ValueError("Invalid reward result state")
    if result.get("execution", {}).get("kind") not in {"native", "test_fixture", "not_executed"}:
        raise ValueError("Unknown reward execution provenance")
    if result["status"] == "complete":
        if result["execution"]["kind"] == "not_executed":
            raise ValueError("Unexecuted reward cannot be complete")
        raw = parse_terminal(result["raw_record"], episode_id=result["request"]["episode_id"], task=result["request"]["task"])
        if result["parse_valid"] is not True or result["score"] != raw["score"] or result["expected_score"] != raw["expected_score"] or result["progress"] != raw["progress"]:
            raise ValueError("Reward result disagrees with the native record")
    elif result["parse_valid"] or any(result[k] is not None for k in ("score", "expected_score", "progress")):
        raise ValueError("Incomplete reward cannot contain a valid training value")
    result["result_digest"] = digest
    return result


def load_result(path):
    """Read a portable result and reject changed content or inconsistent scores."""
    return validate_result(json.loads(Path(path).read_text()))


def validate_binding(result, *, episode_id, task, cameras):
    """Check that outcome evidence belongs to this exact ordered episode input."""
    result = validate_result(result)
    c = result["request"]
    expected = [{"camera": x["camera"], "source_sha256": x["sha256"]} for x in c["sources"]]
    actual = [{"camera": x["camera"], "source_sha256": x["source_sha256"]} for x in cameras]
    if c["episode_id"] != episode_id or c["task"] != task or expected != actual:
        raise ValueError("Reward episode, task, camera order or source hashes do not match the rollout")
    return result


def run_terminal(request, output_dir, *, python=None, flora_source=None, model_dir=None,
                 allow_native=False, retry=False, timeout_s=1800, _test_executor=None):
    """Run one resumable scoring request. No annotation database is needed.

    Native execution requires ``allow_native=True`` and separate prepared assets.
    The private test hook always marks its artifacts as test fixtures. It cannot
    create training rewards. Failed or interrupted attempts need explicit retry.
    """
    if _test_executor is None and not allow_native:
        raise RuntimeError("Native Rewarder execution is disabled; GPU work remains paused pending explicit authorization and budget revision")
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 7200:
        raise ValueError("Reward timeout must be positive and at most two hours")
    if not re.fullmatch(r"[0-9a-f]{64}", request.get("request_id", "")):
        raise ValueError("Invalid reward request identity")
    directory = Path(output_dir).resolve() / request["request_id"]
    directory.mkdir(parents=True, exist_ok=True)
    state_path, result_path = directory / "state.json", directory / "result.json"
    with _lock(directory / ".lock"):
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(attempt=0, status="pending")
        # Recheck source bytes even when no generation is needed.
        try:
            _verify_request(request)
        except Exception:
            if state["status"] == "complete":
                raise ValueError("Completed reward source changed; prepare a new request")
            raise
        if state["status"] == "complete":
            prior = load_result(result_path)
            if prior["request_id"] != request["request_id"]:
                raise ValueError("Stored reward belongs to a different request")
            raw_path = directory / f"attempt-{state['attempt']:04d}" / "native.jsonl"
            if _hash_file(raw_path) != prior["raw_output_sha256"]:
                raise ValueError("Stored native output changed")
            return prior
        if state["status"] in {"failed", "interrupted"} and not retry:
            return load_result(result_path)
        if state["status"] == "running":
            if state.get("execution",{}).get("kind")=="native":
                raise RuntimeError("Interrupted native attempt requires allocation reconciliation before retry in a fresh output directory; a PID alone cannot prove cross-host ownership")
            old = directory / f"attempt-{state['attempt']:04d}"
            process_receipt = old / "process.private.json"
            if not process_receipt.exists() and state.get("execution",{}).get("kind")=="native":
                raise RuntimeError("Interrupted native attempt has no process receipt; reconcile its allocation before retry in a fresh output directory")
            if process_receipt.exists():
                pid = json.loads(process_receipt.read_text())["pid"]
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    # Never cancel a PID here: a live or reused PID is ambiguous.
                    raise RuntimeError("Previous native process may still be active; reconcile it before retry")
            lost = dict(schema_version=_RESULT_SCHEMA, request_id=request["request_id"], request=request["contract"],
                        attempt=state["attempt"], status="interrupted", parse_valid=False, score=None,
                        expected_score=None, progress=None, raw_record=None, raw_output_sha256=None,
                        execution=state["execution"], error={"type": "InterruptedWorker", "message": "Previous worker ended without a complete receipt; no output was promoted"}, finished_at=_now())
            lost = _save_result(old / "result.json", lost)
            _atomic(result_path, lost)
            state["status"] = "interrupted"
            _atomic(state_path, state)
            if not retry:
                return lost
        number = state["attempt"] + 1
        attempt_dir = directory / f"attempt-{number:04d}"
        attempt_dir.mkdir()
        execution = dict(kind="test_fixture" if _test_executor else "native", native_validation="awaiting_validation")
        state = dict(attempt=number, status="running", execution=execution, started_at=_now())
        _atomic(state_path, state)
        result = dict(schema_version=_RESULT_SCHEMA, request_id=request["request_id"], request=request["contract"],
                      attempt=number, status="failed", parse_valid=False, score=None, expected_score=None,
                      progress=None, raw_record=None, raw_output_sha256=None, execution=execution)
        interrupted = None
        try:
            if _test_executor:
                execution["receipt"] = _test_executor(request, attempt_dir)
            else:
                execution["receipt"] = _native_executor(request, attempt_dir, python=python,
                    flora_source=flora_source, model_dir=model_dir, timeout_s=timeout_s)
            raw_path = attempt_dir / "native.jsonl"
            result["raw_output_sha256"] = _hash_file(raw_path)
            lines = raw_path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError("Native terminal output must contain exactly one episode")
            def reject_constant(value):
                raise ValueError("Native output contains a nonfinite JSON number: " + value)
            def finite_float(value):
                number=float(value)
                if not math.isfinite(number):raise ValueError("Native output contains an overflowing JSON number")
                return number
            raw = json.loads(lines[0], parse_constant=reject_constant,parse_float=finite_float)
            # Keep the unmodified JSON record even if semantic validation fails.
            result["raw_record"] = raw
            parsed = parse_terminal(raw, episode_id=request["contract"]["episode_id"], task=request["contract"]["task"])
            _verify_request(request)  # Reject input mutation while inference was running.
            result.update(status="complete", parse_valid=True, score=parsed["score"],
                          expected_score=parsed["expected_score"], progress=parsed["progress"])
        except BaseException as error:
            if not isinstance(error, Exception):
                interrupted = error
                result["status"] = "interrupted"
            # Full exception/log paths stay private. A portable receipt uses a stable reason.
            result["error"] = dict(type=type(error).__name__, message="Scoring failed or was interrupted; no training reward is available")
            _atomic(attempt_dir / "failure.private.json", dict(type=type(error).__name__, message=str(error)))
        result["finished_at"] = _now()
        result = _save_result(attempt_dir / "result.json", result)
        _atomic(result_path, result)
        state["status"] = result["status"]
        _atomic(state_path, state)
        if interrupted:
            raise interrupted
        return result


def terminal_reward(result, *, mapping="expected_progress_v1"):
    """Return a distinct derived value; the trajectory exporter places it.

    This is a retrospective normalized score, not a calibrated success chance.
    Only an actual episode-ending transition may receive it. This function does
    not fabricate terminal flags, transition rewards, or missing evidence.
    """
    if mapping != "expected_progress_v1":
        raise ValueError("Unsupported terminal reward mapping version")
    result = validate_result(result)
    if result.get("status") != "complete" or result.get("parse_valid") is not True or result.get("execution", {}).get("kind") != "native":
        raise ValueError("A valid native score is required; unknown, failed and test-fixture results cannot become rewards")
    receipt = result["execution"].get("receipt", {})
    assets = receipt.get("assets", {})
    if receipt.get("fixture") or assets.get("source_commit") != DEFAULT_CONFIG["source_commit"] or assets.get("model_revision") != DEFAULT_CONFIG["model_revision"] or not assets.get("verified_files") or not receipt.get("runtime"):
        raise ValueError("Native reward lacks the verified asset and runtime receipt")
    raw = parse_terminal(result["raw_record"], episode_id=result["request"]["episode_id"], task=result["request"]["task"])
    return dict(schema_version="bir_terminal_reward_mapping_v1", mapping=mapping,
                value=(raw["expected_score"] - 1.0) / 4.0, placement="actual_episode_end_only",
                reward_result_digest=result["result_digest"], request_id=result["request_id"],
                interpretation="retrospective_normalized_expected_score_not_success_probability")
