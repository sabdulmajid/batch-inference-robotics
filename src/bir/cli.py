"""Command line entry points for validated native inputs and resumable runs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time

from . import __version__
from .engine import Engine
from .manifest import canonical_json, file_sha256, load_manifest
from .native import MODEL_REVISIONS, UPSTREAM_COMMIT, NativeBackend, verify_source
from .resource_ledger import ResourceLedger, check_storage, reject_restarted_execution
from .store import RunStore


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bir-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _policy(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("The resource policy must be a mapping")
    return value


def _model_lock(path):
    value = json.loads(Path(path).read_text())
    if value.get("source", {}).get("revision") != UPSTREAM_COMMIT:
        raise ValueError("The model lock does not use the verified FLORA source revision")
    for kind, revision in MODEL_REVISIONS.items():
        if value.get(kind, {}).get("revision") != revision:
            raise ValueError("The model lock does not use the verified native model revisions")
    if value.get("microbatch_size") != 1:
        raise ValueError("Only native batch size one is currently supported")
    return value


def build_execution_contract(*, model_lock=Path("configs/native-models.json"),
                             captioner_headroom_gib=4.0, condition="optimized", batch_size=1):
    # Include adapter changes even in an uncommitted checkout. Paths and device
    # locations are not part of output identity. Placement headroom is explicit.
    code = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return dict(engine_version=__version__, engine_source_sha256=code,
                native_model_lock=_model_lock(model_lock), condition=condition,
                captioner_headroom_gib=captioner_headroom_gib, batch_size=batch_size)


def _contract(args, condition):
    return build_execution_contract(model_lock=args.model_lock,
                                    captioner_headroom_gib=args.captioner_headroom_gib,
                                    condition=condition, batch_size=args.batch_size)


def _native_options(parser):
    parser.add_argument("--source", type=Path, required=True, help="Pinned NVIDIA FLORA checkout")
    parser.add_argument("--manifest", type=Path, required=True, help="Native video JSONL manifest")
    parser.add_argument("--model-lock", type=Path, default=Path("configs/native-models.json"))
    parser.add_argument("--captioner-headroom-gib", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1)


def build_parser():
    parser = argparse.ArgumentParser(prog="bir", description="Offline native-input FLORA inference")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source-check", help="Check the pinned, unchanged source and model revisions")
    source.add_argument("--source", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Validate native videos on CPU without loading models")
    _native_options(inspect)
    inspect.add_argument("--output", type=Path)
    run = commands.add_parser("run", help="Run or resume native inference")
    _native_options(run)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--condition", choices=("reference", "persistent", "optimized"), default="optimized")
    run.add_argument("--stages", choices=("all", "segment", "caption"), default="all")
    storage = commands.add_parser("storage-check", help="Check measured project storage against the resource policy")
    storage.add_argument("--policy", type=Path, default=Path("configs/resource_policy.yaml"))
    storage.add_argument("--project-path", type=Path, required=True)
    storage.add_argument("--planned-gib", type=float, required=True,
                         help="Full remaining planned new footprint, including pending writes")
    ledger = commands.add_parser("ledger-status", help="Read the private resource ledger")
    ledger.add_argument("--policy", type=Path, default=Path("configs/resource_policy.yaml"))
    ledger.add_argument("--ledger", type=Path, required=True)
    ledger.add_argument("--coordinator", required=True)
    return parser


@contextmanager
def _run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".cli.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another CLI worker owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _inspect(args, started):
    contract = _contract(args, "input_inspection")
    jobs = load_manifest(args.manifest, contract=contract)
    backend = NativeBackend(args.source, captioner_headroom_gib=args.captioner_headroom_gib)
    try:
        rows = []
        for job in jobs:
            prepared = backend.prepare(job.as_native_dict())
            if prepared.metadata["input_sha256"] != dict(job.source_sha256):
                raise ValueError("Input changed during native validation")
            rows.append(dict(id=job.id, request_id=job.request_id, **prepared.metadata))
    finally:
        backend.close()
    result = dict(schema_version="bir_inspection_v1", status="inputs_validated", new_model_calls=0,
                  episodes=rows, total_cli_workflow_s=time.monotonic() - started)
    if args.output:
        atomic_json(args.output, result)
    return result, 0


def execute_run(args, started):
    reject_restarted_execution()
    contract = _contract(args, args.condition)
    jobs = load_manifest(args.manifest, contract=contract)
    directory = args.run_dir.expanduser().resolve()
    fingerprint = hashlib.sha256(canonical_json(contract).encode()).hexdigest()
    # The stage selector is intentionally excluded: segmentation and captioning
    # may run in separate allocations against the same committed checkpoints.
    metadata = dict(schema_version="bir_run_contract_v1", contract=contract, contract_sha256=fingerprint)
    with _run_lock(directory):
        metadata_path = directory / "run-metadata.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != metadata:
                raise ValueError("Run directory has a different code/configuration/condition; use a fresh directory")
        elif (directory / "run.sqlite").exists() or (directory / "summary.json").exists():
            raise ValueError("Run directory has results without matching metadata; use a fresh directory")
        else:
            atomic_json(metadata_path, metadata)
        store = RunStore(directory / "run.sqlite")
        try:
            cache_bytes = 256 * 1024**2 if args.condition == "optimized" else 0
            factory = lambda: NativeBackend(args.source, captioner_headroom_gib=args.captioner_headroom_gib,
                                             frame_cache_bytes=cache_bytes)
            engine = Engine(factory, store, mode="reference" if args.condition == "reference" else "staged",
                            prefetch=1 if args.condition == "optimized" else 0,
                            prepared_cache_size=8 if args.condition == "optimized" else 0)
            result = engine.run(jobs, stages=args.stages, output_dir=directory / "annotations")
            result.update(condition=args.condition, requested_stages=args.stages,
                          contract_sha256=fingerprint, total_cli_workflow_s=time.monotonic() - started,
                          timing_boundary="CLI timer includes manifest hashing, setup, preparation, inference, and export; engine timer excludes CLI setup")
            atomic_json(directory / "summary.json", result)
            return result, 0 if result["complete"] else 1
        except Exception as exc:
            failure = dict(schema_version="bir_run_v1", complete=False, execution_complete=False,
                           condition=args.condition, requested_stages=args.stages, error=str(exc),
                           error_type=type(exc).__name__, contract_sha256=fingerprint,
                           total_cli_workflow_s=time.monotonic() - started)
            atomic_json(directory / "summary.json", failure)
            raise
        finally:
            store.close()


def project_bytes(root):
    """Count project file bytes once per inode; do not follow external symlinks."""
    seen, total = set(), 0
    def walk_error(error):
        raise error
    for directory, _, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in names:
            info = (Path(directory) / name).lstat()
            identity = (info.st_dev, info.st_ino)
            if identity not in seen and (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                seen.add(identity)
                total += info.st_size
    return total


def main(argv=None):
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"inspect", "run"}:
            if args.batch_size != 1:
                raise ValueError("Batch sizes above one require GPU compatibility and memory measurements")
            if not math.isfinite(args.captioner_headroom_gib) or args.captioner_headroom_gib <= 0:
                raise ValueError("Captioner headroom must be finite and positive")
        if args.command == "source-check":
            result, code = verify_source(args.source), 0
        elif args.command == "inspect":
            result, code = _inspect(args, started)
        elif args.command == "run":
            result, code = execute_run(args, started)
        elif args.command == "storage-check":
            if not math.isfinite(args.planned_gib) or args.planned_gib < 0:
                raise ValueError("Planned new GiB must be finite and nonnegative")
            path = args.project_path.expanduser().resolve(strict=True)
            if not path.is_dir():
                raise ValueError("Project storage must be an existing directory")
            capacity = os.statvfs(path)
            result = check_storage(_policy(args.policy), total_bytes=capacity.f_blocks * capacity.f_frsize,
                                   available_bytes=capacity.f_bavail * capacity.f_frsize,
                                   existing_project_bytes=project_bytes(path),
                                   planned_new_bytes=math.ceil(args.planned_gib * 1024**3))
            result["checks_not_established"] = ["write permission", "account quota", "mount health"]
            code = 0 if result["allowed"] else 1
        elif args.command == "ledger-status":
            if not args.ledger.is_file():
                raise ValueError("The private ledger does not exist")
            result = ResourceLedger(args.ledger, _policy(args.policy), args.coordinator).snapshot()
            code = 0
        else:
            raise ValueError("Unknown command")
        print(canonical_json(result))
        return code
    except Exception as exc:
        print(canonical_json(dict(status="failed", error_type=type(exc).__name__, error=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
