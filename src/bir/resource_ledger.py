"""Private, transactional allocation accounting. This module never submits jobs.

Keep the database on the project's persistent private storage, not in Git.
One allocation record is charged once per restart; Slurm step rows are rejected.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Mapping


class LedgerError(ValueError):
    """A request cannot proceed without reconciliation or a corrected request."""


@dataclass(frozen=True)
class ResourceRequest:
    kind: str  # gpu or cpu
    phase: str  # framework_smoke, native_probe, required, optional, release
    minutes: float
    cpus: int
    ram_gib: float
    tmpdisk_mib: int
    gpus: int = 0  # A shard counts as one physical GPU.
    nodes: int = 1
    justification: str = ""
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class AllocationAttempt:
    job_id: str
    restart_index: int
    elapsed_seconds: float
    gpu_count: int
    cpu_count: int
    state: str
    exit_code: str


_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
_ACTIVE = {"reserved", "submission_unknown", "submitted", "restart_blocked"}


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise LedgerError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise LedgerError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def reject_restarted_execution(environ: Mapping[str, str] | None = None) -> None:
    """Call before model work. A requeue never grants another free budget."""
    raw = (os.environ if environ is None else environ).get("SLURM_RESTART_COUNT", "0")
    if not raw.isdigit() or int(raw) != 0:
        raise LedgerError("Restarted allocation blocked: reconcile all prior attempt time before a new job")


def check_storage(policy: Mapping[str, Any], *, total_bytes: int, available_bytes: int,
                  existing_project_bytes: int, planned_new_bytes: int) -> dict[str, Any]:
    """Check the current project policy against measured filesystem capacity.

    Available bytes must cover 120% (or the configured ratio) of the full
    remaining planned new footprint, including pending writes. This is separate
    from the new-project persistent-storage ceiling. The user approved this
    project-sized headroom check instead of a whole-filesystem free percentage.
    Use statvfs f_bavail (space available to the account), not privileged f_bfree.
    This does not establish access, mount health, or quota; verify those first.
    """
    for name, value in (("total_bytes", total_bytes), ("available_bytes", available_bytes),
                        ("existing_project_bytes", existing_project_bytes), ("planned_new_bytes", planned_new_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LedgerError(f"{name} must be a nonnegative measured integer")
    if total_bytes <= 0 or available_bytes > total_bytes:
        raise LedgerError("Invalid measured filesystem capacity")
    projected_project = existing_project_bytes + planned_new_bytes
    projected_available = available_bytes - planned_new_bytes
    fraction = projected_available / total_bytes
    cap = policy["storage"]["max_new_persistent_project_gib"] * 1024 ** 3
    scope = policy["storage"].get("headroom_scope")
    if scope != "remaining_planned_new_project_bytes" or "minimum_free_headroom_fraction" in policy["storage"]:
        raise LedgerError("Storage headroom policy must specify the approved project-sized scope")
    ratio = policy["storage"]["minimum_available_to_planned_new_bytes_ratio"]
    if _number(ratio, "storage headroom ratio") < 1:
        raise LedgerError("Storage headroom ratio must be at least one")
    required_available = math.ceil(planned_new_bytes * ratio)
    result = {"allowed": projected_project <= cap and available_bytes >= required_available,
              "projected_project_bytes": projected_project,
              "projected_available_bytes": projected_available,
              "projected_free_fraction": fraction,
              "project_cap_bytes": cap, "minimum_available_bytes": required_available,
              "minimum_available_to_planned_new_bytes_ratio": ratio,
              "headroom_scope": scope}
    result["reasons"] = []
    if projected_project > cap:
        result["reasons"].append("Project persistent-storage ceiling exceeded")
    if available_bytes < required_available:
        result["reasons"].append("Available storage does not cover planned new bytes plus project headroom")
    return result


class ResourceLedger:
    """Serialize reservations and retain immutable request and audit records.

    ``policy`` is the parsed repository resource_policy.yaml mapping. A policy
    change cannot silently reopen an existing ledger. ``coordinator_id`` is a
    task-owned persistent identity, not the shared Unix username.
    """

    def __init__(self, path: str | Path, policy: Mapping[str, Any], coordinator_id: str):
        if not coordinator_id.strip():
            raise LedgerError("A distinct coordinator identity is required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = json.loads(json.dumps(policy))
        self.coordinator_id = coordinator_id
        canonical = json.dumps(self.policy, sort_keys=True, separators=(",", ":"))
        self.policy_hash = hashlib.sha256(canonical.encode()).hexdigest()
        with self._transaction(initial=True) as db:
            schema = """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
                  job_id TEXT UNIQUE, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                  run_id TEXT NOT NULL, restart_index INTEGER NOT NULL,
                  record TEXT NOT NULL, PRIMARY KEY(run_id, restart_index));
                CREATE TABLE IF NOT EXISTS events (
                  sequence INTEGER PRIMARY KEY, timestamp REAL NOT NULL,
                  action TEXT NOT NULL, run_id TEXT, detail TEXT NOT NULL);
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.execute("INSERT OR IGNORE INTO meta VALUES ('coordinator', ?)", (coordinator_id,))
            db.execute("INSERT OR IGNORE INTO meta VALUES ('policy_hash', ?)", (self.policy_hash,))
            db.execute("INSERT OR IGNORE INTO meta VALUES ('release_reserve', ?)", (json.dumps({"gpu_hours": 0.0, "cpu_core_hours": 0.0, "slots": int(self.policy["gpu"]["minimum_release_submission_slots"]), "pilot_evidence": None}),))
            self._check_owner(db)

    @contextmanager
    def _transaction(self, *, initial: bool = False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            # Durable rollback-journal transactions also work without WAL support.
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            if not initial:
                self._check_owner(db)
            yield db
            if db.in_transaction:
                db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _check_owner(self, db):
        meta = dict(db.execute("SELECT key, value FROM meta"))
        if meta.get("coordinator") != self.coordinator_id:
            raise LedgerError("This ledger belongs to another coordinator")
        if meta.get("policy_hash") != self.policy_hash:
            raise LedgerError("Resource policy changed; reconcile the existing ledger explicitly")

    @staticmethod
    def _event(db, action, run_id, detail):
        db.execute("INSERT INTO events(timestamp,action,run_id,detail) VALUES(?,?,?,?)", (time.time(), action, run_id, json.dumps(detail, sort_keys=True)))

    def _validate(self, r: ResourceRequest):
        p = self.policy
        if r.kind not in {"gpu", "cpu"} or r.phase not in {"framework_smoke", "native_probe", "required", "optional", "release"}:
            raise LedgerError("Unknown resource kind or execution phase")
        for label in ("gpus", "cpus", "nodes", "tmpdisk_mib"):
            value = getattr(r, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise LedgerError(f"{label} must be an integer")
        _number(r.minutes, "minutes", positive=True)
        _number(r.ram_gib, "ram_gib", positive=True)
        if r.cpus < 1 or r.cpus > p["cpu"]["max_cpus_per_task"] or r.ram_gib > p["cpu"]["max_ram_gib"]:
            raise LedgerError("CPU or host RAM limit exceeded")
        if r.tmpdisk_mib <= 0 or r.tmpdisk_mib > p["storage"]["max_tmpdisk_mib"]:
            raise LedgerError("Temporary disk limit exceeded")
        if r.nodes != 1:
            raise LedgerError("Only same-node allocations are supported")
        if r.kind == "gpu":
            if not 1 <= r.gpus <= p["gpu"]["max_gpus_per_job"]:
                raise LedgerError("GPU count limit exceeded")
            limit = p["gpu"]["max_job_minutes"]
            if r.phase == "framework_smoke":
                limit = min(limit, p["gpu"]["framework_smoke_minutes"])
            elif r.phase == "native_probe":
                limit = min(limit, p["gpu"]["initial_native_probe_minutes"])
            if r.gpus > 1 and not r.justification.strip():
                raise LedgerError("Multi-GPU placement requires a recorded justification")
        else:
            if r.gpus != 0:
                raise LedgerError("CPU-only allocations cannot request GPUs")
            limit = p["cpu"]["max_cpu_only_job_minutes"]
        if r.minutes > limit:
            raise LedgerError("Job time limit exceeded")
        if r.provenance is None or not r.provenance:
            raise LedgerError("Record code, inputs, expected outputs, environment, and configuration provenance")

    def _snapshot(self, db):
        runs = [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY created, run_id")]
        used_gpu = used_cpu = reserved_gpu = reserved_cpu = 0.0
        gpu_slots = 0
        active = {"gpu": 0, "cpu": 0}
        for row in runs:
            r = json.loads(row["request"])
            row["request"] = r
            attempts = [json.loads(a[0]) for a in db.execute("SELECT record FROM attempts WHERE run_id=? ORDER BY restart_index", (row["run_id"],))]
            row["attempts"] = attempts
            ug = sum(a["gpu_count"] * a["elapsed_seconds"] / 3600 for a in attempts)
            uc = sum(a["cpu_count"] * a["elapsed_seconds"] / 3600 for a in attempts) if r["kind"] == "cpu" else 0.0
            used_gpu += ug
            used_cpu += uc
            if r["kind"] == "gpu" and row["status"] != "abandoned":
                gpu_slots += 1
            if row["status"] in _ACTIVE:
                active[r["kind"]] += 1
                # Recorded partial use plus outstanding reservation must not be
                # charged twice. Requeues remain blocked until reconciliation.
                reserved_gpu += max(0, r["gpus"] * r["minutes"] / 60 - ug)
                reserved_cpu += max(0, r["cpus"] * r["minutes"] / 60 - uc) if r["kind"] == "cpu" else 0.0
        reserve = json.loads(db.execute("SELECT value FROM meta WHERE key='release_reserve'").fetchone()[0])
        return {"used_gpu_hours": used_gpu, "used_cpu_core_hours": used_cpu,
                "reserved_gpu_hours": reserved_gpu, "reserved_cpu_core_hours": reserved_cpu,
                "gpu_submissions_and_reservations": gpu_slots, "active": active,
                "release_reserve": reserve, "runs": runs,
                "remaining_gpu_hours": self.policy["gpu"]["total_allocated_gpu_hours"] - used_gpu - reserved_gpu,
                "remaining_cpu_core_hours": self.policy["cpu"]["max_cpu_only_core_hours"] - used_cpu - reserved_cpu}

    def snapshot(self) -> dict[str, Any]:
        with self._transaction() as db:
            return self._snapshot(db)

    def set_release_reserve(self, gpu_hours: float, cpu_core_hours: float, slots: int, pilot_evidence: str):
        _number(gpu_hours, "gpu_hours")
        _number(cpu_core_hours, "cpu_core_hours")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < self.policy["gpu"]["minimum_release_submission_slots"]:
            raise LedgerError("Release submission reserve is too small")
        if not pilot_evidence.strip():
            raise LedgerError("Release reserve needs measured pilot evidence")
        reserve = dict(gpu_hours=gpu_hours, cpu_core_hours=cpu_core_hours, slots=slots, pilot_evidence=pilot_evidence)
        with self._transaction() as db:
            s = self._snapshot(db)
            if gpu_hours > s["remaining_gpu_hours"] or cpu_core_hours > s["remaining_cpu_core_hours"] or slots + s["gpu_submissions_and_reservations"] > self.policy["gpu"]["max_gpu_submissions"]:
                raise LedgerError("Release reserve does not fit remaining budget")
            db.execute("UPDATE meta SET value=? WHERE key='release_reserve'", (json.dumps(reserve),))
            self._event(db, "release_reserve", None, reserve)

    def reserve(self, run_id: str, request: ResourceRequest):
        self._validate(request)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", run_id):
            raise LedgerError("Run ID must be a stable, nonempty safe identifier")
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                raise LedgerError("Run ID already exists; inspect it before retrying")
            s = self._snapshot(db)
            if any(row["status"] in {"submission_unknown", "restart_blocked"} for row in s["runs"]):
                raise LedgerError("Ambiguous submission or restart requires scheduler reconciliation")
            cap = self.policy["coordination"][f"max_pending_or_running_{'gpu' if request.kind == 'gpu' else 'cpu_only'}_jobs"]
            if s["active"][request.kind] >= cap:
                raise LedgerError("Project allocation concurrency limit reached")
            release = s["release_reserve"]
            if request.phase == "optional" and (not release["pilot_evidence"] or release["gpu_hours"] <= 0):
                raise LedgerError("Optional work requires a measured release reserve")
            protect = request.phase != "release"
            rg = release["gpu_hours"] if protect else 0
            rc = release["cpu_core_hours"] if protect else 0
            rs = release["slots"] if protect else 0
            if request.gpus * request.minutes / 60 + rg > s["remaining_gpu_hours"] + 1e-12:
                raise LedgerError("GPU-hour budget exceeded")
            if request.kind == "cpu" and request.cpus * request.minutes / 60 + rc > s["remaining_cpu_core_hours"] + 1e-12:
                raise LedgerError("CPU core-hour budget exceeded")
            if request.kind == "gpu" and s["gpu_submissions_and_reservations"] + 1 + rs > self.policy["gpu"]["max_gpu_submissions"]:
                raise LedgerError("GPU submission budget exceeded")
            db.execute("INSERT INTO runs VALUES(?,?,?,NULL,?)", (run_id, json.dumps(asdict(request), sort_keys=True), "reserved", time.time()))
            self._event(db, "reserve", run_id, asdict(request))

    @staticmethod
    def _run(db, run_id):
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise LedgerError("Unknown project run")
        return row

    def begin_submission(self, run_id: str):
        """Call immediately before sbatch; loss of its reply blocks another try."""
        with self._transaction() as db:
            row = self._run(db, run_id)
            if row["status"] != "reserved":
                raise LedgerError("Submission already attempted; reconcile scheduler state")
            db.execute("UPDATE runs SET status='submission_unknown' WHERE run_id=?", (run_id,))
            self._event(db, "begin_submission", run_id, {})

    def attach_job(self, run_id: str, job_id: str):
        if not re.fullmatch(r"[0-9]+", str(job_id)):
            raise LedgerError("Use an exact scalar allocation ID, not a job step or array")
        with self._transaction() as db:
            row = self._run(db, run_id)
            if row["status"] != "submission_unknown" or row["job_id"] is not None:
                raise LedgerError("Job attachment requires an unresolved submission")
            try:
                db.execute("UPDATE runs SET status='submitted', job_id=? WHERE run_id=?", (str(job_id), run_id))
            except sqlite3.IntegrityError as exc:
                raise LedgerError("Job ID already belongs to a recorded run") from exc
            self._event(db, "attach_job", run_id, {"job_id": str(job_id)})

    def abandon_unsubmitted(self, run_id: str, evidence: str):
        """Release only a reservation never sent to sbatch, or proven rejected."""
        if not evidence.strip():
            raise LedgerError("Record evidence that no allocation exists")
        with self._transaction() as db:
            row = self._run(db, run_id)
            if row["status"] not in {"reserved", "submission_unknown"} or row["job_id"] is not None:
                raise LedgerError("An attached job must be reconciled, not abandoned")
            db.execute("UPDATE runs SET status='abandoned' WHERE run_id=?", (run_id,))
            self._event(db, "abandon_unsubmitted", run_id, {"evidence": evidence})

    def assert_owned_job(self, job_id: str) -> str:
        """Require this check before cancellation. Shared username is insufficient."""
        with self._transaction() as db:
            row = db.execute("SELECT run_id FROM runs WHERE job_id=?", (str(job_id),)).fetchone()
            if row is None:
                raise LedgerError("Job ID is not recorded as this project's allocation")
            return row[0]

    def mark_restart(self, run_id: str, restart_count: int):
        if not isinstance(restart_count, int) or restart_count < 1:
            raise LedgerError("Expected a positive restart count")
        with self._transaction() as db:
            row = self._run(db, run_id)
            if row["status"] not in {"submitted", "restart_blocked"}:
                raise LedgerError("Only an active submitted job can restart")
            db.execute("UPDATE runs SET status='restart_blocked' WHERE run_id=?", (run_id,))
            self._event(db, "restart_blocked", run_id, {"restart_count": restart_count})

    def reconcile(self, run_id: str, attempts: list[AllocationAttempt]):
        """Record all final allocation attempts, including failed and idle time.

        Supply one allocation-level row per restart, never .batch/.extern rows.
        Actual overspend is recorded truthfully and blocks future reservations.
        Repeating an identical reconciliation is safe; changed records fail.
        """
        if not attempts:
            raise LedgerError("Final accounting must contain at least one allocation attempt")
        indices = sorted(a.restart_index for a in attempts)
        if any(isinstance(a.restart_index, bool) or not isinstance(a.restart_index, int) for a in attempts) or indices != list(range(len(attempts))):
            raise LedgerError("Account for every restart exactly once, starting at zero")
        for a in attempts:
            _number(a.elapsed_seconds, "elapsed_seconds")
            for label in ("gpu_count", "cpu_count"):
                value = getattr(a, label)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise LedgerError(f"{label} must be a nonnegative integer")
            if a.state not in _TERMINAL or not re.fullmatch(r"\d+:\d+", a.exit_code):
                raise LedgerError("Final scheduler state and exit code are required")
        with self._transaction() as db:
            row = self._run(db, run_id)
            if row["status"] not in {"submitted", "restart_blocked", "reconciled"}:
                raise LedgerError("Only an attached allocation can be reconciled")
            if any(a.job_id != row["job_id"] for a in attempts):
                raise LedgerError("Accounting does not match the exact owned allocation ID")
            request = json.loads(row["request"])
            for a in attempts:
                if a.elapsed_seconds > 0 and (a.cpu_count < 1 or (request["kind"] == "gpu" and a.gpu_count < 1)):
                    raise LedgerError("Allocated resources are missing from a nonzero attempt")
                if request["kind"] == "cpu" and a.gpu_count:
                    raise LedgerError("CPU request unexpectedly has GPUs; investigate its scheduler identity")
            restart_events = db.execute("SELECT detail FROM events WHERE run_id=? AND action='restart_blocked'", (run_id,)).fetchall()
            if restart_events and max(json.loads(e[0])["restart_count"] for e in restart_events) >= len(attempts):
                raise LedgerError("Known restarted allocation is missing attempt accounting")
            old = {a[0]: a[1] for a in db.execute("SELECT restart_index,record FROM attempts WHERE run_id=?", (run_id,))}
            records = {a.restart_index: json.dumps(asdict(a), sort_keys=True) for a in attempts}
            if old and old != records:
                raise LedgerError("Final accounting is immutable; preserve correction evidence separately")
            if not old:
                db.executemany("INSERT INTO attempts VALUES(?,?,?)", [(run_id, i, record) for i, record in records.items()])
                db.execute("UPDATE runs SET status='reconciled' WHERE run_id=?", (run_id,))
                self._event(db, "reconcile", run_id, {"attempt_count": len(attempts)})
