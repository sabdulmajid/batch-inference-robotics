"""Atomic stage results and durable attempt history for one local worker."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from .manifest import canonical_json


class StoreBusy(RuntimeError):
    pass


class RetryExhausted(RuntimeError):
    pass


class RunStore:
    """SQLite keeps results and state in one atomic transaction.

    A process lock prevents duplicate execution. On a later exclusive open,
    unfinished attempts become interrupted failures. They consume retry capacity.
    Use a local or shared filesystem that supports POSIX locks and SQLite.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, identity_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                request_id TEXT NOT NULL REFERENCES requests(request_id),
                stage TEXT NOT NULL CHECK(stage IN ('segment','caption')),
                number INTEGER NOT NULL, state TEXT NOT NULL,
                started_at REAL NOT NULL, ended_at REAL, elapsed_s REAL,
                error_kind TEXT, error TEXT, payload_json TEXT, payload_sha256 TEXT,
                PRIMARY KEY(request_id,stage,number)
            );
        """)
        self._lock_handle = None

    @contextmanager
    def exclusive(self):
        if self._lock_handle is not None:
            raise StoreBusy("The run store is already open by this worker")
        handle = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StoreBusy("Another worker owns this run store") from exc
            self._lock_handle = handle
            with self._db:
                self._db.execute(
                    "UPDATE attempts SET state='failed', ended_at=?, error_kind='interrupted', "
                    "error='Previous worker ended before atomic completion' WHERE state='running'", (time.time(),))
            yield self
        finally:
            if self._lock_handle is handle:
                self._lock_handle = None
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _require_owner(self):
        if self._lock_handle is None:
            raise StoreBusy("Mutations require the exclusive run-store lock")

    def register(self, job):
        self._require_owner()
        encoded = canonical_json(job.identity())
        if hashlib.sha256(encoded.encode()).hexdigest() != job.request_id:
            raise ValueError("Request identity changed after manifest validation")
        previous = self._db.execute("SELECT identity_json FROM requests WHERE request_id=?", (job.request_id,)).fetchone()
        if previous is not None and previous[0] != encoded:
            raise ValueError("Request identity collision")
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO requests VALUES (?,?,?)", (job.request_id, job.id, encoded))

    def latest(self, request_id, stage):
        row = self._db.execute("SELECT * FROM attempts WHERE request_id=? AND stage=? ORDER BY number DESC LIMIT 1",
                               (request_id, stage)).fetchone()
        return dict(row) if row is not None else None

    def result(self, request_id, stage):
        row = self.latest(request_id, stage)
        if row is None or row["state"] != "complete":
            return None
        payload = row["payload_json"]
        if hashlib.sha256(payload.encode()).hexdigest() != row["payload_sha256"]:
            raise ValueError("Stored result checksum mismatch")
        return json.loads(payload)

    def begin(self, request_id, stage, *, max_transient_retries=1):
        self._require_owner()
        if stage not in {"segment", "caption"}:
            raise ValueError("Unknown stage")
        if max_transient_retries not in (0, 1):
            raise ValueError("At most one transient retry is permitted")
        previous = self.latest(request_id, stage)
        if previous:
            if previous["state"] == "complete":
                raise ValueError("A complete stage cannot execute again in this store")
            if previous["state"] == "running":
                raise StoreBusy("The stage already has a running attempt")
            if previous["error_kind"] not in {"transient", "interrupted"}:
                raise RetryExhausted("Deterministic failure requires a changed execution contract")
            if previous["number"] >= 1 + max_transient_retries:
                raise RetryExhausted("Transient retry limit reached")
        if stage == "caption" and self.result(request_id, "segment") is None:
            raise ValueError("Captioning requires a complete segmentation result")
        number = previous["number"] + 1 if previous else 1
        with self._db:
            self._db.execute("INSERT INTO attempts(request_id,stage,number,state,started_at) VALUES (?,?,?,'running',?)",
                             (request_id, stage, number, time.time()))
        return number

    def complete(self, request_id, stage, number, payload, elapsed_s):
        self._require_owner()
        encoded = canonical_json(payload)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self._db:
            cursor = self._db.execute(
                "UPDATE attempts SET state='complete',ended_at=?,elapsed_s=?,payload_json=?,payload_sha256=? "
                "WHERE request_id=? AND stage=? AND number=? AND state='running'",
                (time.time(), elapsed_s, encoded, digest, request_id, stage, number))
            if cursor.rowcount != 1:
                raise ValueError("Only the active attempt can complete")

    def fail(self, request_id, stage, number, error, *, kind, elapsed_s):
        self._require_owner()
        if kind not in {"transient", "deterministic"}:
            raise ValueError("Unknown failure kind")
        with self._db:
            cursor = self._db.execute(
                "UPDATE attempts SET state='failed',ended_at=?,elapsed_s=?,error_kind=?,error=? "
                "WHERE request_id=? AND stage=? AND number=? AND state='running'",
                (time.time(), elapsed_s, kind, str(error), request_id, stage, number))
            if cursor.rowcount != 1:
                raise ValueError("Only the active attempt can fail")

    def attempts(self, request_id=None):
        if request_id is None:
            query, args = "SELECT * FROM attempts ORDER BY started_at,request_id,stage,number", ()
        else:
            query, args = "SELECT * FROM attempts WHERE request_id=? ORDER BY started_at,stage,number", (request_id,)
        return [dict(row) for row in self._db.execute(query, args)]

    def export(self, request_id, directory, stage="caption"):
        """Export only committed results. A partial write never replaces a result."""
        result = self.result(request_id, stage)
        if result is None:
            raise ValueError("No complete result exists")
        if len(request_id) != 64 or any(c not in "0123456789abcdef" for c in request_id):
            raise ValueError("Invalid request id")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{request_id}.{stage}.json"
        fd, temporary = tempfile.mkstemp(prefix=".result-", dir=directory)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(canonical_json(result) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def close(self):
        if self._lock_handle is not None:
            raise StoreBusy("Exit the exclusive context before closing")
        self._db.close()
