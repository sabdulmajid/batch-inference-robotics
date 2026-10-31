"""Persistent staged native execution with bounded CPU preparation."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import errno
import threading
import time

from .store import RetryExhausted


class TransientError(RuntimeError):
    """An explicitly classified transport or worker failure, safe to retry once."""


def is_transient(error):
    return isinstance(error, (TransientError, TimeoutError, ConnectionError)) or (
        isinstance(error, OSError) and error.errno in {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.ECONNRESET})


class Engine:
    def __init__(self, backend_factory, store, *, mode="staged", prefetch=1,
                 prepared_cache_size=8, max_transient_retries=1):
        if mode not in {"staged", "reference"}:
            raise ValueError("mode must be staged or reference")
        if not isinstance(prefetch, int) or not 0 <= prefetch <= 8:
            raise ValueError("prefetch must be between zero and eight")
        if not isinstance(prepared_cache_size, int) or not 0 <= prepared_cache_size <= 100:
            raise ValueError("prepared_cache_size must be between zero and 100")
        if max_transient_retries not in (0, 1):
            raise ValueError("At most one transient retry is permitted")
        self.backend_factory, self.store = backend_factory, store
        self.mode, self.prefetch = mode, prefetch if mode == "staged" else 0
        # The native reference probes once, then uses that metadata for both
        # stages of the same episode. Drop it when that episode closes.
        self.prepared_cache_size = prepared_cache_size if mode == "staged" else 1
        self.max_transient_retries = max_transient_retries
        self._prepared = OrderedDict()
        self._prepared_lock = threading.RLock()

    def _prepare(self, backend, job):
        # Run once per stage unless bounded metadata reuse is enabled. Source
        # hashing protects against files changed after identity was established.
        job.verify_sources()
        with self._prepared_lock:
            if job.request_id in self._prepared:
                result = self._prepared.pop(job.request_id)
                self._prepared[job.request_id] = result
                return result
        result = backend.prepare(job.as_native_dict())
        metadata = getattr(result, "metadata", {})
        if "input_sha256" in metadata and metadata["input_sha256"] != dict(job.source_sha256):
            raise ValueError("Native input checksums differ from the request identity")
        with self._prepared_lock:
            if self.prepared_cache_size:
                self._prepared[job.request_id] = result
                while len(self._prepared) > self.prepared_cache_size:
                    self._prepared.popitem(last=False)
        return result

    def _stream_prepared(self, backend, jobs):
        if not self.prefetch:
            for job in jobs:
                try:
                    yield job, self._prepare(backend, job), None
                except Exception as exc:
                    yield job, None, exc
            return
        # One CPU worker prevents decoder oversubscription. At most prefetch+1
        # metadata records are queued. GPU stage work remains in the main thread.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bir-prepare") as executor:
            iterator, pending = iter(jobs), deque()
            for _ in range(self.prefetch + 1):
                if (job := next(iterator, None)) is not None:
                    pending.append((job, executor.submit(self._prepare, backend, job)))
            while pending:
                job, future = pending.popleft()
                try:
                    prepared, error = future.result(), None
                except Exception as exc:
                    prepared, error = None, exc
                if (next_job := next(iterator, None)) is not None:
                    pending.append((next_job, executor.submit(self._prepare, backend, next_job)))
                yield job, prepared, error

    def _execute(self, backend, job, stage, prepared, preparation_error, records):
        while True:
            try:
                attempt = self.store.begin(job.request_id, stage, max_transient_retries=self.max_transient_retries)
            except RetryExhausted as exc:
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage,
                                    status="failed", error=str(exc), fresh_execution=False))
                return
            started = time.monotonic()
            try:
                if preparation_error is not None:
                    raise preparation_error
                if stage == "segment":
                    result = backend.segment(prepared)
                else:
                    result = backend.caption(prepared, self.store.result(job.request_id, "segment"))
                if not isinstance(result, dict):
                    raise ValueError("The native backend must return a result object")
                # A segmentation payload can contain its own integrity digest.
                # Keep it unchanged when passing it back to the native adapter.
                if stage == "caption":
                    result = dict(result, request_id=job.request_id, episode_id=job.id)
                self.store.complete(job.request_id, stage, attempt, result, time.monotonic() - started)
                status = "partial" if stage == "caption" and any(
                    row.get("status") != "complete" for row in result.get("segments", [])) else "complete"
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status=status,
                                    attempt=attempt, elapsed_s=time.monotonic() - started, fresh_execution=True))
                return
            except Exception as exc:
                kind = "transient" if is_transient(exc) else "deterministic"
                self.store.fail(job.request_id, stage, attempt, exc, kind=kind, elapsed_s=time.monotonic() - started)
                if kind == "transient" and attempt <= self.max_transient_retries:
                    # A preparation failure needs new input inspection. The
                    # native model session remains owned by this same worker.
                    try:
                        prepared, preparation_error = self._prepare(backend, job), None
                    except Exception as preparation_error_value:
                        preparation_error = preparation_error_value
                    continue
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status="failed",
                                    attempt=attempt, error=str(exc), error_kind=kind, fresh_execution=True))
                return

    def _stage(self, backend, jobs, stage, records):
        pending = []
        for job in jobs:
            existing = self.store.result(job.request_id, stage)
            if existing is not None:
                status = "partial" if stage == "caption" and any(
                    row.get("status") != "complete" for row in existing.get("segments", [])) else "resumed"
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status=status,
                                    fresh_execution=False))
            elif stage == "caption" and self.store.result(job.request_id, "segment") is None:
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status="blocked",
                                    error="Segmentation has no complete result", fresh_execution=False))
            else:
                pending.append(job)
        for job, prepared, error in self._stream_prepared(backend, pending):
            self._execute(backend, job, stage, prepared, error, records)

    def run(self, jobs, *, stages="all", output_dir=None):
        if stages not in {"all", "segment", "caption"}:
            raise ValueError("stages must be all, segment, or caption")
        jobs = list(jobs)
        if len({job.id for job in jobs}) != len(jobs):
            raise ValueError("Episode identities must be unique")
        selected_stages = ("segment", "caption") if stages == "all" else (stages,)
        started, records = time.monotonic(), []
        self._prepared.clear()
        with self.store.exclusive():
            before_attempts = {(a["request_id"], a["stage"], a["number"]) for a in self.store.attempts()}
            for job in jobs:
                self.store.register(job)
            if self.mode == "reference":
                for job in jobs:
                    backend = self.backend_factory()
                    try:
                        for stage in selected_stages:
                            self._stage(backend, [job], stage, records)
                    finally:
                        backend.close()
                        self._prepared.clear()
            else:
                backend = self.backend_factory()
                try:
                    for stage in selected_stages:
                        self._stage(backend, jobs, stage, records)
                finally:
                    backend.close()
            exported = []
            if output_dir is not None:
                for job in jobs:
                    final_stage = selected_stages[-1]
                    if self.store.result(job.request_id, final_stage) is not None:
                        exported.append(str(self.store.export(job.request_id, output_dir, final_stage)))
            new_attempts = [dict(request_id=a["request_id"], stage=a["stage"], number=a["number"],
                                 state=a["state"], elapsed_s=a["elapsed_s"], error_kind=a["error_kind"])
                            for a in self.store.attempts()
                            if (a["request_id"], a["stage"], a["number"]) not in before_attempts]
        self._prepared.clear()
        return dict(schema_version="bir_run_v1", mode=self.mode, elapsed_s=time.monotonic() - started,
                    records=records, exports=exported,
                    attempts=new_attempts,
                    complete=all(row["status"] in {"complete", "resumed"} for row in records),
                    execution_complete=all(row["status"] in {"complete", "resumed", "partial"} for row in records),
                    fresh_stage_attempts=len(new_attempts),
                    resumed_stages=sum(row["status"] == "resumed" for row in records))
