"""Persistent staged native execution with bounded CPU preparation."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
                 prepared_cache_size=8, max_transient_retries=1, spill_metadata=False):
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
        self.spill_metadata = bool(spill_metadata) and mode == "staged"
        self._metadata_spool = None
        self._metadata_spool_stats = None
        self._prepared = OrderedDict()
        self._prepared_lock = threading.RLock()
        self._telemetry_lock = threading.RLock()
        self._reset_telemetry()

    def _reset_telemetry(self):
        with self._telemetry_lock:
            self._cpu_work = dict(preparation_requests=0, metadata_cache_hits=0,
                source_verification_calls=0, source_verification_failures=0, source_verification_work_seconds=0.,
                backend_preparation_calls=0, backend_preparation_failures=0, backend_preparation_work_seconds=0.)
            self._backend_telemetry = []

    def _measure_preparation(self, category, callback):
        started, failed = time.monotonic(), False
        try:
            return callback()
        except BaseException:
            failed = True
            raise
        finally:
            seconds = time.monotonic() - started
            with self._telemetry_lock:
                self._cpu_work[category + '_calls'] += 1
                self._cpu_work[category + '_failures'] += int(failed)
                self._cpu_work[category + '_work_seconds'] += seconds

    def _close_backend(self, backend, *, episode_id=None):
        """Save each cache's lifetime counters before backend cleanup."""
        snapshot = dict(episode_id=episode_id, decoded_frame_cache=None)
        try:
            cache = getattr(backend, 'frame_cache', None)
            if cache is not None:
                snapshot['decoded_frame_cache'] = cache.stats()
        except Exception as error:
            snapshot['cache_measurement_error_type'] = type(error).__name__
        finally:
            with self._telemetry_lock:
                snapshot['backend_ordinal'] = len(self._backend_telemetry)
                self._backend_telemetry.append(snapshot)
            backend.close()

    def _telemetry(self):
        with self._telemetry_lock:
            return dict(schema_version='bir_engine_work_telemetry_v1',
                cpu_preparation=dict(self._cpu_work), backends=list(self._backend_telemetry),
                prepared_metadata_spool=self._metadata_spool_stats,
                prefetch=self.prefetch, prepared_metadata_cache_capacity=self.prepared_cache_size,
                cpu_timing_scope='Summed engine worker time includes failed calls. Preparation can overlap inference; these sums are not end-to-end wall time. Initial CLI manifest hashing is outside these counters.',
                decoded_cache_scope='Cumulative counters per backend cache lifetime, captured before close. Byte counts cover retained RGB payloads only; null means no decoded-frame cache.')

    def _prepare(self, backend, job):
        # Run once per stage unless bounded metadata reuse is enabled. Source
        # hashing protects against files changed after identity was established.
        with self._telemetry_lock:
            self._cpu_work['preparation_requests'] += 1
        self._measure_preparation('source_verification', job.verify_sources)
        with self._prepared_lock:
            if job.request_id in self._prepared:
                result = self._prepared.pop(job.request_id)
                self._prepared[job.request_id] = result
                with self._telemetry_lock:
                    self._cpu_work['metadata_cache_hits'] += 1
                return result
        if self._metadata_spool is not None:
            result = self._metadata_spool.get(job, backend)
            if result is not None:
                return result
        result = self._measure_preparation('backend_preparation', lambda: backend.prepare(job.as_native_dict()))
        metadata = getattr(result, "metadata", {})
        if "input_sha256" in metadata and metadata["input_sha256"] != dict(job.source_sha256):
            raise ValueError("Native input checksums differ from the request identity")
        if self._metadata_spool is not None:
            self._metadata_spool.put(job, result)
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
                committed = time.monotonic() - self._run_started
                status = "partial" if stage == "caption" and any(
                    row.get("status") != "complete" for row in result.get("segments", [])) else "complete"
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status=status,
                                    attempt=attempt, elapsed_s=time.monotonic() - started, fresh_execution=True,
                                    stage_committed_after_run_start_s=committed))
                return
            except Exception as exc:
                from .native import NativeModelLoadBlocked, NativeModelLoadError
                kind = "transient" if is_transient(exc) else "deterministic"
                self.store.fail(job.request_id, stage, attempt, exc, kind=kind, elapsed_s=time.monotonic() - started)
                if isinstance(exc, NativeModelLoadError):
                    # This type is raised only inside native loading/verification.
                    # An input-specific generation failure must not stop other jobs.
                    self._model_load_failure = dict(id=job.id, request_id=job.request_id,
                        stage=stage, model_kind=exc.kind, error_type=exc.original_error_type,
                        error=str(exc))
                if kind == "transient" and attempt <= self.max_transient_retries:
                    # A preparation failure needs new input inspection. The
                    # native model session remains owned by this same worker.
                    try:
                        prepared, preparation_error = self._prepare(backend, job), None
                    except Exception as preparation_error_value:
                        preparation_error = preparation_error_value
                    continue
                status = 'blocked' if isinstance(exc, NativeModelLoadBlocked) else 'failed'
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status=status,
                                    attempt=attempt, error=str(exc), error_kind=kind, fresh_execution=True))
                return

    def _record_load_blocked(self, job, stage, records):
        records.append(dict(id=job.id, request_id=job.request_id, stage=stage,
            status="blocked", fresh_execution=False, error_kind="deterministic",
            error="Native model loading failed earlier in this invocation; no new stage was attempted",
            blocked_by=dict(self._model_load_failure)))

    def _stage(self, backend, jobs, stage, records):
        pending = []
        for job in jobs:
            existing = self.store.result(job.request_id, stage)
            if existing is not None:
                status = "partial" if stage == "caption" and any(
                    row.get("status") != "complete" for row in existing.get("segments", [])) else "resumed"
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status=status,
                                    fresh_execution=False,
                                    cached_result_observed_after_run_start_s=time.monotonic() - self._run_started))
            elif self._model_load_failure is not None:
                self._record_load_blocked(job, stage, records)
            elif stage == "caption" and self.store.result(job.request_id, "segment") is None:
                records.append(dict(id=job.id, request_id=job.request_id, stage=stage, status="blocked",
                                    error="Segmentation has no complete result", fresh_execution=False))
            else:
                pending.append(job)
        prepared_jobs = self._stream_prepared(backend, pending)
        try:
            for index, (job, prepared, error) in enumerate(prepared_jobs):
                self._execute(backend, job, stage, prepared, error, records)
                if self._model_load_failure is not None:
                    for remaining in pending[index + 1:]:
                        self._record_load_blocked(remaining, stage, records)
                    break
        finally:
            # Drain only preparation already submitted before the failure. Do
            # not close a backend while a queued CPU task still uses it.
            prepared_jobs.close()

    def run(self, jobs, *, stages="all", output_dir=None):
        if stages not in {"all", "segment", "caption"}:
            raise ValueError("stages must be all, segment, or caption")
        jobs = list(jobs)
        if len({job.id for job in jobs}) != len(jobs):
            raise ValueError("Episode identities must be unique")
        selected_stages = ("segment", "caption") if stages == "all" else (stages,)
        started, records = time.monotonic(), []
        self._run_started = started
        self._model_load_failure = None
        self._reset_telemetry()
        self._prepared.clear()
        self._metadata_spool_stats = None
        with ExitStack() as cleanup, self.store.exclusive():
            if self.spill_metadata:
                from .metadata_cache import PreparedMetadataSpool
                self._metadata_spool = PreparedMetadataSpool()
                def close_spool():
                    self._metadata_spool_stats = self._metadata_spool.stats()
                    self._metadata_spool.close()
                    self._metadata_spool = None
                cleanup.callback(close_spool)
            before_attempts = {(a["request_id"], a["stage"], a["number"]) for a in self.store.attempts()}
            for job in jobs:
                self.store.register(job)
            if self.mode == "reference":
                for job in jobs:
                    backend = self.backend_factory() if self._model_load_failure is None else None
                    try:
                        for stage in selected_stages:
                            self._stage(backend, [job], stage, records)
                    finally:
                        if backend is not None:
                            self._close_backend(backend, episode_id=job.id)
                        self._prepared.clear()
            else:
                backend = self.backend_factory()
                try:
                    for stage in selected_stages:
                        self._stage(backend, jobs, stage, records)
                finally:
                    self._close_backend(backend)
            exported, export_times = [], {}
            if output_dir is not None:
                for job in jobs:
                    final_stage = selected_stages[-1]
                    if self.store.result(job.request_id, final_stage) is not None:
                        exported.append(str(self.store.export(job.request_id, output_dir, final_stage)))
                        export_times[job.request_id] = time.monotonic() - started
            new_attempts = [dict(request_id=a["request_id"], stage=a["stage"], number=a["number"],
                                 state=a["state"], elapsed_s=a["elapsed_s"], error_kind=a["error_kind"])
                            for a in self.store.attempts()
                            if (a["request_id"], a["stage"], a["number"]) not in before_attempts]
        self._prepared.clear()
        episode_availability = []
        for job in jobs:
            rows = [row for row in records if row['request_id'] == job.request_id]
            final = next(row for row in rows if row['stage'] == selected_stages[-1])
            fresh = any(row['fresh_execution'] for row in rows)
            cached = any(not row['fresh_execution'] and row['status'] in {'resumed', 'partial'} for row in rows)
            origin = 'mixed' if fresh and cached else 'fresh' if fresh else 'cached' if cached else 'no_execution'
            available = None
            if final['status'] in {'complete', 'resumed', 'partial'}:
                available = export_times.get(job.request_id) if output_dir is not None else final.get(
                    'stage_committed_after_run_start_s', final.get('cached_result_observed_after_run_start_s'))
            episode_availability.append(dict(episode_id=job.id, request_id=job.request_id,
                final_stage=selected_stages[-1], status='complete' if final['status'] == 'resumed' else final['status'],
                execution_origin=origin,
                fresh_complete=len(rows) == len(selected_stages) and all(
                    row['fresh_execution'] and row['status'] == 'complete' for row in rows),
                final_stage_committed_after_run_start_s=final.get('stage_committed_after_run_start_s'),
                result_available_after_run_start_s=available))
        return dict(schema_version="bir_run_v1", mode=self.mode, elapsed_s=time.monotonic() - started,
                    records=records, exports=exported, telemetry=self._telemetry(),
                    model_load_failure=self._model_load_failure,
                    episode_result_availability=dict(arrival='all_jobs_at_engine_run_start',
                        boundary='serialized_export_completed' if output_dir is not None else 'final_stage_committed_or_cached_result_observed',
                        limitation=('Exports occur after all inference and backend cleanup, in manifest order; export latencies can therefore be close.'
                                    if output_dir is not None else 'No serialized export was requested; availability refers to the stored requested-stage result.'),
                        episodes=episode_availability),
                    attempts=new_attempts,
                    complete=all(row["status"] in {"complete", "resumed"} for row in records),
                    execution_complete=all(row["status"] in {"complete", "resumed", "partial"} for row in records),
                    fresh_stage_attempts=len(new_attempts),
                    resumed_stages=sum(row["status"] == "resumed" for row in records))
