"""Bounded, invocation-local spill storage for validated native video metadata.

No decoded frames, model tensors, or executable pickle data are stored. The
engine verifies source bytes before each lookup. Spill storage avoids the LRU
cliff when the second stage traverses more episodes than fit in memory.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import threading

from .manifest import canonical_json


class PreparedMetadataSpool:
    def __init__(self, max_bytes=128 * 1024**2, max_entry_bytes=8 * 1024**2):
        if max_bytes <= 0 or max_entry_bytes <= 0:
            raise ValueError("Metadata storage bounds must be positive")
        self.max_bytes, self.max_entry_bytes = max_bytes, max_entry_bytes
        self._directory = tempfile.TemporaryDirectory(prefix="bir-metadata-")
        self._entries = {}
        self._lock = threading.RLock()
        self.retained_bytes = self.hits = self.writes = self.refused = 0
        self.write_errors = self.unremoved_partial_bound_bytes = 0
        self.poisoned = False

    def put(self, job, prepared):
        from .native import PreparedEpisode
        if not isinstance(prepared, PreparedEpisode):
            return False
        value = dict(schema_version="bir_prepared_metadata_v1", request_id=job.request_id,
                     profile=prepared.profile, metadata=prepared.metadata, videos={})
        for role, video in prepared.videos.items():
            value["videos"][role] = dict(sha256=video.sha256, rate=str(video.rate),
                pts_us=[str(t) for t in video.pts_us], duration_us=video.duration_us,
                size=list(video.size), cfr=video.cfr)
        payload = canonical_json(value).encode()
        with self._lock:
            if self.poisoned:
                raise RuntimeError("Metadata spill storage failed cleanup; no further writes are permitted")
            if job.request_id in self._entries:
                return True
            if len(payload) > self.max_entry_bytes or self.retained_bytes + len(payload) > self.max_bytes:
                self.refused += 1
                return False
            # A digest-derived basename cannot escape the private temporary root.
            path = Path(self._directory.name) / hashlib.sha256(job.request_id.encode()).hexdigest()
            try:
                path.write_bytes(payload)
            except BaseException:
                self.write_errors += 1
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    # The failed write cannot exceed its planned payload. Keep
                    # that conservative reservation and stop all later writes.
                    self.unremoved_partial_bound_bytes += len(payload)
                    self.poisoned = True
                    raise RuntimeError("Metadata spill storage failed cleanup; preserve the failure") from cleanup_error
                raise
            self._entries[job.request_id] = (path, hashlib.sha256(payload).hexdigest())
            self.retained_bytes += len(payload)
            self.writes += 1
        return True

    def get(self, job, backend):
        from .native import PreparedEpisode
        with self._lock:
            record = self._entries.get(job.request_id)
            if record is None:
                return None
            path, digest = record
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("Prepared metadata changed inside this invocation")
            value = json.loads(payload)
            if (value["schema_version"] != "bir_prepared_metadata_v1" or
                    value["request_id"] != job.request_id or
                    value["metadata"]["input_sha256"] != dict(job.source_sha256)):
                raise ValueError("Prepared metadata identity mismatch")
            paths = job.camera_paths()
            if set(value["videos"]) != set(paths):
                raise ValueError("Prepared metadata camera roles changed")
            videos = {}
            for role, item in value["videos"].items():
                if item["sha256"] != job.source_sha256[role]:
                    raise ValueError("Prepared metadata source identity mismatch")
                videos[role] = backend.media.Video(paths[role], item["sha256"], Fraction(item["rate"]),
                    tuple(Fraction(t) for t in item["pts_us"]), item["duration_us"],
                    tuple(item["size"]), item["cfr"])
            self.hits += 1
        return PreparedEpisode(job.as_native_dict(), videos, value["profile"], value["metadata"])

    def stats(self):
        with self._lock:
            return dict(max_bytes=self.max_bytes, max_entry_bytes=self.max_entry_bytes,
                        written_bytes=self.retained_bytes, hits=self.hits, writes=self.writes, refused=self.refused,
                        write_errors=self.write_errors, poisoned=self.poisoned,
                        unremoved_partial_bound_bytes=self.unremoved_partial_bound_bytes)

    def close(self):
        self._directory.cleanup()
