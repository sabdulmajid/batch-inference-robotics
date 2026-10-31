"""Manifest validation and content-based request identity.

Paths locate inputs. Source digests, native settings, and the supplied execution
contract identify work. Moving an unchanged input does not change its identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ManifestError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class Job:
    id: str
    video: Path
    profile: str
    left_wrist: Path | None
    right_wrist: Path | None
    yam32: bool
    instruction: str
    source_sha256: Mapping[str, str]
    contract: Mapping[str, Any]
    request_id: str

    def as_native_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "video": str(self.video), "profile": self.profile,
            "left_wrist": str(self.left_wrist) if self.left_wrist else None,
            "right_wrist": str(self.right_wrist) if self.right_wrist else None,
            "yam32": self.yam32, "instruction": self.instruction,
        }

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": "bir_request_v1", "id": self.id,
            "source_sha256": dict(self.source_sha256), "profile": self.profile,
            "yam32": self.yam32, "instruction": self.instruction,
            "contract": dict(self.contract),
        }

    def verify_sources(self) -> None:
        for role, path in self.camera_paths().items():
            if file_sha256(path) != self.source_sha256[role]:
                raise ManifestError(f"Source changed after manifest load: {self.id}/{role}")

    def camera_paths(self) -> dict[str, Path]:
        paths = {"main": self.video}
        if self.left_wrist is not None:
            paths.update(left_wrist=self.left_wrist, right_wrist=self.right_wrist)
        return paths


def load_manifest(path: str | Path, *, contract: Mapping[str, Any]) -> list[Job]:
    """Read native-style JSONL. Pin every output-affecting option in contract.

    ``contract`` must include model and source revisions and the native or
    experimental configuration. The CLI constructs it from the verified lock.
    An optional source_sha256 mapping in each row verifies supplied checksums.
    """
    if not contract or not isinstance(contract, Mapping):
        raise ManifestError("A nonempty execution contract is required")
    # Detach from callers and reject values that have no stable JSON encoding.
    frozen_contract = json.loads(canonical_json(dict(contract)))
    manifest = Path(path).expanduser().resolve(strict=True)
    jobs, seen = [], set()
    allowed = {"id", "video", "profile", "left_wrist", "right_wrist", "yam32", "instruction", "source_sha256"}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ManifestError("Each row must be an object")
            if unknown := set(row) - allowed:
                raise ManifestError(f"Unknown fields: {sorted(unknown)}")
            episode_id = row.get("id")
            if not isinstance(episode_id, str) or not episode_id.strip() or len(episode_id) > 200:
                raise ManifestError("id must be a nonempty string of at most 200 characters")
            if episode_id in seen:
                raise ManifestError(f"Duplicate episode id: {episode_id}")
            profile = row.get("profile", "human")
            if not isinstance(profile, str) or profile.lower() not in {"human", "yam", "dex", "tianji"}:
                raise ManifestError("profile must be human, yam, dex, or tianji")
            profile = profile.lower()
            yam32, instruction = row.get("yam32", False), row.get("instruction", "")
            if not isinstance(yam32, bool):
                raise ManifestError("yam32 must be a boolean")
            if not isinstance(instruction, str) or len(instruction.strip()) > 2000:
                raise ManifestError("instruction must be a string of at most 2000 characters")
            if bool(row.get("left_wrist")) != bool(row.get("right_wrist")):
                raise ManifestError("Both wrist videos are required together")
            if profile == "human" and row.get("left_wrist"):
                raise ManifestError("The human profile accepts one camera")
            if yam32 and (profile != "yam" or not row.get("left_wrist")):
                raise ManifestError("yam32 requires YAM and three cameras")
            def resolve(value: Any) -> Path:
                if not isinstance(value, str) or not value.strip():
                    raise ManifestError("Camera paths must be nonempty strings")
                candidate = Path(value).expanduser()
                candidate = (manifest.parent / candidate).resolve(strict=True)
                if not candidate.is_file() or not 0 < candidate.stat().st_size <= 256 * 1024**2:
                    raise ManifestError("Each video must be a nonempty file of at most 256 MiB")
                return candidate
            paths = {"main": resolve(row.get("video"))}
            if row.get("left_wrist"):
                paths.update(left_wrist=resolve(row["left_wrist"]), right_wrist=resolve(row["right_wrist"]))
            digests = {role: file_sha256(value) for role, value in paths.items()}
            expected = row.get("source_sha256")
            if expected is not None and expected != digests:
                raise ManifestError("Supplied source checksums do not match camera files")
            identity = dict(schema_version="bir_request_v1", id=episode_id, source_sha256=digests,
                            profile=profile, yam32=yam32, instruction=instruction, contract=frozen_contract)
            request_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            jobs.append(Job(episode_id, paths["main"], profile, paths.get("left_wrist"), paths.get("right_wrist"),
                            yam32, instruction, digests, frozen_contract, request_id))
            seen.add(episode_id)
        except (ValueError, OSError, TypeError) as exc:
            raise ManifestError(f"Manifest line {line_number}: {exc}") from exc
    if not jobs:
        raise ManifestError("The manifest contains no jobs")
    return jobs
