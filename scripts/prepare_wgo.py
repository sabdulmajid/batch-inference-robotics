"""Inspect and prepare a bounded native WGO subset in a CPU allocation.

Videos are copied from the pinned Parquet bytes without re-encoding. Selection
uses source properties and independent references, never model predictions.
The output split remains a proposal until the coordinator reviews and commits it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

from bir.cli import atomic_json
from bir.manifest import canonical_json, file_sha256
from bir.native import NativeBackend


@dataclass(frozen=True)
class DatasetPin:
    repository: str
    revision: str
    parquet_sha256: str


WGO = DatasetPin("macrodata/WGO-Bench", "9cfb7bb98d60a58319f4b3d7b41374eedfb7980c",
                 "24c6c23447753385b9cea0d17f17d7a8aa74131d800d9bc9ab74d7669d7404ab")
_COLUMNS = ("id", "instruction", "segments", "metadata")
_GROUP_FIELDS = ("scene_id", "scene", "session_id", "recording_id", "source_episode_id",
                 "episode_id", "trajectory_id", "video_id")
_SPLIT_SEED = "bir-wgo-group-split-v1"


def embedded_video(value):
    """Accept embedded bytes only. Never fetch or follow a dataset path."""
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("Video has no embedded byte payload")
    if not 0 < len(value) <= 256 * 1024**2:
        raise ValueError("Embedded video must be nonempty and at most 256 MiB")
    return bytes(value)


def _record(row, index):
    identity, instruction = row.get("id"), row.get("instruction")
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 200:
        raise ValueError("Invalid episode id")
    if not isinstance(instruction, str) or len(instruction.strip()) > 2000:
        raise ValueError("Invalid task instruction")
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    segments = row.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Reference segments must be a list")
    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Reference segments must be objects")
        start, end, caption = segment.get("start_sec"), segment.get("end_sec"), segment.get("subtask")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (float, int)) or
                not isinstance(end, (float, int)) or not math.isfinite(start) or not math.isfinite(end) or
                not 0 <= start < end or not isinstance(caption, str) or not caption.strip()):
            raise ValueError("Invalid independent reference segment")
        normalized.append(dict(start_s=float(start), end_s=float(end), caption=caption))
    task = re.sub(r"\s+", " ", instruction.strip().casefold())
    return dict(row_index=index, id=identity, instruction=instruction, metadata=metadata,
                references=normalized, task_hash=hashlib.sha256(task.encode()).hexdigest() if task else None,
                source_family="homer" if identity.lower().startswith("homer_") else identity.split("_", 1)[0],
                profile="human" if identity.lower().startswith("homer_") else None)


def assign_groups(records):
    """Group shared instructions OR explicit scene/session/source-episode keys.

    Connected components prevent an episode from bridging development and final
    groups through another shared field. No scene identity is inferred from video.
    Unknown source metadata leaves residual leakage risk for coordinator review.
    """
    parents = {r["id"]: r["id"] for r in records}
    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def union(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)
    seen_tokens = {}
    for record in records:
        tokens = []
        if record["task_hash"]:
            tokens.append(("task", record["task_hash"]))
        if record.get("source_sha256"):
            tokens.append(("video_sha256", record["source_sha256"]))
        fields = []
        for key in _GROUP_FIELDS:
            value = record["metadata"].get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                tokens.append((record["source_family"], key, str(value)))
                fields.append(key)
        record["group_metadata_fields"] = fields
        for token in tokens:
            if token in seen_tokens:
                union(record["id"], seen_tokens[token])
            else:
                seen_tokens[token] = record["id"]
    groups = {}
    for record in records:
        groups.setdefault(find(record["id"]), []).append(record["id"])
    for record in records:
        members = sorted(groups[find(record["id"])])
        digest = hashlib.sha256((_SPLIT_SEED + "\n" + "\n".join(members)).encode()).hexdigest()
        record["group_id"] = digest
        record["proposed_split"] = "final" if int(digest[:16], 16) % 5 == 0 else "development"


def inspect_parquet(path, *, pin=WGO):
    import pyarrow.parquet as pq
    path = Path(path).resolve(strict=True)
    if file_sha256(path) != pin.parquet_sha256:
        raise ValueError("Parquet bytes do not match the pinned dataset checksum")
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if not set((*_COLUMNS, "video")) <= names:
        raise ValueError("Parquet is missing required WGO columns")
    records, exclusions, identities = [], [], set()
    index = 0
    for batch in parquet.iter_batches(batch_size=32, columns=list(_COLUMNS), use_threads=False):
        for row in batch.to_pylist():
            try:
                record = _record(row, index)
                if record["id"] in identities:
                    raise ValueError("Duplicate episode id in dataset")
                identities.add(record["id"])
                records.append(record)
            except (ValueError, TypeError, KeyError) as exc:
                exclusions.append(dict(row_index=index, id=row.get("id") if isinstance(row.get("id"), str) else None,
                                       reason=str(exc), native_supported=False))
            index += 1
    assign_groups(records)
    return dict(schema_version="bir_wgo_index_v1", dataset=pin.__dict__,
                row_count=index, columns=[dict(name=field.name, type=str(field.type)) for field in parquet.schema_arrow],
                records=records, exclusions=exclusions,
                split_status="proposal_not_frozen", split_rule=dict(seed=_SPLIT_SEED, final_hash_modulus=5,
                grouping="connected components of normalized task instruction, identical video bytes when extracted, and explicit scene/session/source-episode keys",
                metadata_fields=list(_GROUP_FIELDS)))


def _public(record, pin):
    allowed = ("id", "source_family", "profile", "task_hash", "group_id", "group_metadata_fields",
               "proposed_split", "source_sha256", "duration_s", "source_fps", "source_frames", "size",
               "native_supported", "reference_count", "references_valid", "exclusion_reason", "selected")
    result = {key: record[key] for key in allowed if key in record}
    result.update(dataset_repository=pin.repository, dataset_revision=pin.revision,
                  independent_references_available=bool(record.get("references_valid")))
    return result


def prepare_parquet(path, source, output_dir, *, max_episodes=3, split="development", pin=WGO,
                    public_manifest=None):
    """Validate original human videos and extract shortest eligible episodes."""
    import pyarrow.parquet as pq
    if not isinstance(max_episodes, int) or isinstance(max_episodes, bool) or not 1 <= max_episodes <= 3:
        raise ValueError("Initial native preparation permits one to three episodes")
    if split not in {"development", "final"}:
        raise ValueError("Unknown proposed split")
    index = inspect_parquet(path, pin=pin)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory; preserve the previous selection")
    videos_dir = output / "videos"
    videos_dir.mkdir(exist_ok=True)
    by_row = {record["row_index"]: record for record in index["records"]}
    candidates = []
    backend = NativeBackend(source)
    try:
        with tempfile.TemporaryDirectory(prefix=".wgo-candidates-", dir=output) as temporary:
            parquet, row_index = pq.ParquetFile(path), 0
            for batch in parquet.iter_batches(batch_size=1, columns=["video"], use_threads=False):
                for row in batch.to_pylist():
                    record = by_row.get(row_index)
                    row_index += 1
                    if record is None:
                        continue
                    candidate = {**record, "native_supported": False, "selected": False,
                                 "reference_count": len(record["references"]), "references_valid": False}
                    if record["profile"] != "human":
                        candidate["exclusion_reason"] = "Only HomER human egocentric inputs are enabled in the first gate"
                        candidates.append(candidate)
                        continue
                    video_path = Path(temporary) / (hashlib.sha256(record["id"].encode()).hexdigest() + ".mp4")
                    try:
                        video_bytes = embedded_video(row["video"])
                        candidate["source_sha256"] = hashlib.sha256(video_bytes).hexdigest()
                        video_path.write_bytes(video_bytes)
                        del video_bytes
                        prepared = backend.prepare(dict(id=record["id"], video=str(video_path), profile="human",
                                                        instruction=record["instruction"]))
                        metadata = prepared.metadata
                        if metadata["input_sha256"]["main"] != candidate["source_sha256"]:
                            raise ValueError("Video changed during native validation")
                        camera = metadata["cameras"]["main"]
                        candidate.update(native_supported=True, duration_s=metadata["duration_s"],
                                         source_fps=camera["source_fps"], source_frames=camera["frames"], size=camera["size"])
                        candidate["references_valid"] = bool(record["references"]) and all(
                            segment["end_s"] <= metadata["duration_s"] + 0.001 for segment in record["references"])
                        if not candidate["references_valid"]:
                            candidate["exclusion_reason"] = "Independent reference intervals are missing or exceed source duration"
                    except Exception as exc:
                        candidate["exclusion_reason"] = f"Native validation rejected input ({type(exc).__name__})"
                        candidate["private_validation_error"] = str(exc)
                    finally:
                        video_path.unlink(missing_ok=True)
                    candidates.append(candidate)
    finally:
        backend.close()
    # Video digests can join two source episodes with different instructions.
    # Recompute connected groups before selection, then copy only final choices.
    assign_groups(candidates)
    eligible = sorted((record for record in candidates if record["native_supported"] and
                       record["references_valid"] and record["proposed_split"] == split),
                      key=lambda record: (record["duration_s"], record["id"]))
    selected_records = eligible[:max_episodes]
    selected_rows = {record["row_index"]: record for record in selected_records}
    for record in candidates:
        if record in selected_records:
            record["selected"] = True
        elif record["native_supported"] and record["references_valid"]:
            record["exclusion_reason"] = ("Reserved for the other proposed split" if record["proposed_split"] != split
                                          else "Outside the bounded shortest-duration smoke selection")
    parquet, row_index = pq.ParquetFile(path), 0
    for batch in parquet.iter_batches(batch_size=1, columns=["video"], use_threads=False):
        for row in batch.to_pylist():
            record = selected_rows.get(row_index)
            row_index += 1
            if record is not None:
                payload = embedded_video(row["video"])
                if hashlib.sha256(payload).hexdigest() != record["source_sha256"]:
                    raise ValueError("Selected video changed after native validation")
                (videos_dir / (record["source_sha256"] + ".mp4")).write_bytes(payload)
    selected = [(record, videos_dir / (record["source_sha256"] + ".mp4")) for record in selected_records]
    jobs, references = [], []
    for candidate, video_path in selected:
        jobs.append(dict(id=candidate["id"], video=str(video_path.relative_to(output)), profile="human",
                         instruction=candidate["instruction"], source_sha256={"main": candidate["source_sha256"]}))
        references.append(dict(id=candidate["id"], group_id=candidate["group_id"], proposed_split=candidate["proposed_split"],
                               instruction=candidate["instruction"], segments=candidate["references"],
                               dataset_repository=pin.repository, dataset_revision=pin.revision,
                               reference_type="dataset_provided_manual_annotations"))
    (output / "jobs.jsonl").write_text("".join(canonical_json(row) + "\n" for row in jobs))
    (output / "references.jsonl").write_text("".join(canonical_json(row) + "\n" for row in references))
    summary = dict(schema_version="bir_wgo_preparation_v1", dataset=pin.__dict__,
                   split_status="proposal_not_frozen", proposed_split=split, split_rule=index["split_rule"],
                   selection_rule="shortest native-supported HomER duration, then episode id; valid independent references required",
                   selected_ids=[row["id"] for row in jobs], selected_count=len(jobs), max_episodes=max_episodes,
                   candidates=[_public(record, pin) for record in candidates], invalid_records=index["exclusions"],
                   new_model_calls=0, reencoded_videos=0,
                   limitations=["Scene leakage remains possible where source metadata has no shared scene/session identifier",
                                "The coordinator must review and commit exact splits before final evaluation",
                                "Native CPU validation does not establish GPU feasibility or annotation quality"])
    index["records"] = candidates
    atomic_json(output / "private-index.json", index)
    atomic_json(output / "preparation.json", summary)
    if public_manifest:
        atomic_json(public_manifest, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Verify pinned Parquet and index non-video columns")
    inspect.add_argument("--parquet", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True, help="Private metadata index")
    prepare = commands.add_parser("prepare", help="Validate and extract at most three native human episodes")
    prepare.add_argument("--parquet", type=Path, required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--max-episodes", type=int, default=3)
    prepare.add_argument("--split", choices=("development", "final"), default="development")
    prepare.add_argument("--public-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_parquet(args.parquet)
            atomic_json(args.output, result)
            print(canonical_json(dict(status="indexed", rows=result["row_count"], invalid_records=len(result["exclusions"]),
                                      split_status="proposal_not_frozen")))
        else:
            result = prepare_parquet(args.parquet, args.source, args.output_dir, max_episodes=args.max_episodes,
                                     split=args.split, public_manifest=args.public_manifest)
            print(canonical_json(dict(status="prepared" if result["selected_count"] else "no_supported_selection",
                                      selected_ids=result["selected_ids"], selected_count=result["selected_count"],
                                      split_status="proposal_not_frozen", new_model_calls=0)))
            return 0 if result["selected_count"] else 1
        return 0
    except Exception as exc:
        print(canonical_json(dict(status="failed", error_type=type(exc).__name__, error=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
