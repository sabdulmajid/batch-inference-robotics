#!/usr/bin/env python3
"""Extract the frozen phase-two source bytes in a bounded CPU allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bir.cli import atomic_json
from bir.manifest import canonical_json, file_sha256
from bir.native import NativeBackend
from bir.resource_ledger import reject_restarted_execution
from prepare_wgo import WGO, embedded_video


def load_selection(path):
    selection = json.loads(Path(path).read_text())
    body = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if hashlib.sha256(canonical_json(body).encode()).hexdigest() != selection['selection_sha256']:
        raise ValueError('Phase-two selection digest changed')
    if selection['dataset'] != WGO.__dict__:
        raise ValueError('Phase-two selection uses a different dataset')
    episodes = selection['episodes']
    if not 1 <= len(episodes) <= 100 or len({e['episode_id'] for e in episodes}) != len(episodes):
        raise ValueError('Phase-two selection must have one to 100 unique episodes')
    if any(e['profile'] != 'human' or e['instruction'] != '' for e in episodes):
        raise ValueError('Phase-two protocol requires human inputs and empty instructions')
    return selection


def prepare(parquet_path, source, selection, output_dir, *, backend_factory=NativeBackend):
    import pyarrow.parquet as pq
    started = time.monotonic()
    parquet_path, output = Path(parquet_path).resolve(strict=True), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh extraction directory')
    if file_sha256(parquet_path) != WGO.parquet_sha256:
        raise ValueError('Dataset bytes differ from the pinned Parquet checksum')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'videos').mkdir()
    wanted = {e['episode_id']: e for e in selection['episodes']}
    paths = {}
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=1, columns=['id', 'video'], use_threads=False):
        row = batch.to_pylist()[0]
        episode_id = row['id']
        if episode_id not in wanted:
            continue
        if episode_id in paths:
            raise ValueError('Duplicate selected episode in the dataset')
        payload = embedded_video(row['video'])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != wanted[episode_id]['source_sha256']:
            raise ValueError('Selected source bytes differ from the frozen selection')
        path = output / 'videos' / (digest + '.mp4')
        path.write_bytes(payload)
        paths[episode_id] = path
    if set(paths) != set(wanted):
        raise ValueError('Dataset does not contain every selected episode')
    backend = backend_factory(source)
    receipts, jobs = [], {}
    try:
        for episode_id, episode in wanted.items():
            job = dict(id=episode_id, video=str(paths[episode_id]), profile='human', instruction='',
                       source_sha256={'main': episode['source_sha256']})
            prepared = backend.prepare(job)
            if (prepared.metadata['input_sha256'] != job['source_sha256'] or
                    prepared.metadata['duration_s'] != episode['duration_s'] or
                    prepared.metadata['cameras']['main']['size'] != episode['source_size']):
                raise ValueError('Native source validation differs from the frozen selection')
            receipts.append(dict(episode_id=episode_id, native_metadata=prepared.metadata))
            jobs[episode_id] = {**job, 'video': str(paths[episode_id].relative_to(output))}
    finally:
        backend.close()
    (output / 'jobs.jsonl').write_text(''.join(canonical_json(job) + '\n' for job in jobs.values()))
    for block in selection['blocks']:
        (output / f"block-{block['block']}.jsonl").write_text(''.join(
            canonical_json(jobs[eid]) + '\n' for eid in block['episode_ids']))
    result = dict(schema_version='bir_phase2_preparation_v1', status='complete',
        selection_sha256=selection['selection_sha256'], selected_ids=list(jobs),
        selected_source_seconds=selection['source_seconds'], selected_video_count=len(jobs),
        new_model_calls=0, reencoded_videos=0, elapsed_s=time.monotonic() - started,
        video_bytes=sum(p.stat().st_size for p in paths.values()), receipts=receipts,
        boundary='Includes Parquet hashing, selected byte extraction, native video inspection, and manifest serialization.')
    atomic_json(output / 'preparation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=Path('manifests/phase2-native.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Source extraction requires a reserved CPU allocation')
    reject_restarted_execution()
    result = prepare(args.parquet, args.source, load_selection(args.selection), args.output_dir)
    print(canonical_json({key: result[key] for key in ('status', 'selected_video_count', 'elapsed_s')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
