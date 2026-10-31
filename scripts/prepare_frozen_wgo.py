#!/usr/bin/env python3
"""Extract only exact frozen WGO episode IDs and validate their original bytes.

Run in a CPU allocation. This checks the pinned Parquet checksum, reads metadata,
and decodes only the selected videos. It does not re-probe the full corpus.
Optional reuse directories supply already extracted videos by content hash.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bir.cli import atomic_json
from bir.manifest import canonical_json, file_sha256
from bir.native import NativeBackend
from bir.resource_ledger import reject_restarted_execution
from evaluate_adaptive import load_freeze, validate_references
from prepare_wgo import WGO, _record, embedded_video


def prepare_frozen(parquet_path, source, freeze, output_dir, split, *, reuse_dirs=(),
                   pin=WGO, backend_factory=NativeBackend):
    import pyarrow.parquet as pq
    if split not in {'development', 'final'}:
        raise ValueError('Choose development or final')
    if freeze['dataset'] != pin.__dict__:
        raise ValueError('Frozen dataset differs from the pinned original Parquet')
    parquet_path = Path(parquet_path).resolve(strict=True)
    if file_sha256(parquet_path) != pin.parquet_sha256:
        raise ValueError('Parquet checksum differs from the frozen dataset')
    episodes = {e['episode_id']: e for e in freeze['episodes'] if e['split'] == split}
    if not episodes:
        raise ValueError('Frozen split contains no episodes')
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh extraction directory')
    output.mkdir(parents=True, exist_ok=True)
    videos = output / 'videos'
    videos.mkdir()
    parquet = pq.ParquetFile(parquet_path)
    rows, index = {}, 0
    for batch in parquet.iter_batches(batch_size=32, columns=['id', 'instruction', 'segments', 'metadata'], use_threads=False):
        for value in batch.to_pylist():
            if value['id'] in episodes:
                if value['id'] in rows:
                    raise ValueError('Duplicate frozen episode in Parquet')
                row = _record(value, index)
                row['source_sha256'] = episodes[value['id']]['source_sha256']
                rows[value['id']] = row
            index += 1
    if set(rows) != set(episodes):
        raise ValueError('Parquet does not contain every frozen episode')
    validate_references(freeze, list(rows.values()), split)
    paths, reused = {}, []
    for episode_id, episode in episodes.items():
        expected = episode['source_sha256']
        for reuse_dir in reuse_dirs:
            candidate = Path(reuse_dir) / 'videos' / (expected + '.mp4')
            if candidate.is_file():
                if file_sha256(candidate) != expected:
                    raise ValueError('Reusable original video has a different source checksum')
                destination = videos / candidate.name
                try:
                    os.link(candidate, destination)
                except OSError:
                    shutil.copyfile(candidate, destination)
                paths[episode_id] = destination
                reused.append(episode_id)
                break
    needed = {row['row_index']: episode_id for episode_id, row in rows.items() if episode_id not in paths}
    index = 0
    if needed:
        for batch in parquet.iter_batches(batch_size=1, columns=['video'], use_threads=False):
            if index in needed:
                episode_id = needed[index]
                payload = embedded_video(batch.to_pylist()[0]['video'])
                actual = hashlib.sha256(payload).hexdigest()
                if actual != episodes[episode_id]['source_sha256']:
                    raise ValueError('Selected embedded video differs from its frozen source hash')
                path = videos / (actual + '.mp4')
                path.write_bytes(payload)
                paths[episode_id] = path
            if index >= max(needed):
                break
            index += 1
    backend = backend_factory(source)
    jobs, references, receipts = [], [], []
    try:
        for episode_id, episode in episodes.items():
            path, row = paths[episode_id], rows[episode_id]
            job = dict(id=episode_id, video=str(path), profile='human', instruction=row['instruction'],
                       source_sha256={'main': episode['source_sha256']})
            prepared = backend.prepare(job)
            if prepared.metadata['input_sha256'] != job['source_sha256'] or prepared.metadata['duration_s'] != episode['duration_s']:
                raise ValueError('Native selected-source validation differs from frozen evidence')
            jobs.append({**job, 'video': str(path.relative_to(output))})
            references.append({'id': episode_id, 'source_sha256': episode['source_sha256'],
                'instruction': row['instruction'], 'references': row['references'],
                'duration_s': episode['duration_s'], 'group_id': episode['original_group_id']})
            receipts.append({'episode_id': episode_id, 'source_sha256': episode['source_sha256'],
                             'native_metadata': prepared.metadata, 'reference_count': len(row['references'])})
    finally:
        backend.close()
    (output / 'jobs.jsonl').write_text(''.join(canonical_json(row) + '\n' for row in jobs))
    atomic_json(output / 'references.json', references)
    result = {'schema_version': 'bir_frozen_wgo_preparation_v1', 'status': 'selected_native_inputs_validated',
        'freeze_sha256': freeze['freeze_sha256'], 'split': split, 'selected_ids': list(episodes),
        'new_model_calls': 0, 'reencoded_videos': 0, 'reused_video_ids': reused,
        'selected_video_count': len(receipts), 'unselected_videos_decoded': 0, 'receipts': receipts}
    atomic_json(output / 'preparation.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'final'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-dir', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Frozen video extraction requires a reserved CPU allocation')
    reject_restarted_execution()
    result = prepare_frozen(args.parquet, args.source, load_freeze(args.freeze), args.output_dir,
                            args.split, reuse_dirs=args.reuse_dir)
    print(canonical_json({key: result[key] for key in ['status', 'split', 'selected_ids', 'reencoded_videos', 'unselected_videos_decoded']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
