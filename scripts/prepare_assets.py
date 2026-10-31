"""CPU allocation only. Download and verify the exact native assets once."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

# SHA256 values and byte lengths from the pinned Hub revision's LFS metadata.
# These pin full base weights, in addition to native Gated DeltaNet checks.
BASE_WEIGHTS = {
    'model.safetensors-00001-of-00004.safetensors':
        (5276436216, 'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e'),
    'model.safetensors-00002-of-00004.safetensors':
        (5335161512, '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330'),
    'model.safetensors-00003-of-00004.safetensors':
        (5368717440, '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec'),
    'model.safetensors-00004-of-00004.safetensors':
        (3325995712, 'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6'),
}
DATASET_REVISION = '9cfb7bb98d60a58319f4b3d7b41374eedfb7980c'


def verify_base_weights(path, files=BASE_WEIGHTS):
    """Stream hashes and reject changed or incomplete pinned checkpoint files."""
    path = Path(path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != set(files):
        raise RuntimeError('Base checkpoint shard map differs from the pinned revision')
    receipts = {}
    for name, (expected_bytes, expected_sha256) in files.items():
        file = path / name
        before = file.stat()
        if before.st_size != expected_bytes:
            raise RuntimeError('Base checkpoint file size differs from the pinned revision: ' + name)
        with file.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = file.stat()
        fingerprint = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if actual != expected_sha256 or fingerprint(before) != fingerprint(after):
            raise RuntimeError('Base checkpoint hash check failed: ' + name)
        receipts[name] = {'bytes': expected_bytes, 'sha256': actual}
    return receipts


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Asset preparation requires an authorized CPU allocation')
    if os.environ.get('SLURM_RESTART_COUNT', '0') != '0':
        raise RuntimeError('Reconcile restarted allocation time before preparing assets')
    root = Path(os.environ['BIR_PROJECT_ROOT']).resolve(strict=True)
    source = root / 'Flora'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from bir.native import verify_source
    from bir.cli import atomic_json
    provenance = verify_source(source)
    # Verify source and repository overrides before upstream imports/downloads.
    sys.path.insert(0, str(source / 'captioner'))
    from batch_utils.models import snapshot, validate_manifest
    from batch_utils.config import MODELS
    from huggingface_hub import snapshot_download, hf_hub_download

    for kind in ('tempseg', 'captioner'):
        path = snapshot(kind)
        validate_manifest(path)
        print(json.dumps({'event': 'assets_verified', 'kind': kind,
                          'revision': MODELS[kind]['revision']}), flush=True)
    base = MODELS['tempseg_base']
    base_path = snapshot_download(base['repo'], revision=base['revision'],
                                  allow_patterns=['*.safetensors', '*.json', '*.jinja'])
    base_receipts = verify_base_weights(base_path)
    print(json.dumps({'event': 'assets_verified', 'kind': 'tempseg_base',
                      'revision': base['revision'], 'files': len(base_receipts)}), flush=True)
    for filename in ('README.md', 'data/annotations.parquet', 'data/annotations.parquet.provenance.json'):
        hf_hub_download('macrodata/WGO-Bench', filename, repo_type='dataset',
                        revision=DATASET_REVISION, local_dir=root / 'datasets' / 'wgo')
    atomic_json(root / 'ops' / 'assets-complete.json',
                {'source': provenance['source_commit'], 'models': MODELS,
                 'base_weight_verification': base_receipts, 'dataset': DATASET_REVISION})


if __name__ == '__main__':
    main()
