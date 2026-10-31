"""Stable runtime identity for the explicitly measured human caption collector.

Run directories, episodes, CLI presentation, and evidence file hashes are not
part of this identity. Native implementation and model contracts are pinned.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

from .manifest import canonical_json, file_sha256
from .native import MODEL_REVISIONS, UPSTREAM_COMMIT


def runtime_source_hashes():
    root = Path(__file__).parent
    modules = ('batching.py', 'batching_contract.py', 'caption_batching.py',
               'native.py', 'placement.py', 'frame_cache.py')
    return {name: file_sha256(root / name) for name in modules}


def native_batching_contract(*, model_lock=Path('configs/native-models.json'),
                             captioner_headroom_gib=4.0, max_prepared_bytes=2 * 1024**3):
    if not math.isfinite(captioner_headroom_gib) or captioner_headroom_gib <= 0:
        raise ValueError('Native placement headroom must be positive and finite')
    if type(max_prepared_bytes) is not int or not 1 <= max_prepared_bytes <= 2 * 1024**3:
        raise ValueError('The measured collector tensor bound must be at most2GiB')
    lock = json.loads(Path(model_lock).read_text())
    if lock.get('source', {}).get('revision') != UPSTREAM_COMMIT or any(
            lock.get(kind, {}).get('revision') != revision for kind, revision in MODEL_REVISIONS.items()):
        raise ValueError('Batching requires the pinned native source and model revisions')
    if lock.get('microbatch_size') != 1:
        raise ValueError('Keep the native reference model lock at batch size one')
    value = dict(schema_version='bir_native_batching_contract_v1',
        native_model_lock=lock,
        runtime_source_sha256=runtime_source_hashes(),
        captioner_headroom_gib=captioner_headroom_gib,
        profile='human_single_view', max_pending_requests=2,
        batching_boundary='one_episode',
        max_prepared_bytes=max_prepared_bytes, batch_size=2,
        native_preparation='unchanged pinned FLORA source; exact native tensors and ordered video grids',
        generation=dict(do_sample=False, num_beams=1, max_new_tokens=512, use_cache=True,
                        max_time=300, eos_token_id='verified native tokenizer', pad_token_id='verified native tokenizer'))
    # Validate canonical JSON here. The caller hashes this object independently
    # of any admission or run receipt, avoiding a circular evidence identity.
    canonical_json(value)
    return value


def native_batching_contract_id(contract):
    return hashlib.sha256(canonical_json(contract).encode()).hexdigest()
