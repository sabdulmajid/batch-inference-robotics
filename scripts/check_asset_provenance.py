#!/usr/bin/env python3
"""Verify the pinned Captioner manifest trust anchors without loading a model.

Evidence comes from the exact Hugging Face revision below. The upstream
source_manifest_identity is a source-model identity carried in provenance. It
is NOT the SHA256 of SOURCE_MANIFEST.sha256. Pin and check both separately.

https://huggingface.co/nvidia/Flora-Captioner/tree/fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6

Default execution reads only two small metadata files. --verify-payloads also
streams every manifest-listed file and requires a scheduled allocation. The
normal native loader already performs that payload check; it lacks this
independent anchor check. This script does not modify any upstream file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


CAPTIONER_PINS = {
    'repository': 'nvidia/Flora-Captioner',
    'revision': 'fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6',
    'manifest_sha256': '75241af6f42d26f955f207f8cfb98ba6e6d05c303f74a7c85bde30a6cf997ad8',
    'provenance_sha256': 'b1979e729def7828c8d7600692ba17a3913bdac6ddd152c22bfcabd3f0ad924e',
    'source_model_identity_sha256': '2ab6cafd286df7f446bfb84fbc4af7c54773d5e890dd277ac5a0453d0d4fb778',
}


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _metadata(path):
    before = path.stat()
    if not 0 < before.st_size <= 65536:
        raise ValueError('Expected a small nonempty provenance file')
    with path.open('rb') as stream:
        content = stream.read(65537)
    if len(content) > 65536 or _stamp(before) != _stamp(path.stat()):
        raise ValueError('Provenance metadata changed while it was read')
    return content


def _manifest_entries(content):
    entries = {}
    for line in content.decode('utf-8').splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch('[0-9a-f]{64}', parts[0]):
            raise ValueError('Invalid SHA256 manifest line')
        name = parts[1].lstrip('*')
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Manifest entry escapes the snapshot')
        if name in entries:
            raise ValueError('Duplicate manifest entry')
        entries[name] = parts[0]
    if not entries:
        raise ValueError('Empty source manifest')
    return entries


def check_captioner_provenance(snapshot, *, verify_payloads=False, pins=None):
    """Return public checksums and verification scope; never return local paths.

    An explicit ``pins`` mapping supports a separately reviewed revision or a
    small test fixture. Production callers should use the fixed default pins.
    ``verify_payloads=False`` does not claim that weight bytes were verified.
    """
    pins = CAPTIONER_PINS if pins is None else pins
    root = Path(snapshot).resolve(strict=True)
    manifest = _metadata(root / 'SOURCE_MANIFEST.sha256')
    provenance_bytes = _metadata(root / 'MODEL_PROVENANCE.json')
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    if manifest_hash != pins['manifest_sha256']:
        raise ValueError('Source manifest differs from the independently pinned manifest bytes')
    if provenance_hash != pins['provenance_sha256']:
        raise ValueError('Model provenance differs from the independently pinned provenance bytes')
    entries = _manifest_entries(manifest)
    if entries.get('MODEL_PROVENANCE.json') != provenance_hash:
        raise ValueError('Manifest does not bind the verified model provenance')
    provenance = json.loads(provenance_bytes)
    if provenance.get('source_model_identity_sha256') != pins['source_model_identity_sha256']:
        raise ValueError('Provenance source-model identity differs from the pinned source identity')
    if provenance.get('source_weights_modified') is not False:
        raise ValueError('Provenance does not identify unchanged source weights')
    verified_files = payload_bytes = weight_bytes = 0
    if verify_payloads:
        # HF snapshot symlinks may target the sibling blobs directory. A local
        # directory snapshot must keep its files within that directory.
        allowed_root = root.parent.parent if root.parent.name == 'snapshots' else root
        for name, expected in entries.items():
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(allowed_root):
                raise ValueError('Manifest payload resolves outside authorized snapshot storage')
            before = path.stat()
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected or _stamp(before) != _stamp(path.stat()):
                raise ValueError('Manifest payload hash mismatch or file changed during verification')
            verified_files += 1
            payload_bytes += before.st_size
            if name.endswith('.safetensors'):
                weight_bytes += before.st_size
    return {
        'schema_version': 'bir_asset_provenance_v1',
        'status': 'metadata_and_payloads_verified' if verify_payloads else 'metadata_trust_anchors_verified',
        'repository': pins['repository'], 'revision': pins['revision'],
        'manifest_sha256': manifest_hash, 'provenance_sha256': provenance_hash,
        'source_model_identity_sha256': provenance['source_model_identity_sha256'],
        'source_identity_is_raw_manifest_hash': False,
        'source_weights_modified': False, 'manifest_entry_count': len(entries),
        'payload_files_verified': verified_files, 'payload_bytes_read': payload_bytes,
        'model_weight_bytes_read': weight_bytes,
        'weight_integrity_scope': 'all manifest entries streamed' if verify_payloads else 'not_checked_by_this_metadata_only_call',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--verify-payloads', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_payloads and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_RESTART_COUNT', '0') != '0'):
            raise ValueError('Payload verification requires a budget-reserved allocation without an unreconciled restart')
        result = check_captioner_provenance(args.snapshot, verify_payloads=args.verify_payloads)
        code = 0
    except Exception as error:
        result = {'schema_version': 'bir_asset_provenance_v1', 'status': 'failed',
                  'error_type': type(error).__name__,
                  'error': str(error) if isinstance(error, ValueError) else 'Cannot read or verify asset provenance'}
        code = 1
    encoded = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return code


if __name__ == '__main__':
    sys.exit(main())
