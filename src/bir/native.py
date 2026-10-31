"""Persistent stage adapter around a pinned, unmodified NVIDIA FLORA tree.

Importing this module does not import torch, av, or FLORA. The source and
model files remain in private external storage. Native planning, decoding,
prompts, generation, parsing, and reconstruction stay upstream-owned.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import os

from .placement import captioner_placement, verify_residency

UPSTREAM_COMMIT = '129a2ada74c526ee4692f3d1b111bea6ca19d5b7'
MODEL_REVISIONS = {
    'tempseg': '114a1076f70eb53f4a58ac045505db3087e79347',
    'captioner': 'fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6',
    'tempseg_base': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
}
_NATIVE_LOCK = threading.RLock()


def verify_source(source_root):
    """Require the exact Git tree, including unchanged tracked files."""
    root = Path(source_root).expanduser().resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != UPSTREAM_COMMIT:
        raise RuntimeError('FLORA source revision does not match the pinned native contract')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('FLORA source has changed tracked files')
    specs = json.loads((root / 'captioner/batch_utils/models.json').read_text())
    for kind, revision in MODEL_REVISIONS.items():
        if specs[kind]['revision'] != revision:
            raise RuntimeError('FLORA model revision does not match the pinned native contract')
        override = os.environ.get(specs[kind].get('repo_env', ''))
        if override and override != specs[kind]['repo']:
            raise RuntimeError('Native mode rejects a replacement model repository')
    return {'source_commit': UPSTREAM_COMMIT, 'model_revisions': specs}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _json_audit(value, path='audit'):
    """Copy loader diagnostics to strict JSON without changing upstream data.

    Transformers 5.8.1 returns sets in ``output_loading_info``. Their order
    must be stable before a checkpoint is signed. Unknown diagnostic types
    fail at model load, rather than after an expensive generation call.
    """
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite native diagnostic at {path}')
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError(f'Non-string native diagnostic key at {path}')
        return {key: _json_audit(item, f'{path}.{key}') for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_audit(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_json_audit(item, f'{path}{{item}}') for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True,
                      separators=(',', ':'), allow_nan=False))
    raise TypeError(f'Unsupported native diagnostic type {type(value).__name__} at {path}')


@dataclass(frozen=True)
class PreparedEpisode:
    job: dict
    videos: dict
    profile: str
    metadata: dict


class PersistentSession:
    """Reuse exactly one upstream model until the engine changes stages."""
    def __init__(self, models, *, captioner_headroom_gib=4.0):
        self.native = models.Session()
        self.kind = None
        self.headroom = captioner_headroom_gib
        self.placement_audit = {}
        self.residency_audit = {}
        self.load_counts = {'tempseg': 0, 'captioner': 0}

    @property
    def audit(self):
        return _json_audit(self.native.audit)

    def load(self, kind):
        if kind not in self.load_counts:
            raise ValueError('Unknown native model stage')
        if self.kind == kind and self.native.model is not None:
            return self.native.model, self.native.proc
        self.kind = None
        placement = {}
        context = captioner_placement(self.headroom, placement) if kind == 'captioner' else nullcontext()
        try:
            with context:
                result = self.native.load(kind)
            # Validate all diagnostics before upstream starts model inference.
            self.audit
            self.residency_audit[kind] = verify_residency(result[0])
            if placement:
                self.placement_audit[kind] = placement
            self.load_counts[kind] += 1
            self.kind = kind
            return result
        except BaseException:
            self.native.close()
            raise

    def close(self):
        self.kind = None
        if self.native.model is not None or self.native.proc is not None:
            self.native.close()


class NativeBackend:
    """Native batch-one adapter. Safe for one engine per process.

    CPU video probing may overlap with inference. Inference calls and the
    temporary decode hook use a global lock because upstream modules have
    process-wide state. This adapter does not alter CUDA_VISIBLE_DEVICES.
    """
    def __init__(self, source_root, *, captioner_headroom_gib=4.0,
                 decoded_frame_cache=None, frame_cache_bytes=0, progress=None):
        self.source_root = Path(source_root).expanduser().resolve(strict=True)
        self.provenance = verify_source(self.source_root)
        if not math.isfinite(captioner_headroom_gib) or captioner_headroom_gib <= 0:
            raise ValueError('Captioner headroom must be positive and finite')
        with _NATIVE_LOCK:
            package_root = self.source_root / 'captioner'
            loaded = sys.modules.get('batch_utils')
            if loaded is not None and Path(loaded.__file__).resolve().parent != package_root / 'batch_utils':
                raise RuntimeError('A different batch_utils package is already loaded')
            sys.path.insert(0, str(package_root))
            self.api = importlib.import_module('batch_utils.api')
            self.pipeline = importlib.import_module('batch_utils.pipeline')
            self.models = importlib.import_module('batch_utils.models')
            self.media = importlib.import_module('batch_utils.media')
            self.config = importlib.import_module('batch_utils.config')
        self.session = PersistentSession(self.models, captioner_headroom_gib=captioner_headroom_gib)
        if decoded_frame_cache is None and frame_cache_bytes:
            from .frame_cache import DecodedFrameCache
            decoded_frame_cache = DecodedFrameCache(frame_cache_bytes)
        self.frame_cache = decoded_frame_cache
        self.progress = progress or (lambda *args, **kwargs: None)
        self._decode = self.media.decode_selected

    def prepare(self, job):
        """Run the upstream CPU validation path before any model call."""
        row = dict(job)
        values = dict(profile=row.get('profile', 'human'), left_wrist=row.get('left_wrist'),
                      right_wrist=row.get('right_wrist'), yam32=row.get('yam32', False),
                      instruction=row.get('instruction', ''))
        videos, profile = self.api.DataEnginePipeline._inputs(row['video'], **values)
        metadata = {
            'status': 'inputs_validated', 'profile': profile, 'yam32': values['yam32'],
            'duration_s': videos['main'].duration_s,
            'input_sha256': {role: v.sha256 for role, v in videos.items()},
            'instruction_sha256': hashlib.sha256(values['instruction'].encode()).hexdigest(),
            'cameras': {role: {'sha256': v.sha256, 'source_fps': str(v.rate),
                              'frames': len(v.pts_us), 'size': list(v.size),
                              'constant_frame_rate': v.cfr, 'duration_s': v.duration_s}
                        for role, v in videos.items()},
        }
        return PreparedEpisode(row, videos, profile, metadata)

    @contextmanager
    def _execution(self):
        with _NATIVE_LOCK:
            if self.frame_cache is None:
                yield
            else:
                def decode(video, wanted, canonical=False):
                    return self.frame_cache.decode(video, wanted, canonical=canonical, decoder=self._decode)
                with patch.object(self.media, 'decode_selected', side_effect=decode):
                    yield

    def _binding(self, prepared):
        return {key: prepared.metadata[key] for key in
                ('input_sha256', 'profile', 'yam32', 'instruction_sha256', 'duration_s')}

    def segment(self, prepared):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for native segmentation')
        pool = 'yam32' if prepared.job.get('yam32', False) else self.config.PROFILE_POOLS[prepared.profile]
        start = time.monotonic()
        with self._execution():
            segments, audit = self.pipeline.segment(prepared.videos, pool, self.session,
                                                     prepared.job.get('instruction', ''), self.progress)
        result = {**self._binding(prepared), 'schema_version': 'bir_segmentation_v1',
                  'segments': segments, 'temporal_audit': audit, 'elapsed_s': time.monotonic() - start,
                  'source_commit': UPSTREAM_COMMIT, 'model_revisions': self.config.MODELS,
                  'engine': {'model_verification': dict(self.session.audit),
                             'residency': dict(self.session.residency_audit),
                             'model_load_counts': dict(self.session.load_counts)}}
        result['stage_digest'] = _json_digest(result)
        return result

    def caption(self, prepared, segmentation):
        expected_digest = segmentation.get('stage_digest')
        stage_body = {k: v for k, v in segmentation.items() if k != 'stage_digest'}
        if expected_digest != _json_digest(stage_body):
            raise RuntimeError('Segmentation checkpoint digest mismatch')
        if any(segmentation.get(key) != value for key, value in self._binding(prepared).items()):
            raise RuntimeError('Segmentation checkpoint belongs to a different input/configuration')
        if segmentation.get('source_commit') != UPSTREAM_COMMIT or segmentation.get('model_revisions') != self.config.MODELS:
            raise RuntimeError('Segmentation checkpoint native contract changed')
        segments = segmentation['segments']
        started = time.monotonic()
        with self._execution():
            if len(prepared.videos) == 3:
                captions, audit = self.pipeline.caption_three(prepared.videos, segments, self.session,
                    prepared.job.get('instruction', ''), self.progress, profile=prepared.profile)
            else:
                captions, audit = self.pipeline.caption_single(prepared.videos, segments, prepared.profile,
                    self.session, prepared.job.get('instruction', ''), self.progress)
        for row, segment in zip(captions, segments, strict=True):
            row['segmentation_provenance'] = segment['metadata'].get('segmentation_provenance', 'fine')
            if not (math.isfinite(row['start_s']) and math.isfinite(row['end_s']) and
                    0 <= row['start_s'] < row['end_s'] <= prepared.metadata['duration_s'] + 0.001):
                raise RuntimeError('Native pipeline returned an invalid caption interval')
        caption_elapsed = time.monotonic() - started
        return {
            'schema_version': 'flora_demo_annotation_v1', 'duration_s': prepared.metadata['duration_s'],
            'segments': captions, 'elapsed_s': segmentation['elapsed_s'] + caption_elapsed,
            'source_clock': 'seconds relative to the first native source presentation timestamp',
            'camera_roles': list(prepared.videos), 'profile': prepared.profile,
            'model_revisions': self.config.MODELS, 'input_sha256': prepared.metadata['input_sha256'],
            'pipeline': {'temporal_windows': segmentation['temporal_audit'], 'caption_inputs': audit},
            'engine': {'source_commit': UPSTREAM_COMMIT, 'mode': 'placement_adapted_native',
                       'request_batch_size': 1, 'stage_elapsed_s': {'tempseg': segmentation['elapsed_s'],
                       'captioner': caption_elapsed}, 'model_load_counts': dict(self.session.load_counts),
                       'placement': dict(self.session.placement_audit),
                       'residency': {**segmentation.get('engine', {}).get('residency', {}),
                                     **self.session.residency_audit},
                       'model_verification': {**segmentation.get('engine', {}).get('model_verification', {}),
                                              **self.session.audit}},
        }

    def close(self):
        with _NATIVE_LOCK:
            self.session.close()
