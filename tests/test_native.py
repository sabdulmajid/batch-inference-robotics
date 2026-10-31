import importlib.util
from pathlib import Path
import tempfile
import json
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from bir.native import NativeBackend, PersistentSession, verify_source, UPSTREAM_COMMIT

SOURCE = Path(__file__).resolve().parents[1] / 'third_party' / 'Flora'
HAS_SOURCE = (SOURCE / 'captioner/batch_utils/models.json').is_file()
HAS_AV = importlib.util.find_spec('av') is not None


class SessionTests(unittest.TestCase):
    def test_model_lifetime_reused_and_stage_change_loads_once(self):
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}; self.loads = []
            def load(self, kind):
                self.loads.append(kind)
                self.model, self.proc = object(), object()
                return self.model, self.proc
            def close(self): self.model = self.proc = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.verify_residency', return_value={}), patch('bir.native.captioner_placement') as placement:
            first = session.load('tempseg')
            self.assertIs(first[0], session.load('tempseg')[0])
            session.load('captioner')
            session.load('captioner')
            self.assertEqual(session.native.loads, ['tempseg', 'captioner'])
            placement.assert_called_once()
        session.close()
        self.assertIsNone(session.kind)

    def test_failed_residency_cannot_remain_cached(self):
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind): self.model = object(); return self.model, None
            def close(self): self.model = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.verify_residency', side_effect=RuntimeError('cpu offload')):
            with self.assertRaises(RuntimeError): session.load('tempseg')
        self.assertIsNone(session.kind)
        self.assertIsNone(session.native.model)


class ProbeGuardTests(unittest.TestCase):
    def test_restarted_allocation_exits_before_importing_gpu_framework(self):
        script_root = Path(__file__).resolve().parents[1] / 'scripts'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = dict(os.environ, SLURM_RESTART_COUNT='1', SLURM_JOB_ID='test')
            smoke = subprocess.run([sys.executable, str(script_root / 'framework_smoke.py'),
                                    '--output', str(root / 'smoke.json')],
                                   env=environment, capture_output=True, text=True)
            self.assertEqual(smoke.returncode, 1)
            report = json.loads((root / 'smoke.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('budget reconciliation', report['error'])
            native = subprocess.run([sys.executable, str(script_root / 'native_probe.py'),
                                     '--source-root', temporary, '--manifest', str(root / 'jobs.jsonl'),
                                     '--state-dir', str(root / 'state'), '--output-dir', str(root / 'native')],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(native.returncode, 1)
            report = json.loads((root / 'native/probe-summary.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('budget reconciliation', report['error'])


@unittest.skipUnless(HAS_SOURCE and HAS_AV, 'requires pinned FLORA checkout and optional CPU video dependencies')
class NativeCpuIntegrationTests(unittest.TestCase):
    def test_real_video_validation_decode_cache_and_native_frame_plan(self):
        import av
        import numpy as np
        from bir.frame_cache import DecodedFrameCache
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / 'synthetic.mp4'
            with av.open(str(video_path), 'w') as container:
                stream = container.add_stream('mpeg4', rate=8)
                stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
                for i in range(16):
                    pixels = np.full((48, 64, 3), (i * 13) % 256, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            backend = NativeBackend(SOURCE)
            prepared = backend.prepare({'id': 'synthetic', 'video': str(video_path), 'profile': 'human'})
            self.assertEqual(prepared.metadata['cameras']['main']['frames'], 16)
            self.assertEqual(prepared.metadata['duration_s'], 2.0)
            original = backend.media.decode_selected(prepared.videos['main'], {0, 7, 15})
            cache = DecodedFrameCache(1024 * 1024)
            for _ in range(2):
                cached = cache.decode(prepared.videos['main'], {0, 7, 15}, decoder=backend.media.decode_selected)
                self.assertEqual({i: image.tobytes() for i, image in original.items()},
                                 {i: image.tobytes() for i, image in cached.items()})
            plans, dispositions = backend.pipeline.single_plans(prepared.videos,
                [{'start_s': 0.0, 'end_s': 2.0, 'metadata': {}}], prepared.profile)
            self.assertEqual(len(plans), 1)
            self.assertEqual(dispositions[0]['status'], 'caption_input')
            self.assertTrue(all(0 <= i < 16 for b in plans[0]['blocks'] for i in b['frame_indices']))
            self.assertEqual(backend.session.load_counts, {'tempseg': 0, 'captioner': 0})
            backend.close()

    def test_pinned_source_and_repository_replacement_guard(self):
        self.assertEqual(verify_source(SOURCE)['source_commit'], UPSTREAM_COMMIT)
        with patch.dict('os.environ', {'FLORA_CAPTIONER_REPO': 'unrelated/model'}):
            with self.assertRaisesRegex(RuntimeError, 'replacement model'):
                verify_source(SOURCE)


if __name__ == '__main__':
    unittest.main()
