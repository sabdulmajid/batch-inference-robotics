import importlib.util
import errno
from pathlib import Path
import tempfile
import json
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from bir.native import (NativeBackend, NativeModelLoadBlocked, NativeModelLoadError, PersistentSession, PreparedEpisode,
                        _json_audit, verify_source, UPSTREAM_COMMIT)

SOURCE = Path(__file__).resolve().parents[1] / 'third_party' / 'Flora'
HAS_SOURCE = (SOURCE / 'captioner/batch_utils/models.json').is_file()
HAS_AV = importlib.util.find_spec('av') is not None


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.sync_patch = patch('bir.native._synchronize_cuda')
        self.sync = self.sync_patch.start()
        self.addCleanup(self.sync_patch.stop)
        self.event_patch = patch('bir.native._runtime_event')
        self.event = self.event_patch.start()
        self.addCleanup(self.event_patch.stop)
        self.memory_patch = patch('bir.native._synchronized_memory_snapshot', return_value={'devices': [], 'synchronization_complete': True})
        self.memory = self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)

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
            self.assertEqual(self.sync.call_count, 2)
            self.assertEqual([call.args[0] for call in self.event.call_args_list], ['model_load_end', 'model_load_end'])
            self.assertTrue(all(call.kwargs['cold_load'] and call.kwargs['synchronization_complete']
                                for call in self.event.call_args_list))
            self.assertTrue(all(call.kwargs['elapsed_s'] >= 0 for call in self.event.call_args_list))
        session.close()
        self.assertIsNone(session.kind)

    def test_failed_residency_cannot_remain_cached(self):
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind): self.model = object(); return self.model, None
            def close(self): self.model = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.verify_residency', side_effect=RuntimeError('cpu offload')):
            with self.assertRaises(NativeModelLoadError) as caught: session.load('tempseg')
            self.assertEqual(caught.exception.kind, 'tempseg')
            self.assertEqual(caught.exception.original_error_type, 'RuntimeError')
            self.assertEqual(str(caught.exception), 'cpu offload')
            self.assertIs(type(caught.exception.__cause__), RuntimeError)
            with self.assertRaises(NativeModelLoadBlocked): session.load('tempseg')
        self.assertIsNone(session.kind)
        self.assertIsNone(session.native.model)
        self.assertEqual([call.args[0] for call in self.event.call_args_list if call.args[0].startswith('model_load')],
                         ['model_load_end', 'model_load_blocked'])
        self.assertEqual(session.load_timings['tempseg'][0]['status'], 'failed')
        self.assertEqual(len(session.load_timings['tempseg']), 1)

    def test_transient_load_failure_keeps_explicit_bounded_retry_available(self):
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}; self.calls = 0
            def load(self, kind):
                self.calls += 1
                if self.calls == 1: raise OSError(errno.EIO, 'temporary I/O fixture')
                self.model = object()
                return self.model, None
            def close(self): self.model = self.proc = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.verify_residency', return_value={}):
            with self.assertRaises(OSError): session.load('tempseg')
            session.load('tempseg')
        self.assertEqual(session.native.calls, 2)
        self.assertEqual([row['status'] for row in session.load_timings['tempseg']], ['failed', 'complete'])
        self.assertTrue(session.load_timings['tempseg'][0]['transient'])

    def test_native_stage_close_preserves_upstream_cleanup_and_records_both_snapshots(self):
        order = []
        class Session:
            def __init__(self): self.model = object(); self.proc = None; self.audit = {}
            def load(self, kind):
                self.close()
                order.append('load_new_model')
                self.model = object()
                return self.model, None
            def close(self):
                order.append('original_native_close')
                self.model = self.proc = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        session.kind = 'captioner'
        snapshots = iter([{'devices': [{'allocated_bytes': 100, 'reserved_bytes': 150}]},
                          {'devices': [{'allocated_bytes': 0, 'reserved_bytes': 50}]}])
        def snapshot():
            order.append('synchronized_snapshot')
            return next(snapshots)
        self.memory.side_effect = snapshot
        with patch('bir.native.verify_residency', return_value={}): session.load('tempseg')
        self.assertEqual(order, ['synchronized_snapshot', 'original_native_close', 'synchronized_snapshot', 'load_new_model'])
        close = next(call.kwargs for call in self.event.call_args_list if call.args[0] == 'model_close_end')
        self.assertEqual(close['kind'], 'captioner')
        self.assertEqual(close['before']['devices'][0]['allocated_bytes'], 100)
        self.assertEqual(close['after']['devices'][0]['allocated_bytes'], 0)
        self.assertEqual(close['after']['devices'][0]['reserved_bytes'], 50)

    def test_failed_placement_receipt_survives_typed_load_failure(self):
        from contextlib import contextmanager
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind): raise RuntimeError('weights exceed measured budget')
            def close(self): self.model = self.proc = None
        @contextmanager
        def placement(headroom, audit):
            audit.update(feasibility='failed', device_memory=[{'allocated_bytes': 10, 'reserved_bytes': 20, 'free_bytes': 30}])
            yield
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.captioner_placement', side_effect=placement):
            with self.assertRaises(NativeModelLoadError): session.load('captioner')
        self.assertEqual(session.load_timings['captioner'][0]['placement'], session.placement_audit['captioner'])
        self.assertEqual(session.placement_audit['captioner']['device_memory'][0]['reserved_bytes'], 20)

    def test_close_weak_reference_does_not_retain_the_model(self):
        class Model: pass
        class Session:
            def __init__(self): self.model = Model(); self.proc = None; self.audit = {}
            def close(self): self.model = self.proc = None
        for externally_retained in (False, True):
            session = PersistentSession(types.SimpleNamespace(Session=Session))
            external = session.native.model if externally_retained else None
            session.close()
            receipt = self.event.call_args.kwargs
            self.assertTrue(receipt['model_weakref_available'])
            self.assertEqual(receipt['model_alive_after_close'], externally_retained)
            del external

    def test_full_load_event_follows_residency_and_all_device_synchronization(self):
        order = []
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind):
                order.append('native_load_and_processor')
                self.model = object()
                return self.model, None
            def close(self): self.model = None
        self.sync.side_effect = lambda: order.append('synchronize_all_visible_devices')
        self.event.side_effect = lambda *args, **kwargs: order.append('timed_event')
        with patch('bir.native.verify_residency', side_effect=lambda _: order.append('verify_residency')):
            PersistentSession(types.SimpleNamespace(Session=Session)).load('tempseg')
        self.assertEqual(order, ['native_load_and_processor', 'verify_residency',
                                 'synchronize_all_visible_devices', 'timed_event'])


class AuditSerializationTests(unittest.TestCase):
    def setUp(self):
        sync = patch('bir.native._synchronize_cuda')
        sync.start()
        self.addCleanup(sync.stop)
        event = patch('bir.native._runtime_event')
        event.start()
        self.addCleanup(event.stop)
        memory = patch('bir.native._synchronized_memory_snapshot', return_value={'devices': [], 'synchronization_complete': True})
        memory.start()
        self.addCleanup(memory.stop)
    def test_loading_diagnostics_preserve_values_with_stable_order(self):
        raw = {'missing_keys': {'z', 'a'}, 'unexpected_keys': set(),
               'mismatched_keys': {('z', (4, 5), (5, 4)), ('a', (1,), (2,))},
               'error_msgs': [], 'nested': frozenset({'b', 'a'})}
        normalized = _json_audit(raw)
        self.assertEqual(normalized['missing_keys'], ['a', 'z'])
        self.assertEqual(normalized['unexpected_keys'], [])
        self.assertEqual(normalized['mismatched_keys'], [['a', [1], [2]], ['z', [4, 5], [5, 4]]])
        self.assertEqual(normalized['nested'], ['a', 'b'])
        self.assertEqual(json.loads(json.dumps(normalized, allow_nan=False)), normalized)
        self.assertIsInstance(raw['missing_keys'], set)
        for invalid in (object(), {1: 'invalid'}, float('nan')):
            with self.assertRaises((TypeError, ValueError)):
                _json_audit({'nested': invalid})

    @unittest.skipUnless(importlib.util.find_spec('transformers') is not None,
                         'requires optional pinned Transformers CPU environment')
    def test_actual_transformers_loading_info(self):
        from transformers.utils.loading_report import LoadStateDictInfo
        info = LoadStateDictInfo(set(), set(), set(), [], {}).to_dict()
        self.assertIsInstance(info['missing_keys'], set)
        self.assertEqual(_json_audit(info), {'missing_keys': [], 'unexpected_keys': [],
                                           'mismatched_keys': [], 'error_msgs': []})

    def test_unknown_audit_fails_during_load_and_clears_model(self):
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind):
                self.model = object()
                self.audit[kind] = {'invalid': object()}
                return self.model, None
            def close(self): self.model = None
        session = PersistentSession(types.SimpleNamespace(Session=Session))
        with patch('bir.native.verify_residency') as residency:
            with self.assertRaisesRegex(NativeModelLoadError, 'audit.tempseg.invalid') as caught:
                session.load('tempseg')
            self.assertIs(type(caught.exception.__cause__), TypeError)
            residency.assert_not_called()
        self.assertIsNone(session.kind)
        self.assertIsNone(session.native.model)

    def test_signed_checkpoint_store_resume_and_final_caption_diagnostics(self):
        """Exercise actual adapter/store boundaries; model calls are CPU stubs."""
        from contextlib import nullcontext
        from bir.manifest import load_manifest
        from bir.store import RunStore
        class Session:
            def __init__(self): self.model = self.proc = None; self.audit = {}
            def load(self, kind):
                self.model, self.proc = object(), object()
                info = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=set(), error_msgs=[])
                self.audit[kind] = ({'MODEL_LOADED.json': {'base_loading_info': info}}
                                    if kind == 'tempseg' else {'loading_info': info})
                return self.model, self.proc
            def close(self): self.model = self.proc = None
        backend = NativeBackend.__new__(NativeBackend)
        backend.frame_cache = None
        backend.progress = lambda *args: None
        backend.config = types.SimpleNamespace(PROFILE_POOLS={'human': 'human'}, MODELS={'test': 'pinned'})
        backend.session = PersistentSession(types.SimpleNamespace(Session=Session))
        def segment(videos, pool, session, instruction, progress):
            session.load('tempseg')
            return [{'start_s': 0.0, 'end_s': 1.0, 'metadata': {}}], [{'source_pts_us': [0, 999999]}]
        def caption(videos, segments, profile, session, instruction, progress):
            session.load('captioner')
            return [{'start_s': 0.0, 'end_s': 1.0, 'status': 'complete', 'caption': 'fixture'}], []
        backend.pipeline = types.SimpleNamespace(segment=segment, caption_single=caption)
        metadata = dict(input_sha256={'main': 'hash'}, profile='human', yam32=False,
                        instruction_sha256='instruction', duration_s=1.0)
        prepared = PreparedEpisode({}, {'main': object()}, 'human', metadata)
        cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'torch': cuda}), \
                patch('bir.native.verify_residency', return_value={}), \
                patch('bir.native.captioner_placement', return_value=nullcontext()):
            root = Path(directory)
            (root / 'video.mp4').write_bytes(b'identity-only fixture')
            (root / 'manifest.jsonl').write_text(json.dumps({'id': 'test', 'video': 'video.mp4'}))
            job = load_manifest(root / 'manifest.jsonl', contract={'test': 'audit boundary'})[0]
            store = RunStore(root / 'run.sqlite')
            try:
                with store.exclusive():
                    store.register(job)
                    attempt = store.begin(job.request_id, 'segment')
                    signed = backend.segment(prepared)
                    store.complete(job.request_id, 'segment', attempt, signed, 1.0)
                    restored = store.result(job.request_id, 'segment')
                    self.assertEqual(restored, signed)
                    annotation = backend.caption(prepared, restored)
                    caption_attempt = store.begin(job.request_id, 'caption')
                    store.complete(job.request_id, 'caption', caption_attempt, annotation, 1.0)
                    self.assertEqual(store.result(job.request_id, 'caption'), annotation)
                    verification = annotation['engine']['model_verification']
                    self.assertEqual(verification['tempseg']['MODEL_LOADED.json']['base_loading_info']['missing_keys'], [])
                    self.assertEqual(verification['captioner']['loading_info']['missing_keys'], [])
                    self.assertIsInstance(backend.session.native.audit['captioner']['loading_info']['missing_keys'], set)
                    restored['temporal_audit'][0]['source_pts_us'][1] -= 1
                    with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                        backend.caption(prepared, restored)
            finally:
                store.close()
                backend.close()


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
