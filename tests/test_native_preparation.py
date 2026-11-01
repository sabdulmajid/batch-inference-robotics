import importlib.util
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import patch

from bir.native import NativeBackend, PreparedEpisode
from bir.native_preparation import PreparedCaption, caption_single_direct, serialized_processor_access
from bir.native_profile import NativeProfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/Flora'
ASSETS = ROOT / '.local/adaptive-processor'
HAS_PROCESSOR = (importlib.util.find_spec('transformers') is not None and
                 importlib.util.find_spec('av') is not None and
                 (ASSETS / 'tokenizer.json').is_file() and SOURCE.is_dir())


class DirectSchedulingTests(unittest.TestCase):
    def backend(self):
        plans = [dict(event_id=f'event_{i:04d}', start_s=float(i), end_s=float(i + 1),
                      blocks=[{'frame_indices': [i]}]) for i in range(3)]
        dispositions = [dict(p, status='caption_input') for p in plans]
        image = types.SimpleNamespace(width=2, height=2, getbands=lambda: ['R', 'G', 'B'])
        pipeline = types.SimpleNamespace(single_plans=lambda *a: (plans, dispositions),
            frames_for=lambda videos, wanted: {i: image for i in wanted})
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=lambda *a, **k: 'text',
                                          convert_tokens_to_ids=lambda *a: 7),
                                          batch_decode=lambda *a, **k: ['text'])
        backend = types.SimpleNamespace(pipeline=pipeline, native_profile=NativeProfile(),
            session=types.SimpleNamespace(load=lambda _: (object(), processor)), progress=lambda *a, **k: None)
        prepared = PreparedEpisode({'id': 'episode'}, {'main': types.SimpleNamespace(size=(2, 2))}, 'Human', {})
        return backend, prepared

    def test_one_ready_request_overlaps_generation_and_routes_exact_order(self):
        backend, prepared = self.backend()
        second_started = threading.Event()
        first_generating = threading.Event()
        captures, generations = [], []
        def prepare(pipeline, plan, *args, **kwargs):
            index = int(plan['event_id'][-4:])
            with kwargs['processor_lock']:
                captures.append(index)
                if index == 1:
                    second_started.set()
                    self.assertTrue(first_generating.wait(2))
            return PreparedCaption({'identity': index}, f'prompt {index}', 10)
        def generate(model, proc, inputs, budget):
            index = inputs['identity']
            self.assertEqual(budget, 512)
            if index == 0:
                first_generating.set()
                self.assertTrue(second_started.wait(2))
                self.assertNotIn(2, captures)  # one pending request, not a corpus queue
            generations.append(index)
            return f'label {index}'
        backend.pipeline.generate = generate
        with patch('bir.native_preparation.direct_processor_inputs', side_effect=prepare):
            rows, audit = caption_single_direct(backend, prepared, [], prefetch=1)
        self.assertEqual(generations, [0, 1, 2])
        self.assertEqual([r['caption'] for r in rows], ['label 0', 'label 1', 'label 2'])
        self.assertEqual([r['event_id'] for r in audit], ['event_0000', 'event_0001', 'event_0002'])

    def test_generation_failure_does_not_retry_or_start_later_generations(self):
        backend, prepared = self.backend()
        calls = []
        def generate(*args):
            calls.append(1)
            raise RuntimeError('fixture model failure')
        backend.pipeline.generate = generate
        with patch('bir.native_preparation.direct_processor_inputs', return_value=PreparedCaption({}, 'prompt', 1)):
            with self.assertRaisesRegex(RuntimeError, 'fixture model failure'):
                caption_single_direct(backend, prepared, [], prefetch=1)
        self.assertEqual(calls, [1])

    def test_profile_and_payload_guards_prevent_unbounded_work(self):
        backend, prepared = self.backend()
        backend.pipeline.generate = lambda *a: self.fail('guard must prevent inference')
        with self.assertRaisesRegex(ValueError, 'payload exceeds'):
            caption_single_direct(backend, prepared, [], max_prepared_bytes=1)
        invalid = PreparedEpisode({}, prepared.videos, 'YAM', {})
        backend.session.load = lambda *a: self.fail('profile guard must precede loading')
        with self.assertRaisesRegex(ValueError, 'Human single-view'):
            caption_single_direct(backend, invalid, [])

    def test_shared_tokenizer_lock_blocks_encoding_during_decode_and_restores_methods(self):
        decoding, release, encoding, attempted = (threading.Event() for _ in range(4))
        def decode(ids, **kwargs):
            if ids == 'error':
                raise RuntimeError('native decode failed')
            if ids == 'wait':
                decoding.set()
                if not release.wait(2):
                    raise TimeoutError('test decoder was not released')
            return str(ids)
        converted, convert_attempted = threading.Event(), threading.Event()
        def convert(token):
            converted.set()
            return 7
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=decode, convert_tokens_to_ids=convert))
        processor.batch_decode = lambda rows, **kw: [processor.tokenizer.decode(row, **kw) for row in rows]
        original_batch = processor.batch_decode
        with serialized_processor_access(processor) as lock, ThreadPoolExecutor(max_workers=2) as workers:
            decoded = workers.submit(processor.tokenizer.decode, 'wait')
            try:
                self.assertTrue(decoding.wait(2))
                def encode():
                    attempted.set()
                    with lock:
                        encoding.set()
                encoded = workers.submit(encode)
                self.assertTrue(attempted.wait(2))
                self.assertFalse(encoding.wait(.05))
            finally:
                release.set()
            self.assertEqual(decoded.result(2), 'wait')
            encoded.result(2)
            self.assertTrue(encoding.is_set())
            with lock:
                def lookup_eos():
                    convert_attempted.set()
                    return processor.tokenizer.convert_tokens_to_ids('EOS')
                converted_future = workers.submit(lookup_eos)
                self.assertTrue(convert_attempted.wait(2))
                self.assertFalse(converted.wait(.05))
            self.assertEqual(converted_future.result(2), 7)
            # Batch decoding re-enters the same lock through tokenizer.decode.
            self.assertEqual(processor.batch_decode(['a', 'b']), ['a', 'b'])
            with self.assertRaisesRegex(RuntimeError, 'native decode failed'):
                processor.tokenizer.decode('error')
        self.assertIs(processor.tokenizer.decode, decode)
        self.assertIs(processor.tokenizer.convert_tokens_to_ids, convert)
        self.assertIs(processor.batch_decode, original_batch)


@unittest.skipUnless(HAS_PROCESSOR, 'requires pinned local CPU processor and native source')
class ActualProcessorEquivalenceTests(unittest.TestCase):
    def test_native_png_and_direct_frames_have_identical_tensors_and_audits(self):
        import numpy as np
        from PIL import Image
        import torch
        torch.set_num_threads(4)
        backend = NativeBackend(SOURCE)
        processor = backend.models.processor('captioner', ASSETS)
        model = object()
        backend.session = types.SimpleNamespace(load=lambda _: (model, processor))
        video = types.SimpleNamespace(rate=Fraction(30), pts_us=tuple(Fraction(i * 1000000, 30) for i in range(150)),
            duration_us=5000000, duration_s=5., size=(701, 511))
        segments = [dict(start_s=0., end_s=1., metadata={}), dict(start_s=2., end_s=3., metadata={})]
        pixels = np.indices((511, 701), dtype=np.uint32)
        def frames(videos, wanted, **kwargs):
            return {i: Image.fromarray(np.stack(((pixels[0] + i * 3) % 256,
                (pixels[1] + i * 7) % 256, (pixels[0] + pixels[1] + i * 11) % 256), axis=-1).astype('uint8')) for i in wanted}
        for instruction in ('', 'Move the object to the left.'):
            prepared = PreparedEpisode({'id': 'synthetic', 'instruction': instruction}, {'main': video}, 'Human', {})
            captures = []
            def generate(m, proc, inputs, budget):
                self.assertIs(m, model)
                self.assertIs(proc, processor)
                self.assertEqual(budget, 512)
                captures.append({k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)})
                return f'label {len(captures) % 2}'
            with patch.object(backend.pipeline, 'frames_for', side_effect=frames), \
                    patch.object(backend.pipeline, 'generate', side_effect=generate):
                original, original_audit = backend.pipeline.caption_single(prepared.videos, segments,
                    prepared.profile, backend.session, instruction, backend.progress)
                expected = captures[:]
                for prefetch in (0, 1):
                    captures.clear()
                    direct, direct_audit = caption_single_direct(backend, prepared, segments, prefetch=prefetch)
                    self.assertEqual(direct, original)
                    self.assertEqual(direct_audit, original_audit)
                    for left, right in zip(expected, captures, strict=True):
                        self.assertEqual(set(left), set(right))
                        for name in left:
                            self.assertEqual(left[name].dtype, right[name].dtype)
                            self.assertTrue(torch.equal(left[name], right[name]), name)
