"""CPU collector/routing tests; fabricated test receipts never admit real GPUs."""
import importlib.util
import json
import types
import unittest
from unittest.mock import patch

from bir.batching import (BatchAdmissionError, BatchCompatibilityError, CaptionRequest,
                          MeasuredBatchAdmission, collate_caption_requests)
from bir.caption_batching import CaptionCollector, DeferredCaption, caption_with_microbatch


class CollectorDisabledTests(unittest.TestCase):
    def test_no_measured_admission_disables_collector(self):
        with self.assertRaisesRegex(BatchAdmissionError, 'explicit measured'):
            CaptionCollector(lambda: None, session=None, episode_id='ep', request_id='parent',
                profile='human', contract_id='contract', admissions=[], topology_fingerprint='topology',
                max_pending_requests=2, max_prepared_bytes=100000)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires optional torch CPU environment')
class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.proc = types.SimpleNamespace(video_token_id=99, video_processor=types.SimpleNamespace(merge_size=2))
        self.session = types.SimpleNamespace(kind='captioner', audit={'captioner': {}})
        self.singles, self.batches = [], []
        first = CaptionRequest('one', 'ep', 'event0', 'Human', 'contract', self.inputs(7))
        second = CaptionRequest('two', 'ep', 'event1', 'Human', 'contract', self.inputs(6))
        batch = collate_caption_requests([first, second], video_token_id=99,
                                        max_batch_size=2, max_prepared_bytes=1024 * 1024)
        self.admission = MeasuredBatchAdmission('cpu-test-only', batch.compatibility_signature, 'no-gpu',
            2, (100,), (100,), (150,), (150,), 20, 512, True)
        self.binding_patch = patch('bir.caption_batching.bind_verified_native', return_value=object())
        self.binding_patch.start()
        self.addCleanup(self.binding_patch.stop)
        def generate(model, processor, batch, **kwargs):
            self.batches.append(batch)
            return {'complete': True, 'rows': [dict(request_id=r.request_id, episode_id=r.episode_id,
                event_id=r.event_id, status='complete', caption=str(int(r.inputs['input_ids'][0, 0])))
                for r in batch.requests]}
        self.batch_patch = patch('bir.caption_batching.generate_caption_batch', side_effect=generate)
        self.batch_mock = self.batch_patch.start()
        self.addCleanup(self.batch_patch.stop)

    def inputs(self, token):
        import torch
        return {'input_ids': torch.tensor([[token, 99, 8, 9]], dtype=torch.int64),
                'attention_mask': torch.ones((1, 4), dtype=torch.int64),
                'mm_token_type_ids': torch.tensor([[0, 2, 0, 0]], dtype=torch.int64),
                'pixel_values_videos': torch.full((4, 1536), float(token)),
                'video_grid_thw': torch.tensor([[1, 2, 2]], dtype=torch.int64)}

    def original(self, model, processor, inputs, budget):
        self.assertIs(model, self.model)
        self.assertIs(processor, self.proc)
        self.assertEqual(budget, 512)
        self.singles.append(int(inputs['input_ids'][0, 0]))
        return str(self.singles[-1])

    def collector(self, max_bytes=1024 * 1024):
        return CaptionCollector(self.original, session=self.session, episode_id='ep', request_id='parent',
            profile='Human', contract_id='contract', admissions=[self.admission], topology_fingerprint='topology',
            max_pending_requests=2, max_prepared_bytes=max_bytes, synchronize=lambda: None)

    def annotation(self, handles, indices=None):
        indices = indices or list(range(len(handles)))
        rows = [dict(status='unsupported_target_frames', caption='') for _ in range(max(indices) + 1)]
        for index, handle in zip(indices, handles, strict=True): rows[index] = dict(status='complete', caption=handle)
        return dict(segments=rows, pipeline={'caption_inputs': [{'event_id': f'event_{i:04d}'} for i in indices]})

    def test_exact_group_and_unmatched_single_resolve_through_actual_event_audits(self):
        collector = self.collector()
        handles = [collector(self.model, self.proc, self.inputs(token), 512) for token in (7, 6, 5)]
        self.assertTrue(all(isinstance(h, DeferredCaption) for h in handles))
        with self.assertRaises(TypeError): json.dumps(handles)
        result = collector.finish(self.annotation(handles, [0, 1, 3]))
        self.assertEqual([r['caption'] for r in result['segments']], ['7', '6', '', '5'])
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(self.singles, [5])
        self.assertEqual([r['native_event_id'] for r in collector.audit['request_routes']],
                         ['event_0000', 'event_0001', 'event_0003'])
        self.assertEqual(collector.source_bytes, 0)
        json.dumps(result, allow_nan=False)

    def test_byte_bound_flushes_singles_without_unsafe_batch(self):
        size = sum(t.numel() * t.element_size() for t in self.inputs(7).values())
        collector = self.collector(max_bytes=2 * size + 10)
        handles = [collector(self.model, self.proc, self.inputs(token), 512) for token in (7, 6, 5)]
        collector.finish(self.annotation(handles))
        self.assertEqual(self.singles, [7, 6, 5])
        self.assertEqual(self.batches, [])
        self.assertLessEqual(collector.audit['pending_source_peak_bytes'], collector.max_bytes // 2)

    def test_pending_mutation_and_output_swaps_fail_without_single_retry(self):
        collector = self.collector()
        inputs = self.inputs(7)
        collector(self.model, self.proc, inputs, 512)
        inputs['input_ids'][0, 0] = 4
        with self.assertRaisesRegex(BatchCompatibilityError, 'changed while'):
            collector(self.model, self.proc, self.inputs(6), 512)
        self.assertEqual(self.singles, [])
        self.assertEqual(self.batches, [])
        collector = self.collector()
        self.batch_mock.side_effect = lambda *args, **kwargs: {'complete': True, 'rows': []}
        collector(self.model, self.proc, self.inputs(7), 512)
        with self.assertRaisesRegex(BatchCompatibilityError, 'request order'):
            collector(self.model, self.proc, self.inputs(6), 512)
        self.assertEqual(self.singles, [])

    def test_tempseg_and_changed_native_event_routes_are_rejected(self):
        collector = self.collector()
        self.session.kind = 'tempseg'
        with self.assertRaisesRegex(RuntimeError, 'only native Captioner'):
            collector(self.model, self.proc, self.inputs(7), 1024)
        self.session.kind = 'captioner'
        handles = [collector(self.model, self.proc, self.inputs(7), 512)]
        output = self.annotation(handles)
        output['pipeline']['caption_inputs'][0]['event_id'] = 'event_0009'
        with self.assertRaisesRegex(BatchCompatibilityError, 'event audit'):
            collector.finish(output)

    def test_wrapper_restores_hook_and_returns_only_resolved_json(self):
        pipeline = types.SimpleNamespace(generate=self.original)
        def caption(prepared, segmentation):
            handles = [pipeline.generate(self.model, self.proc, self.inputs(token), 512) for token in (7, 6)]
            result = self.annotation(handles)
            result.update(elapsed_s=0, engine={'stage_elapsed_s': {'captioner': 0}})
            return result
        backend = types.SimpleNamespace(pipeline=pipeline, session=self.session, caption=caption)
        original = pipeline.generate
        result = caption_with_microbatch(backend, types.SimpleNamespace(profile='Human', videos={'main': object()}), {'elapsed_s': 2},
            episode_id='ep', request_id='parent', contract_id='contract', admissions=[self.admission],
            topology_fingerprint='topology', max_pending_requests=2, max_prepared_bytes=1024 * 1024)
        self.assertIs(pipeline.generate, original)
        self.assertEqual(result['engine']['request_batch_size'], 2)
        self.assertEqual(result['elapsed_s'], 2 + result['engine']['stage_elapsed_s']['captioner'])
        self.assertTrue(result['engine']['caption_collector']['complete'])
        json.dumps(result, allow_nan=False)

    def test_nonhuman_wrapper_rejects_before_native_pipeline_or_model_work(self):
        with self.assertRaisesRegex(BatchCompatibilityError, 'human single-view'):
            caption_with_microbatch(None, types.SimpleNamespace(profile='YAM', videos={'main': object()}), {},
                episode_id='ep', request_id='parent', contract_id='contract', admissions=[self.admission],
                topology_fingerprint='topology', max_pending_requests=2, max_prepared_bytes=1024 * 1024)


if __name__ == '__main__':
    unittest.main()
