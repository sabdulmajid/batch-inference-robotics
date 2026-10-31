import importlib.util
from pathlib import Path
import unittest

from bir.batching import (BatchAdmissionError, BatchCompatibilityError, CaptionRequest,
                          MeasuredBatchAdmission, collate_caption_requests,
                          extract_batch_outputs, verify_batch_integrity, generate_caption_batch,
                          VerifiedNativeBinding, runtime_fingerprint)

HAS_TORCH = importlib.util.find_spec('torch') is not None
ASSETS = Path(__file__).resolve().parents[1] / '.local' / 'adaptive-processor'
HAS_PROCESSOR = HAS_TORCH and importlib.util.find_spec('transformers') is not None and (ASSETS / 'tokenizer.json').is_file()


@unittest.skipUnless(HAS_TORCH, 'requires optional torch CPU package')
class BatchTests(unittest.TestCase):
    def make_request(self, name, first_token=7):
        import torch
        values = {'input_ids': torch.tensor([[first_token, 99, 8, 9]], dtype=torch.int64),
                  'attention_mask': torch.ones((1, 4), dtype=torch.int64),
                  'mm_token_type_ids': torch.tensor([[0, 2, 0, 0]], dtype=torch.int64),
                  'pixel_values_videos': torch.full((4, 1536), float(first_token)),
                  'video_grid_thw': torch.tensor([[1, 2, 2]], dtype=torch.int64)}
        return CaptionRequest(name, 'episode-' + name, 'event-0', 'human', 'synthetic-test-contract', values)

    def batch(self, requests=None):
        return collate_caption_requests(requests or [self.make_request('a'), self.make_request('b', 6)],
            video_token_id=99, max_batch_size=2, max_prepared_bytes=1024 * 1024)

    def test_strict_shape_key_and_identity_rejection(self):
        first, second = self.make_request('a'), self.make_request('b')
        second.inputs['position_ids'] = second.inputs['input_ids'].clone()
        with self.assertRaisesRegex(BatchCompatibilityError, 'Unrecognized'):
            self.batch([first, second])
        with self.assertRaisesRegex(BatchCompatibilityError, 'Duplicate'):
            self.batch([first, first])
        second = self.make_request('b')
        second.inputs['attention_mask'][0, 0] = 0
        with self.assertRaisesRegex(BatchCompatibilityError, 'Padded'):
            self.batch([first, second])
        with self.assertRaisesRegex(BatchCompatibilityError, 'byte bound'):
            collate_caption_requests([first], video_token_id=99, max_batch_size=2, max_prepared_bytes=1)

    def test_visual_order_and_mutation_guard(self):
        import torch
        batch = self.batch()
        verify_batch_integrity(batch)
        self.assertTrue(torch.equal(batch.inputs['pixel_values_videos'][:4], batch.requests[0].inputs['pixel_values_videos']))
        self.assertTrue(torch.equal(batch.inputs['pixel_values_videos'][4:], batch.requests[1].inputs['pixel_values_videos']))
        self.assertEqual(batch.routing[1]['pixel_rows'], [4, 8])
        batch.inputs['pixel_values_videos'][4, 0] = 0
        with self.assertRaisesRegex(BatchCompatibilityError, 'changed after collation'):
            verify_batch_integrity(batch)

    def test_each_output_has_its_own_eos_and_request_identity(self):
        import torch
        batch = self.batch()
        tokenizer = type('Tokenizer', (), {'eos_token_id': 2, 'pad_token_id': 0,
            'decode': lambda self, ids, skip_special_tokens: ' '.join(str(i) for i in ids if i not in (0, 2))})()
        output = torch.cat((batch.inputs['input_ids'], torch.tensor([[10, 2, 0], [11, 12, 2]])), dim=1)
        rows = extract_batch_outputs(output, batch, tokenizer)
        self.assertEqual([(r['request_id'], r['caption'], r['generated_tokens']) for r in rows], [('a', '10', 2), ('b', '11 12', 3)])
        self.assertEqual(rows[0]['padding_tokens'], 1)
        self.assertEqual(rows[1]['padding_tokens'], 0)
        with self.assertRaisesRegex(RuntimeError, 'prefixes'):
            extract_batch_outputs(output.flip(0), batch, tokenizer)

    def test_missing_eos_or_text_is_never_complete(self):
        import torch
        batch = self.batch()
        tokenizer = type('Tokenizer', (), {'eos_token_id': [2, 3], 'pad_token_id': 0,
            'decode': lambda self, ids, skip_special_tokens: ' '.join(str(i) for i in ids if i not in (0, 2, 3))})()
        output = torch.cat((batch.inputs['input_ids'], torch.tensor([[10, 11], [3, 0]])), dim=1)
        rows = extract_batch_outputs(output, batch, tokenizer)
        self.assertEqual([r['status'] for r in rows], ['failed_no_eos', 'failed_empty_caption'])
        output = torch.cat((batch.inputs['input_ids'], torch.tensor([[2, 10], [11, 3]])), dim=1)
        rows = extract_batch_outputs(output, batch, tokenizer)
        self.assertEqual(rows[0]['status'], 'failed_tokens_after_eos')
        self.assertEqual(rows[1]['status'], 'complete')

    def test_rejected_admission_keeps_preflight_elapsed_work(self):
        batch = self.batch()
        with self.assertRaises(BatchAdmissionError) as raised:
            generate_caption_batch(None, None, batch, admission=None,
                                   topology_fingerprint='synthetic-test', native_binding=None)
        record = raised.exception.record
        self.assertEqual(record['status'], 'failed_preflight')
        self.assertGreaterEqual(record['admission_and_integrity_seconds'], 0)
        self.assertEqual(record['total_work_including_collation_seconds'],
                         record['runner_total_seconds'] + batch.collation_seconds)

    def test_processor_video_token_must_match_prepared_inputs(self):
        from types import SimpleNamespace
        from bir.native import UPSTREAM_COMMIT, MODEL_REVISIONS
        batch = self.batch()
        model, processor = object(), SimpleNamespace(video_token_id=123)
        binding = VerifiedNativeBinding(UPSTREAM_COMMIT, MODEL_REVISIONS['captioner'],
            MODEL_REVISIONS['captioner'], 'synthetic-test-contract', 'synthetic-receipt',
            id(model), id(processor))
        with self.assertRaisesRegex(BatchAdmissionError, 'video token'):
            runtime_fingerprint(model, processor, batch, topology_fingerprint='synthetic-test', native_binding=binding)

    def test_admission_requires_exact_measured_configuration_and_headroom(self):
        batch = self.batch()
        # Synthetic test values only. This is not a usable GPU admission record.
        values = dict(evidence_id='synthetic-unit-test', compatibility_signature=batch.compatibility_signature,
                      runtime_fingerprint='synthetic-runtime', batch_size=2,
                      baseline_allocated_bytes=(100,), baseline_reserved_bytes=(100,),
                      peak_allocated_bytes=(140,), peak_reserved_bytes=(150,), safety_headroom_bytes=20,
                      max_generated_tokens_observed=512)
        environment = dict(runtime_fingerprint='synthetic-runtime', total_bytes=(200,), free_bytes=(100,), allocated_bytes=(100,), reserved_bytes=(100,))
        with self.assertRaisesRegex(BatchAdmissionError, 'native-agreement'):
            MeasuredBatchAdmission(**values).require(batch, **environment)
        with self.assertRaisesRegex(BatchAdmissionError, '512-token'):
            MeasuredBatchAdmission(**dict(values, max_generated_tokens_observed=32), native_agreement_verified=True).require(batch, **environment)
        admission = MeasuredBatchAdmission(**values, native_agreement_verified=True)
        admission.require(batch, **environment)
        with self.assertRaisesRegex(BatchAdmissionError, 'Current free'):
            admission.require(batch, **dict(environment, free_bytes=(50,)))
        with self.assertRaisesRegex(BatchAdmissionError, 'live allocation'):
            admission.require(batch, **dict(environment, allocated_bytes=(159,), reserved_bytes=(160,), free_bytes=(40,)))
        with self.assertRaisesRegex(BatchAdmissionError, 'Hardware'):
            admission.require(batch, **dict(environment, runtime_fingerprint='different'))
        with self.assertRaisesRegex(BatchAdmissionError, 'Batch size'):
            admission.require(self.batch([self.make_request('a')]), **environment)


@unittest.skipUnless(HAS_PROCESSOR, 'requires pinned processor packages and assets')
class ActualProcessorBatchTests(unittest.TestCase):
    def test_collation_equals_actual_joint_processor_call(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        from transformers.video_utils import VideoMetadata
        proc = AutoProcessor.from_pretrained(ASSETS, local_files_only=True, use_fast=True,
            min_pixels=1024, max_pixels=589824, video_processor=Qwen3VLVideoProcessor.from_pretrained(
                ASSETS, local_files_only=True, fps=8, max_frames=40))
        text = proc.apply_chat_template([{'role': 'user', 'content': 'BEFORE:\n' + proc.video_token + '\nTARGET:\n' + proc.video_token + '\nDescribe TARGET.'}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        frames = [[[Image.new('RGB', (64, 64), (shade, block, 0)) for _ in range(count)]
                   for block, count in enumerate((4, 8))] for shade in (16, 240)]
        metadata = [[VideoMetadata(total_num_frames=len(block), fps=8, frames_indices=list(range(len(block))))
                     for block in videos] for videos in frames]
        kwargs = dict(do_sample_frames=False, return_metadata=True, return_tensors='pt')
        single = [proc(text=[text], videos=videos, video_metadata=clocks, **kwargs)
                  for videos, clocks in zip(frames, metadata)]
        requests = [CaptionRequest(str(i), 'episode-' + str(i), 'event-0', 'human', 'synthetic-cpu-processor', inputs)
                    for i, inputs in enumerate(single)]
        batch = collate_caption_requests(requests, video_token_id=proc.video_token_id,
                                         max_batch_size=2, max_prepared_bytes=4 * 1024 * 1024)
        joint = proc(text=[text, text], videos=[block for videos in frames for block in videos],
                     video_metadata=[clock for clocks in metadata for clock in clocks], **kwargs)
        for key in batch.inputs:
            self.assertTrue(torch.equal(batch.inputs[key], joint[key]), key)
        verify_batch_integrity(batch)
        tensor_bytes = sum(v.numel() * v.element_size() for v in batch.inputs.values())
        self.assertEqual(batch.collated_tensor_bytes, tensor_bytes)
        self.assertEqual(batch.prepared_bytes, 2 * tensor_bytes)
        self.assertEqual([r['request_id'] for r in batch.routing], ['0', '1'])
        self.assertEqual([r['grid_rows'] for r in batch.routing], [[0, 2], [2, 4]])
        self.assertNotEqual(batch.routing[0]['input_tensor_digest'], batch.routing[1]['input_tensor_digest'])


if __name__ == '__main__':
    unittest.main()
