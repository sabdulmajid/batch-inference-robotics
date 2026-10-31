import importlib.util
import json
from pathlib import Path
import unittest

from bir.adaptive import (CaptionSpatialProcessor, COMPACT_TOTAL_PIXELS,
                          InvalidCompactOutput, SpatialPreset,
                          fallback_decision, run_compact_then_native)

ASSETS = Path(__file__).resolve().parents[1] / '.local' / 'adaptive-processor'
HAS_PROCESSOR = (importlib.util.find_spec('transformers') is not None and
                 (ASSETS / 'tokenizer.json').is_file())


class AdaptiveGateTests(unittest.TestCase):
    def test_valid_compact_does_not_inspect_native_output(self):
        def forbidden_native():
            self.fail('The format gate must not inspect the native output')
        result = run_compact_then_native(lambda: 'Move the cup.', forbidden_native)
        self.assertEqual(result['chosen_pass'], 'compact')
        self.assertEqual([r['category'] for r in result['attempts']], ['compact', 'gate'])
        self.assertFalse(result['quality_assessed'])

    def test_invalid_compact_counts_both_passes_and_gate(self):
        def incomplete():
            raise InvalidCompactOutput('No stop token')
        result = run_compact_then_native(incomplete, lambda: 'Move the cup.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['chosen_pass'], 'native_fallback')
        self.assertEqual([r['category'] for r in result['attempts']], ['compact', 'gate', 'native_fallback'])
        self.assertEqual(result['attempts'][0]['status'], 'invalid_output')
        self.assertEqual(result['application_work_seconds'], sum(r['wall_seconds'] for r in result['attempts']))

    def test_failure_is_recorded_without_automatic_native_retry(self):
        def fail(): raise RuntimeError('out of memory')
        result = run_compact_then_native(fail, lambda: self.fail('No fallback after infrastructure failure'))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['attempts']), 1)
        self.assertGreaterEqual(result['application_work_seconds'], 0)

    def test_failed_fallback_remains_failed(self):
        result = run_compact_then_native(lambda: '', lambda: '')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['caption'])
        self.assertEqual(len(result['attempts']), 3)

    def test_format_valid_is_not_a_quality_judgment(self):
        self.assertFalse(fallback_decision('A plausible but unverified caption.')['fallback'])
        self.assertTrue(fallback_decision('caption', generation_complete=False)['fallback'])
        self.assertTrue(fallback_decision('  ')['fallback'])
        with self.assertRaises(ValueError): SpatialPreset(max_total_pixels_per_block=-1)


@unittest.skipUnless(HAS_PROCESSOR, 'requires exact processor packages and pinned local tokenizer assets')
class ActualProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        cls.processor = AutoProcessor.from_pretrained(ASSETS, local_files_only=True, use_fast=True,
            min_pixels=1024, max_pixels=589824,
            video_processor=Qwen3VLVideoProcessor.from_pretrained(ASSETS, local_files_only=True, fps=8, max_frames=40))
        cls.processor.chat_template = (ASSETS / 'chat_template.jinja').read_text()

    def test_real_spatial_override_keeps_temporal_evidence_and_nonvisual_tokens(self):
        from PIL import Image
        from transformers.video_utils import VideoMetadata
        videos = [[Image.new('RGB', (512, 512), (i * 3, i * 2, i)) for i in range(count)]
                  for count in (4, 40)]
        metadata = [VideoMetadata(total_num_frames=len(frames), fps=8,
                                   frames_indices=list(range(len(frames)))) for frames in videos]
        prompt = self.processor.apply_chat_template([{'role': 'user', 'content':
            'BEFORE:\n' + self.processor.video_token + '\nTARGET:\n' + self.processor.video_token +
            '\nWrite the instruction for TARGET.'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        kwargs = dict(text=[prompt], videos=videos, video_metadata=metadata,
                      do_sample_frames=False, return_metadata=True, return_tensors='pt')
        native = self.processor(**kwargs)
        original_size = (self.processor.video_processor.size.shortest_edge,
                         self.processor.video_processor.size.longest_edge)
        compact_processor = CaptionSpatialProcessor(self.processor)
        compact = compact_processor(**kwargs)
        native_grid = native['video_grid_thw'].tolist()
        compact_grid = compact['video_grid_thw'].tolist()
        record = compact_processor.records[-1]
        self.assertEqual(record['compact_processor_size']['longest_edge'], COMPACT_TOTAL_PIXELS)
        self.assertEqual(native_grid, [b['native_grid_thw_from_geometry'] for b in record['blocks']])
        self.assertEqual([grid[0] for grid in native_grid], [grid[0] for grid in compact_grid])
        self.assertEqual(native_grid[0], compact_grid[0])  # short context is a measured no-op
        self.assertLess(compact_grid[1][1] * compact_grid[1][2], native_grid[1][1] * native_grid[1][2])
        self.assertEqual(record['frame_and_clock_sha256_before'], record['frame_and_clock_sha256_after'])
        self.assertEqual(original_size, (self.processor.video_processor.size.shortest_edge,
                                         self.processor.video_processor.size.longest_edge))
        video_id = self.processor.video_token_id
        without_video_tokens = lambda inputs: [i for i in inputs['input_ids'][0].tolist() if i != video_id]
        self.assertEqual(without_video_tokens(native), without_video_tokens(compact))
        self.assertEqual([m.frames_indices for m in native['video_metadata']],
                         [m.frames_indices for m in compact['video_metadata']])
        self.assertTrue(record['blocks'][0]['spatial_no_op'])
        self.assertFalse(record['blocks'][1]['spatial_no_op'])

    def test_temporal_resampling_is_rejected(self):
        compact = CaptionSpatialProcessor(self.processor)
        with self.assertRaisesRegex(ValueError, 'do_sample_frames=False'):
            compact(text=['text'], videos=[], do_sample_frames=True)
        with self.assertRaisesRegex(ValueError, 'spatial override'):
            compact(text=['text'], videos=[], do_sample_frames=False,
                    videos_kwargs={'size': {'shortest_edge': 4096, 'longest_edge': 8192}})


if __name__ == '__main__':
    unittest.main()
