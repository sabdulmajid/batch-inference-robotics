import importlib.util
import json
from pathlib import Path
import unittest
import types
from contextlib import nullcontext
from unittest.mock import patch

from bir.adaptive import (CaptionSpatialProcessor, COMPACT_TOTAL_PIXELS,
                          InvalidCompactOutput, SpatialPreset,
                          fallback_decision, run_compact_then_native, run_reference_interval)
from bir.native import PreparedEpisode

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


class ReferenceIntervalTests(unittest.TestCase):
    def setUp(self):
        self.native_processor = lambda: None
        self.model = object()
        self.calls, self.events = [], []
        class InputErrorForModel(RuntimeError): pass
        self.models = types.SimpleNamespace(InputErrorForModel=InputErrorForModel,
            runtime_event=lambda event, **values: self.events.append((event, values)))
        self.session = types.SimpleNamespace(kind='captioner', native=types.SimpleNamespace(model=self.model),
            load=lambda kind: (self.model, self.native_processor))
        self.error = None
        self.output_status = 'complete'
        def caption_single(videos, segments, profile, session, instruction, progress):
            model, processor = session.load('captioner')
            self.assertIs(model, self.model)
            compact = processor is not self.native_processor
            self.calls.append((compact, segments, videos, instruction))
            processor()
            self.models.runtime_event('generation_start', input_tokens=8, output_limit=512,
                                      input_tensor_digest='input-hash')
            if self.error is not None and compact:
                raise self.error
            self.models.runtime_event('generation_end', generated_tokens=3, complete=True)
            row = dict(start_s=segments[0]['start_s'], end_s=segments[0]['end_s'],
                       status=self.output_status, caption='Move the cup.')
            return [row], [{'event_id': 'event_0000', 'blocks': ['native-block']}]
        self.backend = types.SimpleNamespace(session=self.session, models=self.models,
            pipeline=types.SimpleNamespace(caption_single=caption_single), progress=lambda: None,
            _execution=nullcontext)
        self.prepared = PreparedEpisode({'instruction': 'original instruction'}, {'main': object()},
            'Human', {'duration_s': 10, 'input_sha256': {'main': 'source-hash'}})
        class Proxy:
            def __init__(self, processor, *, audit_sink): self.sink = audit_sink
            def __call__(self): self.sink({'frame_and_clock_sha256_before': 'same',
                                          'frame_and_clock_sha256_after': 'same'})
        self.proxy_patch = patch('bir.adaptive.CaptionSpatialProcessor', Proxy)
        self.proxy_patch.start()
        self.addCleanup(self.proxy_patch.stop)

    def run_policy(self, policy):
        self.syncs = []
        return run_reference_interval(self.backend, self.prepared, {'start_s': 2.0, 'end_s': 5.0},
            episode_id='episode', reference_index=3, policy=policy, freeze_sha256='a' * 64,
            synchronize=lambda: self.syncs.append(True))

    def test_native_default_preserves_processor_and_reference_identity(self):
        result = self.run_policy('native')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['interval_provenance'], 'fixed_independent_reference_interval')
        self.assertEqual(result['reference_index'], 3)
        self.assertEqual(result['spatial_audits'], [])
        self.assertFalse(self.calls[0][0])
        self.assertIs(self.calls[0][2], self.prepared.videos)
        self.assertEqual(self.calls[0][3], 'original instruction')
        self.assertEqual(len(self.syncs), 2)
        self.assertEqual(result['generation_events'][1]['generated_tokens'], 3)
        self.assertEqual(len(self.events), 2)  # original runtime logging remains active
        self.assertGreaterEqual(result['policy_wall_seconds'], result['application_work_seconds'])
        json.dumps(result, allow_nan=False)

    def test_compact_valid_and_adaptive_keep_one_native_model(self):
        for policy in ('compact', 'adaptive'):
            with self.subTest(policy=policy):
                self.calls.clear()
                result = self.run_policy(policy)
                self.assertEqual(result['chosen_pass'], 'compact')
                self.assertEqual(len(self.calls), 1)
                self.assertTrue(self.calls[0][0])
                self.assertEqual(result['spatial_audits'][0]['pass'], 'compact')
                self.assertIs(self.session.native.model, self.model)
                self.assertIs(self.session.load('captioner')[1], self.native_processor)
                self.assertEqual(result['gate'] is not None, policy == 'adaptive')

    def test_only_incomplete_native_output_triggers_adaptive_fallback(self):
        self.error = self.models.InputErrorForModel('The model did not return a complete output for this input.')
        result = self.run_policy('adaptive')
        self.assertEqual(result['chosen_pass'], 'native_fallback')
        self.assertEqual([item['category'] for item in result['attempts']], ['compact', 'gate', 'native_fallback'])
        self.assertEqual([item[0] for item in self.calls], [True, False])
        self.assertEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(len(self.syncs), 4)
        self.assertGreaterEqual(result['policy_wall_seconds'], result['application_work_seconds'])

    def test_context_failure_and_unsupported_intervals_do_not_trigger_fallback(self):
        for error in (RuntimeError('out of memory'),
                      self.models.InputErrorForModel('The selected video exceeds the native model context budget.')):
            self.calls.clear()
            self.error = error
            result = self.run_policy('adaptive')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(self.calls), 1)
            self.assertIsNone(result['gate'])
        self.error = None
        self.output_status = 'unsupported_target_frames'
        result = self.run_policy('adaptive')
        self.assertEqual(result['status'], 'unsupported_target_frames')
        self.assertIsNone(result['gate'])
        self.assertIsNone(result['caption'])

    def test_preloaded_model_and_supported_profile_required(self):
        self.session.kind = None
        with self.assertRaisesRegex(RuntimeError, 'Load and time'):
            self.run_policy('native')
        self.assertEqual(self.calls, [])


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
