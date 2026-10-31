"""Local calibration controls. No GPU evidence or production admission is made."""
import importlib.util
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from bir.batching import CaptionRequest, collate_caption_requests
from bir.batching_contract import native_batching_contract, native_batching_contract_id
from scripts.calibrate_caption_batch import (CompatiblePairSearch, admission_from_evidence, memory_stress,
                                             verify_frozen_release, calibration_intervals, capture_episode)
from bir.native import MODEL_REVISIONS, UPSTREAM_COMMIT, _json_digest


class CalibrationControlTests(unittest.TestCase):
    def signed_fixture(self, root):
        episode = dict(episode_id='homer_41', split='development', source_sha256='a' * 64,
                       instruction_sha256='b' * 64, duration_s=5, targets=[dict(reference_index=0, start_s=0, end_s=5)])
        freeze = dict(freeze_sha256='f' * 64, episodes=[episode])
        stage = dict(schema_version='bir_segmentation_v1', source_commit=UPSTREAM_COMMIT,
            input_sha256={'main': episode['source_sha256']}, instruction_sha256=episode['instruction_sha256'],
            model_revisions={k: {'revision': v} for k, v in MODEL_REVISIONS.items()},
            profile='Human', yam32=False, duration_s=5, temporal_audit=[{'native': 'window'}],
            segments=[dict(start_s=1, end_s=2, metadata={}), dict(start_s=3, end_s=4, metadata={})])
        stage['stage_digest'] = _json_digest(stage)
        checkpoint = root / 'segment.json'; checkpoint.write_text(json.dumps(stage))
        output = dict(episode_id='homer_41', input_sha256=stage['input_sha256'], profile='Human',
                      model_revisions=stage['model_revisions'], segments=stage['segments'],
                      pipeline={'temporal_windows': stage['temporal_audit']})
        (root / 'caption.json').write_text(json.dumps(output))
        args = types.SimpleNamespace(signed_segmentation=checkpoint, segment_indices=[0, 1], gate_evidence=root / 'gate.json')
        return args, freeze, {'run': {'exports': ['caption.json']}}, stage, output

    def test_signed_selection_binds_checkpoint_gate_intervals_and_prediction_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, gate, stage, output = self.signed_fixture(Path(temporary))
            targets, receipt = calibration_intervals(args, freeze, gate)
            self.assertEqual([r['signed_segment_index'] for r in targets['homer_41']], [0, 1])
            self.assertFalse(receipt['independent_reference'])
            self.assertEqual(receipt['stage_digest'], stage['stage_digest'])
            self.assertNotIn('reference_index', targets['homer_41'][0])
            args.signed_segmentation.write_text(json.dumps(stage, indent=2))
            changed = calibration_intervals(args, freeze, gate)[1]
            self.assertNotEqual(receipt['checkpoint_sha256'], changed['checkpoint_sha256'])
            self.assertEqual(receipt['stage_digest'], changed['stage_digest'])
            args.signed_segmentation = args.segment_indices = None
            references, receipt = calibration_intervals(args, freeze, gate)
            self.assertEqual(references['homer_41'], freeze['episodes'][0]['targets'])
            self.assertEqual(receipt['kind'], 'fixed_development_reference_intervals')

    def test_signed_selection_rejects_corruption_wrong_source_or_changed_gate_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, freeze, gate, stage, output = self.signed_fixture(root)
            stage['segments'][0]['end_s'] = 2.1
            args.signed_segmentation.write_text(json.dumps(stage))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                calibration_intervals(args, freeze, gate)
            stage['stage_digest'] = _json_digest({k: v for k, v in stage.items() if k != 'stage_digest'})
            args.signed_segmentation.write_text(json.dumps(stage))
            with self.assertRaisesRegex(ValueError, 'Gate C episode intervals'):
                calibration_intervals(args, freeze, gate)
            stage['input_sha256'] = {'main': 'c' * 64}
            stage['stage_digest'] = _json_digest({k: v for k, v in stage.items() if k != 'stage_digest'})
            args.signed_segmentation.write_text(json.dumps(stage))
            with self.assertRaisesRegex(ValueError, 'frozen development'):
                calibration_intervals(args, freeze, gate)

    def test_signed_selection_rejects_invalid_or_partial_pair_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, freeze, gate, _, _ = self.signed_fixture(Path(temporary))
            for indices in (None, [0, 0], [1, 0], [0, 9], [True, 1]):
                args.segment_indices = indices
                with self.assertRaises(ValueError): calibration_intervals(args, freeze, gate)

    def test_capture_keeps_predicted_interval_indices_separate_from_references(self):
        pipeline = types.SimpleNamespace(generate=None)
        def caption(videos, segments, profile, session, instruction, progress):
            model, processor = session.load('captioner')
            rows = [dict(caption=pipeline.generate(model, processor, {}, 512)) for segment in segments]
            return rows, [dict(event_id=f'event_{i:04d}') for i in range(len(segments))]
        pipeline.caption_single = caption
        backend = types.SimpleNamespace(_execution=nullcontext, pipeline=pipeline, progress=None)
        prepared = types.SimpleNamespace(videos={}, profile='Human', job={'instruction': ''})
        targets = [dict(signed_segment_index=i, start_s=i, end_s=i + 1,
                        interval_provenance='signed_tempseg_model_predicted_intervals') for i in (10, 13)]
        routes = capture_episode(backend, prepared, types.SimpleNamespace(id='homer_41', request_id='request'),
                                 targets, object(), types.SimpleNamespace(offer=lambda request: None), 'contract')
        self.assertEqual([r['signed_segment_index'] for r in routes.values()], [10, 13])
        self.assertTrue(all('reference_index' not in r for r in routes.values()))

    def test_live_release_guard_rejects_changed_code_or_gate_evidence(self):
        args = types.SimpleNamespace(model_lock=None, captioner_headroom_gib=4,
                                     max_prepared_mib=2, freeze=None, gate_evidence=None)
        contract = {'runtime': 'original'}
        report = {'calibration_sources_sha256': {'runner': 'original'},
                  'freeze_sha256': 'freeze', 'gate_c': {'receipt': 'original'}}
        with patch('scripts.calibrate_caption_batch.native_batching_contract', return_value=contract), \
                patch('scripts.calibrate_caption_batch.calibration_sources', return_value={'runner': 'original'}) as sources, \
                patch('scripts.calibrate_caption_batch.load_freeze', return_value={'freeze_sha256': 'freeze'}), \
                patch('scripts.calibrate_caption_batch.verify_gate_c', return_value={'receipt': 'original'}) as gate:
            verify_frozen_release(args, report, contract)
            sources.return_value = {'runner': 'changed'}
            with self.assertRaisesRegex(RuntimeError, 'implementation'):
                verify_frozen_release(args, report, contract)
            sources.return_value = {'runner': 'original'}
            gate.return_value = {'receipt': 'changed'}
            with self.assertRaisesRegex(RuntimeError, 'Gate C evidence'):
                verify_frozen_release(args, report, contract)

    def test_live_release_guard_rechecks_signed_interval_receipt(self):
        args = types.SimpleNamespace(model_lock=None, captioner_headroom_gib=4,
                                     max_prepared_mib=2, freeze=None, gate_evidence=None)
        report = dict(calibration_sources_sha256={}, freeze_sha256='freeze', gate_c={},
                      interval_selection={'checkpoint_sha256': 'original'})
        with patch('scripts.calibrate_caption_batch.native_batching_contract', return_value={}), \
                patch('scripts.calibrate_caption_batch.calibration_sources', return_value={}), \
                patch('scripts.calibrate_caption_batch.load_freeze', return_value={'freeze_sha256': 'freeze'}), \
                patch('scripts.calibrate_caption_batch.verify_gate_c', return_value={}), \
                patch('scripts.calibrate_caption_batch.read_json', return_value={}), \
                patch('scripts.calibrate_caption_batch.calibration_intervals',
                      return_value=({}, report['interval_selection'])) as selection:
            verify_frozen_release(args, report, {})
            selection.return_value = ({}, {'checkpoint_sha256': 'changed'})
            with self.assertRaisesRegex(RuntimeError, 'interval selection changed'):
                verify_frozen_release(args, report, {})

    def test_contract_is_path_independent_and_separate_from_run_evidence(self):
        original = native_batching_contract()
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / 'same-lock.json'
            lock.write_bytes(Path('configs/native-models.json').read_bytes())
            self.assertEqual(original, native_batching_contract(model_lock=lock))
        self.assertNotEqual(native_batching_contract_id(original),
                            native_batching_contract_id(native_batching_contract(max_prepared_bytes=1024**2)))
        self.assertNotIn('episode_id', original)
        self.assertNotIn('evidence_id', original)
        self.assertIn('caption_batching.py', original['runtime_source_sha256'])
        self.assertIn('batching_contract.py', original['runtime_source_sha256'])

    def records(self):
        def record(baseline, peak):
            return dict(status='complete', baseline=[dict(logical_device=0, allocated_bytes=baseline,
                reserved_bytes=baseline + 10, total_bytes=1000)], gpu_memory=[dict(logical_device=0,
                peak_allocated_bytes=peak - 20, peak_reserved_bytes=peak)])
        return record(100, 300), record(120, 400)

    def test_admission_requires_agreement_stress_and_all_device_headroom(self):
        native, stress = self.records()
        args = dict(batch=types.SimpleNamespace(compatibility_signature='test-shape'), fingerprint='test-only',
                    evidence_id='test-only', native_agreement=True, batch_record=native,
                    stress_record=stress, stress_complete=True, headroom_bytes=100)
        admission = admission_from_evidence(**args)
        self.assertEqual(admission.baseline_allocated_bytes, (100,))
        self.assertEqual(admission.peak_reserved_bytes, (400,))
        self.assertEqual(admission.max_generated_tokens_observed, 512)
        for change in ({'native_agreement': False}, {'stress_complete': False}, {'headroom_bytes': 700}):
            with self.assertRaises(ValueError): admission_from_evidence(**dict(args, **change))
        stress['gpu_memory'][0]['logical_device'] = 1
        with self.assertRaisesRegex(ValueError, 'device order'):
            admission_from_evidence(**args)

    def test_restart_rejection_cannot_write_a_production_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run([sys.executable, 'scripts/calibrate_caption_batch.py',
                '--source-root', str(root), '--manifest', str(root / 'missing.jsonl'),
                '--gate-evidence', str(root / 'missing.json'), '--output-dir', str(root / 'run')],
                capture_output=True, text=True, env=dict(os.environ, SLURM_RESTART_COUNT='1'))
            self.assertEqual(result.returncode, 1)
            report = json.loads((root / 'run/calibration.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertFalse(report['admission_written'])
            self.assertEqual(report['passes'], [])
            self.assertFalse((root / 'run/admission.json').exists())


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires optional torch CPU package')
class CalibrationCpuTests(unittest.TestCase):
    def request(self, name, token, length=4, episode='episode'):
        import torch
        ids = [token, 99] + [8] * (length - 2)
        values = {'input_ids': torch.tensor([ids], dtype=torch.int64),
                  'attention_mask': torch.ones((1, length), dtype=torch.int64),
                  'mm_token_type_ids': torch.tensor([[0, 2] + [0] * (length - 2)], dtype=torch.int64),
                  'pixel_values_videos': torch.full((4, 1536), float(token)),
                  'video_grid_thw': torch.tensor([[1, 2, 2]], dtype=torch.int64)}
        return CaptionRequest(name, episode, name, 'Human', 'cpu-test-contract', values)

    def test_first_distinct_exact_pair_respects_bounds_and_episode_boundary(self):
        search = CompatiblePairSearch(video_token_id=99, max_prepared_bytes=1024**2, max_candidates=2)
        first = self.request('one', 7)
        search.offer(first)
        search.offer(self.request('same-content', 7))
        self.assertIsNone(search.selected)
        search.offer(self.request('other-episode', 6, episode='other'))
        self.assertIsNone(search.selected)
        search.offer(self.request('two', 5))
        self.assertEqual([r.request_id for r in search.selected], ['one', 'two'])
        self.assertLessEqual(search.peak_retained_source_bytes, search.max_bytes // 2)

    def test_search_eviction_and_oversize_inputs_do_not_create_a_fake_pair(self):
        search = CompatiblePairSearch(video_token_id=99, max_prepared_bytes=1024**2, max_candidates=1)
        for request in (self.request('a', 7), self.request('b', 6, length=5), self.request('c', 5)):
            search.offer(request)
        self.assertIsNone(search.selected)
        small = CompatiblePairSearch(video_token_id=99, max_prepared_bytes=1024, max_candidates=1)
        small.offer(self.request('large', 7))
        self.assertIsNone(small.selected)
        self.assertEqual(small.source_bytes, 0)

    def stress_fixture(self, mutate_global=False, output_tokens=512):
        import torch
        class Config:
            def __init__(self): self.value = 2
            def to_dict(self): return {'value': self.value}
        class Model:
            def __init__(self): self.config = Config(); self.generation_config = Config()
            def parameters(self): yield torch.zeros(1)
            def generate(model, **kwargs):
                self.assertIsNot(kwargs['generation_config'], model.generation_config)
                self.assertIsNone(kwargs['eos_token_id'])
                kwargs['generation_config'].value = 99
                if mutate_global: model.generation_config.value = 100
                return torch.cat((kwargs['input_ids'], torch.ones((2, output_tokens), dtype=torch.int64)), dim=1)
        model = Model()
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))
        batch = collate_caption_requests([self.request('one', 7), self.request('two', 6)],
                                         video_token_id=99, max_batch_size=2, max_prepared_bytes=1024**2)
        return model, processor, batch

    def test_stress_uses_private_settings_and_requires_all512_tokens(self):
        with patch('scripts.calibrate_caption_batch.processor_signature', return_value='cpu-test-processor'):
            model, processor, batch = self.stress_fixture()
            result = memory_stress(model, processor, batch)
            self.assertTrue(result['complete'])
            self.assertTrue(result['persistent_settings_unchanged'])
            self.assertFalse(result['native_generation'])
            self.assertFalse(result['quality_evidence'])
            self.assertEqual(model.generation_config.value, 2)
            short = memory_stress(*self.stress_fixture(output_tokens=511))
            self.assertFalse(short['complete'])
            with self.assertRaisesRegex(RuntimeError, 'changed persistent'):
                memory_stress(*self.stress_fixture(mutate_global=True))


if __name__ == '__main__':
    unittest.main()
