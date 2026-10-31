"""Local calibration controls. No GPU evidence or production admission is made."""
import importlib.util
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
                                             verify_frozen_release)


class CalibrationControlTests(unittest.TestCase):
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
