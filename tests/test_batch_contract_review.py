"""Independent stale-admission checks with simulated device identity, not GPU proof."""
from contextlib import ExitStack
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bir.batching import BatchAdmissionError, MeasuredBatchAdmission, VerifiedNativeBinding, runtime_fingerprint
from bir.native import MODEL_REVISIONS, UPSTREAM_COMMIT


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires optional CPU torch dependency')
class BatchContractReviewTests(unittest.TestCase):
    def setUp(self):
        import torch
        config = SimpleNamespace(to_dict=lambda: {'test_configuration': True})
        self.model = SimpleNamespace(config=config, generation_config=config, hf_device_map={'test': 0},
                                     named_parameters=lambda: [('test_weight', torch.zeros(1))])
        tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: '{"test_tokenizer":true}'),
                                    eos_token_id=2, pad_token_id=0)
        self.processor = SimpleNamespace(tokenizer=tokenizer, video_token_id=99, chat_template='test template',
                                         video_processor=config, image_processor=config)
        self.batch = SimpleNamespace(requests=[SimpleNamespace(contract_id='old-supplied-contract')] * 2,
                                     video_token_id=99, compatibility_signature='synthetic-shape')
        self.binding = VerifiedNativeBinding(UPSTREAM_COMMIT, MODEL_REVISIONS['captioner'],
            MODEL_REVISIONS['captioner'], 'old-supplied-contract', 'synthetic-load-audit',
            id(self.model), id(self.processor))
        self.hashes = {name: 'a' * 64 for name in ('batching.py', 'batching_contract.py',
            'caption_batching.py', 'native.py', 'placement.py', 'frame_cache.py')}

    def fingerprint(self, *, hashes=None, uuid='physical-device-one'):
        with ExitStack() as stack:
            for target, result in [
                ('torch.cuda.device_count', 1),
                ('torch.cuda.get_device_properties', SimpleNamespace(uuid=uuid, total_memory=1000)),
                ('torch.cuda.get_device_name', 'same device model'),
                ('torch.cuda.get_device_capability', (8, 6)),
                ('bir.batching.importlib.metadata.version', '5.8.1'),
                ('bir.batching_contract.runtime_source_hashes', self.hashes if hashes is None else hashes),
            ]:
                stack.enter_context(patch(target, return_value=result))
            return runtime_fingerprint(self.model, self.processor, self.batch,
                topology_fingerprint='old-supplied-topology', native_binding=self.binding)

    def admission(self, fingerprint):
        return MeasuredBatchAdmission('synthetic-test-only', self.batch.compatibility_signature,
            fingerprint, 2, (100,), (100,), (140,), (150,), 20, 512, True)

    def assert_stale_rejected(self, original, changed):
        self.assertNotEqual(original, changed)
        with self.assertRaisesRegex(BatchAdmissionError, 'Hardware, placement, or software'):
            self.admission(original).require(self.batch, runtime_fingerprint=changed,
                total_bytes=(1000,), free_bytes=(900,), allocated_bytes=(100,), reserved_bytes=(100,))

    def test_current_module_changes_reject_admission_even_with_old_supplied_contract_id(self):
        original, details = self.fingerprint()
        self.assertEqual(details['native_binding']['execution_contract_id'], 'old-supplied-contract')
        for module in self.hashes:
            with self.subTest(module=module):
                changed, _ = self.fingerprint(hashes=dict(self.hashes, **{module: 'b' * 64}))
                self.assert_stale_rejected(original, changed)

    def test_current_device_uuid_rejects_old_supplied_topology_admission(self):
        original, _ = self.fingerprint()
        changed, details = self.fingerprint(uuid='physical-device-two')
        self.assertEqual(details['topology_fingerprint'], 'old-supplied-topology')
        self.assert_stale_rejected(original, changed)
        with self.assertRaisesRegex(BatchAdmissionError, 'physical UUID'):
            self.fingerprint(uuid=None)


if __name__ == '__main__':
    unittest.main()
