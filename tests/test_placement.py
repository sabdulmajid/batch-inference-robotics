import json
from pathlib import Path
import tempfile
import types
import unittest

from bir.placement import GIB, checkpoint_weight_bytes, memory_budgets, verify_residency, inspect_checkpoint


class PlacementTests(unittest.TestCase):
    def test_27b_cannot_fit_two_24gib_devices(self):
        with self.assertRaisesRegex(RuntimeError, 'weights exceed'):
            memory_budgets([(24 * GIB, 24 * GIB)] * 2, 4, weight_bytes=54_000_000_000)
        result = memory_budgets([(24 * GIB, 24 * GIB)] * 4, 4, weight_bytes=54_000_000_000)
        self.assertEqual(result, {i: 20 * GIB for i in range(4)})

    def test_uses_free_memory_not_advertised_capacity(self):
        self.assertEqual(memory_budgets([(24 * GIB, 10 * GIB)], 4), {0: 6 * GIB})
        with self.assertRaises(RuntimeError):
            memory_budgets([(24 * GIB, 3 * GIB)], 4)

    def test_requires_positive_finite_headroom_and_visible_devices(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                memory_budgets([(24 * GIB, 24 * GIB)], value)
        with self.assertRaises(RuntimeError):
            memory_budgets([])

    def test_fp32_storage_is_converted_to_native_mixed_dtype_estimate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            header = {'layer.weight': {'dtype': 'F32', 'shape': [100], 'data_offsets': [0, 400]},
                      'layer.A_log': {'dtype': 'F32', 'shape': [10], 'data_offsets': [400, 440]}}
            encoded = json.dumps(header).encode()
            (path / 'model-00001.safetensors').write_bytes(len(encoded).to_bytes(8, 'little') + encoded + bytes(440))
            index = {'metadata': {'total_size': 440},
                     'weight_map': {name: 'model-00001.safetensors' for name in header}}
            (path / 'model.safetensors.index.json').write_text(json.dumps(index))
            self.assertEqual(checkpoint_weight_bytes(root), 240)
            stats = inspect_checkpoint(root)
            self.assertEqual(stats['storage_tensor_bytes'], 440)
            self.assertEqual(stats['native_fp32_elements'], 10)
            index['metadata']['total_size'] = 441
            (path / 'model.safetensors.index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'total_size disagrees'):
                inspect_checkpoint(root)

    def test_rejects_cpu_buffers_and_parameters(self):
        class Tensor:
            def __init__(self, kind='cuda', dtype='torch.bfloat16'):
                self.device = types.SimpleNamespace(type=kind)
                self.dtype = dtype
            def numel(self): return 4
            def element_size(self): return 2
        class Model:
            def named_parameters(self): return [('weight', Tensor())]
            def named_buffers(self): return [('rotary', Tensor('cpu'))]
        with self.assertRaisesRegex(RuntimeError, 'non_cuda_buffers'):
            verify_residency(Model())
        Model.named_buffers = lambda self: []
        Model.named_parameters = lambda self: [('weight', Tensor('cuda', 'torch.int8'))]
        with self.assertRaisesRegex(RuntimeError, 'non_native_parameter_dtypes'):
            verify_residency(Model())


if __name__ == '__main__':
    unittest.main()
