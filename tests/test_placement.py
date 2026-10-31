import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from bir.placement import GIB, captioner_placement, checkpoint_weight_bytes, memory_budgets, verify_residency, inspect_checkpoint


class PlacementTests(unittest.TestCase):
    def test_failed_capacity_floor_retains_all_device_measurements_without_loading(self):
        original = Mock()
        auto_model = types.SimpleNamespace(from_pretrained=original)
        modules = {'torch': types.SimpleNamespace(bfloat16='bf16', cuda=types.SimpleNamespace(
            device_count=lambda: 3, mem_get_info=lambda i: (20 * GIB, 24 * GIB),
            memory=types.SimpleNamespace(get_allocator_backend=lambda: 'native'),
            memory_allocated=lambda i: (i + 1) * GIB, memory_reserved=lambda i: (i + 2) * GIB)),
            'transformers': types.SimpleNamespace(AutoModelForImageTextToText=auto_model),
            'transformers.models.qwen3_5.modeling_qwen3_5': types.SimpleNamespace(
                Qwen3_5PreTrainedModel=types.SimpleNamespace(_keep_in_fp32_modules_strict=['A_log', 'dt_bias']))}
        audit = {}
        with patch.dict('sys.modules', modules), patch('bir.placement.inspect_checkpoint',
                return_value={'native_weight_bytes_estimate': 55 * GIB}):
            with captioner_placement(4, audit):
                with self.assertRaisesRegex(RuntimeError, 'weights exceed'):
                    auto_model.from_pretrained('private-snapshot', device_map='balanced', dtype='bf16',
                        local_files_only=True, attn_implementation={'text_config': 'sdpa', 'vision_config': 'sdpa'})
        original.assert_not_called()
        self.assertEqual(audit['feasibility'], 'failed')
        self.assertEqual(audit['headroom_gib'], 4)
        self.assertEqual([r['logical_device'] for r in audit['device_memory']], [0, 1, 2])
        self.assertEqual(audit['device_memory'][2], dict(logical_device=2, total_bytes=24 * GIB,
                         free_bytes=20 * GIB, allocated_bytes=3 * GIB, reserved_bytes=4 * GIB,
                         unused_allocator_bytes=GIB, credited_allocator_cache_bytes=GIB,
                         effective_available_bytes=21 * GIB))

    def test_observed_reload_cache_is_capacity_without_counting_live_tensors(self):
        # Exact failed-floor receipt from the two-episode diagnostic. This
        # checks accounting only; successful GPU reload remains a separate gate.
        total = 25_296_044_032
        free = [24_834_408_448, 5_926_486_016, 5_683_216_384]
        allocated = [19_136_512, 8_519_680, 8_519_680]
        reserved = [109_051_904, 19_031_654_400, 19_274_924_032]
        devices = [(total, n) for n in free]
        with self.assertRaisesRegex(RuntimeError, 'weights exceed'):
            memory_budgets(devices, 4, weight_bytes=54_713_466_336)
        budget = memory_budgets(devices, 4, weight_bytes=54_713_466_336,
                                allocator_bytes=list(zip(allocated, reserved)))
        self.assertEqual(budget, {0: 20_629_356_544, 1: 20_654_653_440, 2: 20_654_653_440})
        self.assertTrue(all(budget[i] + 4 * GIB <= total - allocated[i] for i in range(3)))
        # A large reserve that is occupied by tensors is not reusable cache.
        with self.assertRaisesRegex(RuntimeError, 'weights exceed'):
            memory_budgets(devices, 4, weight_bytes=54_713_466_336,
                           allocator_bytes=list(zip(reserved, reserved)))

    def test_allocator_credit_is_validated_capped_and_optional(self):
        self.assertEqual(memory_budgets([(24 * GIB, 22 * GIB)], 4,
                         allocator_bytes=[(3 * GIB, 8 * GIB)]), {0: 17 * GIB})
        self.assertEqual(memory_budgets([(24 * GIB, 0)], 4,
                         allocator_bytes=[(2 * GIB, 20 * GIB)]), {0: 14 * GIB})
        for values in ([(4, 3)], [(-1, 3)], [(0, 25 * GIB)], []):
            with self.assertRaises(ValueError):
                memory_budgets([(24 * GIB, 20 * GIB)], allocator_bytes=values)

    def test_unknown_allocator_gets_no_cache_credit(self):
        original = Mock(return_value=object())
        auto_model = types.SimpleNamespace(from_pretrained=original)
        modules = {'torch': types.SimpleNamespace(bfloat16='bf16', cuda=types.SimpleNamespace(
            device_count=lambda: 1, mem_get_info=lambda i: (6 * GIB, 24 * GIB),
            memory=types.SimpleNamespace(get_allocator_backend=lambda: 'cudaMallocAsync'),
            memory_allocated=lambda i: GIB, memory_reserved=lambda i: 18 * GIB)),
            'transformers': types.SimpleNamespace(AutoModelForImageTextToText=auto_model),
            'transformers.models.qwen3_5.modeling_qwen3_5': types.SimpleNamespace(
                Qwen3_5PreTrainedModel=types.SimpleNamespace(_keep_in_fp32_modules_strict=['A_log', 'dt_bias']))}
        audit = {}
        with patch.dict('sys.modules', modules), patch('bir.placement.inspect_checkpoint',
                return_value={'native_weight_bytes_estimate': GIB}):
            with captioner_placement(4, audit):
                auto_model.from_pretrained('private-snapshot', device_map='balanced', dtype='bf16',
                    local_files_only=True, attn_implementation={'text_config': 'sdpa', 'vision_config': 'sdpa'})
        self.assertEqual(original.call_args.kwargs['max_memory'], {0: 2 * GIB})
        self.assertEqual(audit['device_memory'][0]['credited_allocator_cache_bytes'], 0)
        self.assertEqual(audit['capacity_basis'], 'driver_free_only_unsupported_allocator')

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
