import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_assets.py'
SPEC = importlib.util.spec_from_file_location('bir_prepare_assets_test', SCRIPT)
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class AssetPreparationTests(unittest.TestCase):
    def test_base_hash_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'checkpoint test fixture'
            shard = root / 'shard.safetensors'
            shard.write_bytes(payload)
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'tensor': shard.name}}))
            expected = {shard.name: (len(payload), hashlib.sha256(payload).hexdigest())}
            self.assertEqual(ASSETS.verify_base_weights(root, expected)[shard.name]['bytes'], len(payload))
            shard.write_bytes(b'x' * len(payload))
            with self.assertRaisesRegex(RuntimeError, 'hash check failed'):
                ASSETS.verify_base_weights(root, expected)

    def test_base_rejects_unexpected_checkpoint_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'tensor': 'unexpected.safetensors'}}))
            with self.assertRaisesRegex(RuntimeError, 'shard map differs'):
                ASSETS.verify_base_weights(root)

    def test_allocation_guards_run_before_imports_or_downloads(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'authorized CPU allocation'):
                ASSETS.main()
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '1'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'restarted allocation'):
                ASSETS.main()


if __name__ == '__main__':
    unittest.main()
