import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_asset_provenance.py'
SPEC = importlib.util.spec_from_file_location('bir_asset_provenance_test', SCRIPT)
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)


class AssetProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_identity = 'a' * 64
        metadata = {'source_model_identity_sha256': self.source_identity, 'source_weights_modified': False}
        provenance = json.dumps(metadata).encode()
        weights = b'small fixture only'
        self.provenance_sha = hashlib.sha256(provenance).hexdigest()
        manifest = (self.provenance_sha + '  MODEL_PROVENANCE.json\n' +
                    hashlib.sha256(weights).hexdigest() + '  model.safetensors\n').encode()
        (self.root / 'MODEL_PROVENANCE.json').write_bytes(provenance)
        (self.root / 'SOURCE_MANIFEST.sha256').write_bytes(manifest)
        (self.root / 'model.safetensors').write_bytes(weights)
        self.pins = {'repository': 'test/fixture', 'revision': 'fixture-revision',
                     'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                     'provenance_sha256': self.provenance_sha,
                     'source_model_identity_sha256': self.source_identity}

    def test_source_identity_is_separate_from_manifest_checksum(self):
        result = PROVENANCE.check_captioner_provenance(self.root, pins=self.pins)
        self.assertEqual(result['source_model_identity_sha256'], self.source_identity)
        self.assertNotEqual(result['manifest_sha256'], self.source_identity)
        self.assertEqual(result['model_weight_bytes_read'], 0)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_rewritten_self_consistent_manifest_cannot_replace_pinned_bytes(self):
        payload = b'modified fixture'
        (self.root / 'model.safetensors').write_bytes(payload)
        (self.root / 'SOURCE_MANIFEST.sha256').write_text(self.provenance_sha + '  MODEL_PROVENANCE.json\n' +
            hashlib.sha256(payload).hexdigest() + '  model.safetensors\n')
        with self.assertRaisesRegex(ValueError, 'independently pinned manifest'):
            PROVENANCE.check_captioner_provenance(self.root, pins=self.pins)

    def test_metadata_only_call_does_not_claim_payload_integrity(self):
        (self.root / 'model.safetensors').write_bytes(b'corrupt')
        result = PROVENANCE.check_captioner_provenance(self.root, pins=self.pins)
        self.assertEqual(result['status'], 'metadata_trust_anchors_verified')
        with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
            PROVENANCE.check_captioner_provenance(self.root, pins=self.pins, verify_payloads=True)

    def test_full_fixture_verification_counts_bytes(self):
        result = PROVENANCE.check_captioner_provenance(self.root, pins=self.pins, verify_payloads=True)
        self.assertEqual(result['payload_files_verified'], 2)
        self.assertEqual(result['model_weight_bytes_read'], len(b'small fixture only'))

    def test_rewritten_provenance_is_rejected_even_when_manifest_unchanged(self):
        (self.root / 'MODEL_PROVENANCE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'independently pinned provenance'):
            PROVENANCE.check_captioner_provenance(self.root, pins=self.pins)

    def test_large_payload_cli_rejects_login_or_restarted_execution(self):
        with patch.dict('os.environ', {}, clear=True), patch('builtins.print'):
            self.assertEqual(PROVENANCE.main([str(self.root), '--verify-payloads']), 1)
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'fixture', 'SLURM_RESTART_COUNT': '1'}, clear=True), patch('builtins.print'):
            self.assertEqual(PROVENANCE.main([str(self.root), '--verify-payloads']), 1)


if __name__ == '__main__':
    unittest.main()
