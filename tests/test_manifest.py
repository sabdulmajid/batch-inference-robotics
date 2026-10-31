import json
from pathlib import Path
import tempfile
import unittest

from bir.manifest import load_manifest, ManifestError


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "video.mp4").write_bytes(b"fixture bytes, not a valid video")
        self.path = self.root / "jobs.jsonl"

    def load(self, rows, contract=None):
        self.path.write_text("\n".join(json.dumps(r) for r in rows))
        return load_manifest(self.path, contract=contract or {"source": "pinned", "models": "pinned"})

    def test_identity_tracks_content_and_settings_but_not_paths(self):
        row = dict(id="episode", video="video.mp4")
        first = self.load([row])[0]
        (self.root / "moved.mp4").write_bytes((self.root / "video.mp4").read_bytes())
        moved = self.load([dict(row, video="moved.mp4")])[0]
        self.assertEqual(first.request_id, moved.request_id)
        instructed = self.load([dict(row, instruction="Move the cup")])[0]
        self.assertNotEqual(first.request_id, instructed.request_id)
        changed = self.load([row], {"source": "new", "models": "pinned"})[0]
        self.assertNotEqual(first.request_id, changed.request_id)
        (self.root / "video.mp4").write_bytes(b"changed")
        with self.assertRaises(ManifestError):
            first.verify_sources()
        self.assertNotEqual(first.request_id, self.load([row])[0].request_id)

    def test_rejects_duplicate_ids_and_checksum_mismatch(self):
        row = dict(id="episode", video="video.mp4")
        with self.assertRaisesRegex(ManifestError, "Duplicate"):
            self.load([row, row])
        with self.assertRaisesRegex(ManifestError, "checksums"):
            self.load([dict(row, source_sha256={"main": "0" * 64})])

    def test_rejects_invalid_native_camera_contract(self):
        row = dict(id="episode", video="video.mp4")
        for updates in ({"yam32": 1}, {"left_wrist": "video.mp4"}, {"yam32": True}, {"unknown": 1}):
            with self.assertRaises(ManifestError):
                self.load([dict(row, **updates)])


if __name__ == "__main__":
    unittest.main()
