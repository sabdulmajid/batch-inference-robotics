from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_phase2 as script
from bir.manifest import canonical_json, load_manifest


class PreparePhaseTwoTests(unittest.TestCase):
    def test_frozen_selection_digest_rejects_changed_episode(self):
        selection = script.load_selection('manifests/phase2-native.json')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'selection.json'
            selection['episodes'][0]['duration_s'] += 1
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'digest changed'):
                script.load_selection(path)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'requires preparation dependency')
    def test_exact_video_bytes_and_relocated_block_manifests(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        payloads = {'one': b'original-first-video', 'two': b'original-second-video'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parquet = root / 'data.parquet'
            pq.write_table(pa.Table.from_pylist([dict(id=k, video={'bytes': v}) for k, v in payloads.items()]), parquet)
            pin = replace(script.WGO, parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest())
            episodes = [dict(episode_id=k, source_sha256=hashlib.sha256(v).hexdigest(),
                profile='human', instruction='', duration_s=2., source_size=[64, 48]) for k, v in payloads.items()]
            selection = dict(dataset=pin.__dict__, episodes=episodes, source_seconds=4.,
                blocks=[dict(block=0, episode_ids=['two', 'one'])])
            selection['selection_sha256'] = hashlib.sha256(canonical_json(selection).encode()).hexdigest()
            class Backend:
                def __init__(self, source):
                    pass
                def prepare(self, job):
                    self_path = Path(job['video'])
                    assert self_path.read_bytes() == payloads[job['id']]
                    return SimpleNamespace(metadata=dict(input_sha256=job['source_sha256'],
                        duration_s=2., cameras={'main': {'size': [64, 48]}}))
                def close(self):
                    pass
            output = root / 'selected'
            with patch.object(script, 'WGO', pin):
                result = script.prepare(parquet, 'unused', selection, output, backend_factory=Backend)
            self.assertEqual(result['selected_video_count'], 2)
            self.assertEqual(result['reencoded_videos'], 0)
            self.assertEqual(result['new_model_calls'], 0)
            moved = root / 'relocated'
            output.rename(moved)
            jobs = load_manifest(moved / 'block-0.jsonl', contract={'test': 1})
            self.assertEqual([job.id for job in jobs], ['two', 'one'])
            for job in jobs:
                self.assertEqual(job.video.read_bytes(), payloads[job.id])
            with patch.object(script, 'WGO', pin), self.assertRaisesRegex(ValueError, 'fresh extraction'):
                script.prepare(parquet, 'unused', selection, moved, backend_factory=Backend)


if __name__ == '__main__':
    unittest.main()
