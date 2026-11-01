import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from bir.manifest import canonical_json
from bir.review import ReviewServer
from bir.workflow import _record, prepare_demo, records


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'original.mp4'
        self.video.write_bytes(b'0123456789abcdef')
        self.sha = hashlib.sha256(self.video.read_bytes()).hexdigest()
        identity = dict(id='demo', source_sha256={'main': self.sha}, profile='human', contract={'fixture': True})
        row = _record(identity, '1' * 64, 0, dict(start_s=0.5, end_s=1.5, caption='<script>unsafe</script>'),
                      state='complete', failure=None, origin={'precomputed': True})
        self.example = self.root / 'example.json'
        self.example.write_text(canonical_json(dict(schema_version='bir_precomputed_example_v1', precomputed=True,
            source=dict(episode_id='demo', sha256=self.sha), attribution={'dataset': 'test fixture'}, records=[row])))
        self.run = self.root / 'demo'
        self.result = prepare_demo(self.run, self.example, video=self.video)

    def server(self):
        server = ReviewServer(self.run, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        return server

    def test_local_routes_range_stream_and_edit_export_roundtrip(self):
        server = self.server()
        self.assertEqual(server.server_address[0], '127.0.0.1')
        self.assertEqual(self.result['new_model_calls'], 0)
        with urlopen(server.url) as response:
            page = response.read().decode()
        self.assertIn('PRECOMPUTED DEMONSTRATION', page)
        self.assertNotIn('<script>unsafe</script>', page)
        with urlopen(server.url + 'data') as response:
            data = json.load(response)
        self.assertTrue(data['precomputed'])
        row = data['records'][0]
        request = Request(server.url + 'video/' + self.sha, headers={'Range': 'bytes=3-7'})
        with urlopen(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers['Content-Range'], 'bytes 3-7/16')
            self.assertEqual(response.read(), b'34567')
        payload = dict(record_id=row['record_id'], reviewed_caption='Verified user correction.', reviewer='reviewer', expected_revision=0)
        request = Request(server.url + 'review', data=canonical_json(payload).encode(),
                          headers={'Content-Type': 'application/json', 'X-BIR-Review': server.token,
                                   'Origin': f'http://127.0.0.1:{server.server_port}'})
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
        with urlopen(server.url + 'export') as response:
            output = json.loads(response.read())
        self.assertEqual(output['generated_caption'], '<script>unsafe</script>')
        self.assertEqual(output['reviewed_caption'], 'Verified user correction.')
        self.assertTrue(output['generation']['precomputed'])

    def test_cross_origin_token_and_unknown_files_are_rejected(self):
        server = self.server()
        for path in ['../example.json', 'video/../../example.json']:
            with self.assertRaises(HTTPError) as error:
                urlopen(server.url + path)
            self.assertEqual(error.exception.code, 404)
        for headers in [{'Content-Type': 'application/json'}, {'Content-Type': 'application/json', 'X-BIR-Review': server.token, 'Origin': 'https://untrusted.example'}]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(server.url + 'review', data=b'{}', headers=headers))
            self.assertEqual(error.exception.code, 403)
        self.assertFalse((self.run / 'reviews.jsonl').exists())

    def test_source_change_and_invalid_video_ranges_are_rejected(self):
        server = self.server()
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(server.url + 'video/' + self.sha, headers={'Range': 'bytes=100-200'}))
        self.assertEqual(error.exception.code, 416)
        self.video.write_bytes(b'changed original')
        with self.assertRaises(HTTPError) as error:
            urlopen(server.url + 'video/' + self.sha)
        self.assertEqual(error.exception.code, 400)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            ReviewServer(self.run, port=0)

    def test_demo_requires_exact_original_bytes_and_preserves_precomputed_identity(self):
        self.video.write_bytes(b'wrong video')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prepare_demo(self.root / 'wrong', self.example, video=self.video)
        self.assertFalse((self.root / 'wrong').exists())
        self.assertEqual(records(self.run)[0]['reviewed_caption'], None)


if __name__ == '__main__':
    unittest.main()
