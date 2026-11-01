import unittest
import types
from unittest.mock import patch
from bir.native import NativeBackend, PreparedEpisode
from bir.native_profile import NativeProfile


class NativeProfileTests(unittest.TestCase):
    def test_failure_records_original_identity_and_preserves_error(self):
        records = []
        profile = NativeProfile(True, sink=records.append)
        error = ValueError('fixture')
        with self.assertRaises(ValueError) as caught:
            with profile.span('processor', episode_id='video', request_ordinal=4):
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['episode_id'], 'video')
        self.assertEqual(records[0]['request_ordinal'], 4)
        self.assertEqual(records[0]['status'], 'failed')
        self.assertGreaterEqual(records[0]['elapsed_s'], 0)
        self.assertGreaterEqual(records[0]['end_monotonic_s'], records[0]['start_monotonic_s'])

    def test_disabled_has_no_sink_or_clock_effect(self):
        with NativeProfile(False, sink=lambda _: self.fail('disabled sink')).span('disabled'):
            pass

    def test_request_markers_bind_unchanged_upstream_calls_and_failures(self):
        backend = NativeBackend.__new__(NativeBackend)
        backend.frame_cache = None
        calls = []
        def generate(*args, **kwargs):
            calls.append((args, kwargs))
            if args[0] == 'failure':
                raise ValueError('original error')
            return 'unchanged caption'
        backend.pipeline = types.SimpleNamespace(generate=generate)
        prepared = PreparedEpisode({'id': 'clip'}, {}, 'Human', {'input_sha256': {'main': 'hash'}})
        with patch('bir.native._runtime_event') as event:
            with backend._execution(prepared, 'caption'):
                self.assertEqual(backend.pipeline.generate('model', budget=512), 'unchanged caption')
                with self.assertRaisesRegex(ValueError, 'original error'):
                    backend.pipeline.generate('failure', budget=512)
        self.assertIs(backend.pipeline.generate, generate)
        self.assertEqual(calls, [(('model',), {'budget': 512}), (('failure',), {'budget': 512})])
        self.assertEqual([c.args[0] for c in event.call_args_list],
            ['native_request_start', 'native_request_end', 'native_request_start', 'native_request_end'])
        self.assertEqual([c.kwargs['request_ordinal'] for c in event.call_args_list], [0, 0, 1, 1])
        self.assertTrue(all(c.kwargs['episode_id'] == 'clip' and c.kwargs['stage'] == 'caption'
                            for c in event.call_args_list))
        self.assertEqual(event.call_args.kwargs['status'], 'failed')
