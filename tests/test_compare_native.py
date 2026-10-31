import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_native.py'
SPEC = importlib.util.spec_from_file_location('bir_compare_native_test', SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = {
            'episode_id': 'episode-1', 'request_id': 'a' * 64, 'duration_s': 60,
            'profile': 'Human', 'input_sha256': {'main': 'source-hash'},
            'model_revisions': {'captioner': {'revision': 'pinned'}}, 'camera_roles': ['main'],
            'segments': [{'start_s': 0, 'end_s': 60, 'status': 'complete', 'caption': 'Move a cup.',
                          'segmentation_provenance': 'fine'}],
            'pipeline': {'temporal_windows': [{'window': {'scope_id': 's1'},
                'frame_plan': {'source_frame_indices': [0, 1]}, 'prompt_sha256': 'prompt', 'output': 'raw'}],
                'caption_inputs': [{'event_id': 'event_0', 'blocks': [{'frame_indices': [0, 1]}],
                                    'prompt_sha256': 'caption-prompt'}]},
            'engine': {'source_commit': 'source-revision', 'placement': {'arbitrary': 'runtime-only'}},
            'elapsed_s': 100,
        }

    def make_run(self, name, *, fresh=True, elapsed=10, stage='all', output=None):
        directory = self.root / name
        (directory / 'annotations').mkdir(parents=True)
        output = copy.deepcopy(output or self.output)
        request = output['request_id']
        stages = ['segment', 'caption'] if stage == 'all' else [stage]
        summary = {'condition': name, 'complete': True, 'elapsed_s': elapsed,
            'total_cli_workflow_s': elapsed, 'fresh_stage_attempts': len(stages) if fresh else 0,
            'requested_stages': stage,
            'records': [{'id': output['episode_id'], 'request_id': request, 'stage': item,
                         'status': 'complete' if fresh else 'resumed', 'fresh_execution': fresh, 'attempt': 1}
                        for item in stages],
            'attempts': [{'request_id': request, 'stage': item, 'number': 1,
                          'state': 'complete', 'elapsed_s': 2} for item in stages] if fresh else []}
        (directory / 'summary.json').write_text(json.dumps(summary))
        suffix = 'segment' if stage == 'segment' else 'caption'
        (directory / 'annotations' / f'{request}.{suffix}.json').write_text(json.dumps(output))
        return directory

    def test_identity_comparison_ignores_placement_timing_and_request_condition_ids(self):
        left = self.make_run('reference', elapsed=20)
        other = copy.deepcopy(self.output)
        other['request_id'] = 'b' * 64
        other['elapsed_s'] = 1
        other['engine']['placement'] = {'other': 'ignored'}
        right = self.make_run('optimized', elapsed=10, output=other)
        result = COMPARE.compare(left, right)
        self.assertTrue(result['compatible'])
        self.assertFalse(result['tensor_input_agreement_verified'])
        self.assertEqual(result['right_wall_speedup_over_left'], 2)
        self.assertIsNone(result['performance']['left']['allocated_gpu_hours'])

    def test_reports_caption_and_frame_plan_differences_without_raw_values_or_paths(self):
        left = self.make_run('reference')
        other = copy.deepcopy(self.output)
        other['segments'][0]['caption'] = '/private/secret/source.mp4'
        other['pipeline']['temporal_windows'][0]['frame_plan']['source_frame_indices'][1] = 2
        right = self.make_run('optimized', output=other)
        result = COMPARE.compare(left, right)
        self.assertFalse(result['compatible'])
        self.assertEqual(result['difference_count'], 2)
        self.assertEqual(result['differences_by_group'], {'segments': 1, 'temporal': 1})
        self.assertNotIn('/private/secret', json.dumps(result))

    def test_cached_retrieval_cannot_be_an_inference_benchmark(self):
        result = COMPARE.compare(self.make_run('reference'), self.make_run('resumed', fresh=False))
        self.assertFalse(result['valid_fresh_comparison'])
        self.assertIsNone(result['right_wall_speedup_over_left'])
        self.assertIsNone(result['performance']['right']['source_hours_per_wall_hour'])

    def test_missing_and_duplicate_episode_artifacts_fail(self):
        left, right = self.make_run('reference'), self.make_run('optimized')
        next((right / 'annotations').glob('*.json')).unlink()
        self.assertFalse(COMPARE.compare(left, right)['valid_fresh_comparison'])
        right = self.make_run('duplicate')
        summary = COMPARE.read_json(right / 'summary.json')
        summary['records'].append(copy.deepcopy(summary['records'][0]))
        (right / 'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'Duplicate episode/stage'):
            COMPARE.compare(left, right)

    def test_tensor_digests_compare_multisets_across_staged_order(self):
        def log(name, records):
            path = self.root / name
            path.write_text(''.join('FLORA_RUNTIME ' + json.dumps(row) + '\n' for row in records))
            return path
        event = lambda kind, value: [dict(event='model_load_start', kind=kind),
            dict(event='generation_start', input_tensor_digest=value * 64, input_tokens=100, output_limit=512)]
        a = log('left.log', event('tempseg', 'a') + event('captioner', 'b'))
        b = log('right.log', event('captioner', 'b') + event('tempseg', 'a'))
        result = COMPARE.compare(self.make_run('reference'), self.make_run('optimized'), left_logs=[a], right_logs=[b])
        self.assertTrue(result['tensor_input_agreement_verified'])
        b.write_text(b.read_text().replace('b' * 64, 'c' * 64))
        result = COMPARE.compare(self.root / 'reference', self.root / 'optimized', left_logs=[a], right_logs=[b])
        self.assertFalse(result['compatible'])
        self.assertEqual(result['processor_tensors']['right_only_calls'], 1)

    def test_known_allocations_include_failed_time_and_reject_double_charge(self):
        a = {'allocation_key': 'private-a', 'gpus': 4, 'elapsed_seconds': 900, 'final_state': 'COMPLETED'}
        b = {'allocation_key': 'private-b', 'gpus': 4, 'elapsed_seconds': 450, 'final_state': 'FAILED'}
        left, right = self.make_run('reference'), self.make_run('optimized')
        result = COMPARE.compare(left, right, accounting={'left': [a, b]})
        self.assertEqual(result['performance']['left']['allocated_gpu_hours'], 1.5)
        self.assertNotIn('private-a', json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'both conditions'):
            COMPARE.compare(left, right, accounting={'left': [a], 'right': [a]})

    def test_caption_only_comparison_requires_named_stage(self):
        left, right = self.make_run('reference', stage='caption'), self.make_run('optimized', stage='caption')
        self.assertFalse(COMPARE.compare(left, right)['valid_fresh_comparison'])
        self.assertTrue(COMPARE.compare(left, right, stage='caption')['compatible'])

    def test_segmentation_subset_checks_checkpoint_integrity(self):
        output = copy.deepcopy(self.output)
        output['temporal_audit'] = output.pop('pipeline')['temporal_windows']
        output['source_commit'] = output['engine']['source_commit']
        output['stage_digest'] = COMPARE.digest(output)
        left, right = self.make_run('reference', stage='segment', output=output), self.make_run('optimized', stage='segment', output=output)
        self.assertTrue(COMPARE.compare(left, right, stage='segment')['compatible'])
        artifact = next((right / 'annotations').glob('*.segment.json'))
        corrupted = COMPARE.read_json(artifact)
        corrupted['duration_s'] = 61
        artifact.write_text(json.dumps(corrupted))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            COMPARE.compare(left, right, stage='segment')

    def test_claimed_freshness_requires_matching_successful_attempt_records(self):
        left, right = self.make_run('reference'), self.make_run('optimized')
        summary = COMPARE.read_json(right / 'summary.json')
        summary['attempts'][1]['state'] = 'failed'
        (right / 'summary.json').write_text(json.dumps(summary))
        result = COMPARE.compare(left, right)
        self.assertFalse(result['valid_fresh_comparison'])

    def test_retry_from_an_earlier_run_is_not_a_fresh_benchmark_boundary(self):
        left, right = self.make_run('reference'), self.make_run('optimized')
        summary = COMPARE.read_json(right / 'summary.json')
        summary['attempts'][0]['number'] = 2
        summary['records'][0]['attempt'] = 2
        (right / 'summary.json').write_text(json.dumps(summary))
        result = COMPARE.compare(left, right)
        self.assertFalse(result['valid_fresh_comparison'])
        self.assertTrue(any(i['error'] == 'prior_attempts_are_outside_this_run_wall_measurement'
                            for i in result['issues']))


if __name__ == '__main__':
    unittest.main()
