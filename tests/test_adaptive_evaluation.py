import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import evaluate_adaptive as evaluator
import prepare_frozen_wgo as extract


def fixture():
    ids = ['dev', 'final1', 'final2']
    references, candidates = [], []
    for i, identity in enumerate(ids):
        source = hashlib.sha256(identity.encode()).hexdigest()
        references.append({'id': identity, 'source_sha256': source, 'instruction': '', 'references':
                           [{'start_s': 0., 'end_s': 10., 'caption': 'Place the cup on the table.'}]})
        candidates.append({'id': identity, 'source_sha256': source, 'duration_s': 10., 'group_id': str(i),
                           'native_supported': True, 'references_valid': True, 'reference_count': 1})
    protocol = {'frozen': True, 'maximum_adaptive_development_revisions': 0, 'development_ids': ['dev'],
                'final_ids': ['final1', 'final2'], 'reviewed_task_groups': {i: i for i in ids}}
    candidate_manifest = {'candidates': candidates, 'dataset': {'repository': 'fixture', 'revision': 'test', 'parquet_sha256': 'fixture'}}
    return evaluator.create_freeze(candidate_manifest, references, protocol), references, candidate_manifest, protocol


def outcomes(freeze, split, selection=None):
    rows = [{'episode_id': episode, 'reference_index': index, 'policy': policy,
             'freeze_sha256': freeze['freeze_sha256'], 'start_s': target['start_s'], 'end_s': target['end_s'],
             'execution_contract_sha256': 'c' * 64,
             'status': 'complete', 'caption': 'Put the cup onto the table.',
             'policy_wall_seconds': 2 if policy == 'native' else 1,
             'input_sha256': {'main': target['episode']['source_sha256']}}
            for (episode, index), target in evaluator.targets_for(freeze, split).items() for policy in evaluator.POLICIES]
    for row in rows:
        if selection is not None:
            row['development_selection_digest'] = evaluator.digest(selection)
        row['result_sha256'] = evaluator.digest(row)
    return rows


def reviews(mapping):
    return [{'review_id': f'reviewer-{i}', 'evaluator_type': 'AI', 'evaluator_model': 'fixture-only',
             'rubric_sha256': evaluator.RUBRIC_SHA256,
             'ratings': [{'card_id': identity, 'action_match': 'yes', 'object_match': 'yes', 'result_match': 'yes',
                          'contradiction': 'no', 'unsupported_extra_claims': False, 'rationale': 'Fixture synonym agreement.'}
                         for identity in mapping['card_ids']]} for i in range(2)]


class AdaptiveEvaluationTests(unittest.TestCase):
    def test_freeze_rejects_task_leakage_and_reference_changes(self):
        freeze, references, candidates, protocol = fixture()
        protocol['reviewed_task_groups']['final1'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            evaluator.create_freeze(candidates, references, protocol)
        changed = copy.deepcopy(references)
        changed[1]['references'][0]['caption'] = 'A different event.'
        with self.assertRaisesRegex(ValueError, 'captions changed'):
            evaluator.validate_references(freeze, changed)

    def test_blinding_deduplicates_same_caption_without_hiding_policy_costs(self):
        freeze, references, _, _ = fixture()
        cards, mapping = evaluator.prepare_review(freeze, references, outcomes(freeze, 'final'), 'final')
        self.assertEqual(len(cards), 2)
        self.assertEqual(len(mapping['mapping']), 6)
        self.assertNotIn('policy', cards[0])
        self.assertNotIn('episode_id', cards[0])
        self.assertEqual(sum(r['policy_wall_seconds'] for r in mapping['mapping']), 8)

    def test_fixed_intervals_cannot_be_changed_by_output_matching(self):
        freeze, references, _, _ = fixture()
        rows = outcomes(freeze, 'final')
        rows[0]['end_s'] = 9.9
        rows[0]['result_sha256'] = evaluator.digest({k: v for k, v in rows[0].items() if k != 'result_sha256'})
        with self.assertRaisesRegex(ValueError, 'changed the frozen interval'):
            evaluator.prepare_review(freeze, references, rows, 'final')
        rows = outcomes(freeze, 'final')
        rows[0].pop('input_sha256')
        rows[0]['result_sha256'] = evaluator.digest({k: v for k, v in rows[0].items() if k != 'result_sha256'})
        with self.assertRaisesRegex(ValueError, 'different source'):
            evaluator.prepare_review(freeze, references, rows, 'final')

    def test_final_two_group_margin_remains_inconclusive_with_perfect_reviews(self):
        freeze, references, _, _ = fixture()
        _, development = evaluator.prepare_review(freeze, references, outcomes(freeze, 'development'), 'development')
        selection = evaluator.score_reviews(freeze, development, reviews(development))
        self.assertEqual(selection['best_fixed_policy'], 'compact')
        _, final = evaluator.prepare_review(freeze, references, outcomes(freeze, 'final', selection), 'final')
        scored = evaluator.score_reviews(freeze, final, reviews(final), selection)
        self.assertEqual(scored['adaptive_conclusion'], 'inconclusive')
        self.assertEqual(scored['quality_margin']['episodes'], 2)
        self.assertFalse(scored['is_human_evaluation'])
        self.assertNotIn('candidate_caption', json.dumps(scored))
        self.assertEqual(scored['outcome_set_sha256'], final['outcome_set_sha256'])
        self.assertEqual(scored['execution_contract_sha256'], 'c' * 64)
        changed = dict(selection, best_fixed_policy='native')
        with self.assertRaisesRegex(ValueError, 'bound before inference'):
            evaluator.score_reviews(freeze, final, reviews(final), changed)

    def test_missing_outputs_score_zero_and_review_disagreement_is_visible(self):
        freeze, references, _, _ = fixture()
        rows = outcomes(freeze, 'development')
        rows = [r for r in rows if r['policy'] != 'adaptive']
        _, mapping = evaluator.prepare_review(freeze, references, rows, 'development')
        ratings = reviews(mapping)
        ratings[1]['ratings'][0]['action_match'] = 'no'
        result = evaluator.score_reviews(freeze, mapping, ratings)
        self.assertEqual(result['policy_metrics']['adaptive']['mean_episode_score'], 0)
        self.assertIsNone(result['policy_metrics']['adaptive']['policy_application_seconds'])
        self.assertEqual(result['policy_metrics']['native']['review_disagreements'], 1)
        self.assertEqual(result['offline_audit']['all_policy_interval_items'], 3)
        self.assertEqual(result['offline_audit']['automatic_failure_items_in_denominator'], 1)
        self.assertIsNone(result['offline_audit']['reviewer_wall_seconds_sum'])
        self.assertEqual(result['offline_audit']['reviewers_with_unmeasured_wall_time'], 2)

    def test_offline_review_time_is_separate_and_paid_api_calls_are_rejected(self):
        freeze, references, _, _ = fixture()
        _, mapping = evaluator.prepare_review(freeze, references, outcomes(freeze, 'development'), 'development')
        ratings = reviews(mapping)
        for review, seconds in zip(ratings, [7.5, 9.]):
            review['review_wall_seconds'] = seconds
        result = evaluator.score_reviews(freeze, mapping, ratings)
        self.assertEqual(result['offline_audit']['reviewer_wall_seconds_sum'], 16.5)
        self.assertEqual(result['offline_audit']['card_presentations'], 2)
        self.assertEqual(result['offline_audit']['new_paid_api_calls'], 0)
        self.assertFalse(result['offline_audit']['included_in_deployment_policy_cost'])
        self.assertEqual(result['policy_metrics']['native']['policy_application_seconds'], 2)
        ratings[0]['new_paid_api_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'zero new paid'):
            evaluator.score_reviews(freeze, mapping, ratings)

    def test_mapping_tampering_and_missing_reviews_fail(self):
        freeze, references, _, _ = fixture()
        _, mapping = evaluator.prepare_review(freeze, references, outcomes(freeze, 'development'), 'development')
        missing = reviews(mapping)
        missing[1]['ratings'] = []
        with self.assertRaisesRegex(ValueError, 'missing, duplicate'):
            evaluator.score_reviews(freeze, mapping, missing)
        changed = copy.deepcopy(mapping)
        changed['mapping'].pop()
        with self.assertRaisesRegex(ValueError, 'mapping changed'):
            evaluator.score_reviews(freeze, changed, reviews(mapping))


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'requires optional Parquet preparation dependency')
class FrozenExtractionTests(unittest.TestCase):
    def test_extracts_original_selected_bytes_and_never_decodes_unselected_video(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        freeze, references, candidates, protocol = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / 'fixture.parquet'
            # Identity bytes hash to the source checksums declared by fixture().
            records = [{'id': row['id'], 'instruction': '', 'metadata': '{}', 'video': row['id'].encode(),
                        'segments': [{'start_sec': 0., 'end_sec': 10., 'subtask': row['references'][0]['caption']}]}
                       for row in references]
            records.insert(1, {'id': 'unselected', 'instruction': '', 'metadata': '{}', 'video': b'not decoded', 'segments': []})
            pq.write_table(pa.Table.from_pylist(records), parquet)
            pin = extract.WGO.__class__('fixture', 'test', hashlib.sha256(parquet.read_bytes()).hexdigest())
            candidates['dataset'] = pin.__dict__
            freeze = evaluator.create_freeze(candidates, references, protocol)
            calls = []
            class Backend:
                def __init__(self, source): pass
                def prepare(self, job):
                    calls.append(job['id'])
                    return types.SimpleNamespace(metadata={'input_sha256': job['source_sha256'], 'duration_s': 10.})
                def close(self): pass
            first = extract.prepare_frozen(parquet, 'unused', freeze, root / 'first', 'final', pin=pin, backend_factory=Backend)
            self.assertEqual(calls, ['final1', 'final2'])
            self.assertEqual(first['unselected_videos_decoded'], 0)
            for identity in ('final1', 'final2'):
                sha = hashlib.sha256(identity.encode()).hexdigest()
                self.assertEqual((root / 'first' / 'videos' / (sha + '.mp4')).read_bytes(), identity.encode())
            second = extract.prepare_frozen(parquet, 'unused', freeze, root / 'second', 'final',
                reuse_dirs=[root / 'first'], pin=pin, backend_factory=Backend)
            self.assertEqual(second['reused_video_ids'], ['final1', 'final2'])


if __name__ == '__main__':
    unittest.main()
