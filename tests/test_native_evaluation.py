import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluate_native_annotations import (RUBRIC_SHA256, WGO, build_reference_contract,
                                                digest, gate_input_receipts, prepare, score)


class NativeEvaluationTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(id='one', source_sha256='a' * 64, instruction='', duration_s=1., references=[
            dict(start_s=0., end_s=1., caption=text) for text in ('Move a cup.', 'Move a book.')]),
            dict(id='missing', source_sha256='b' * 64, instruction='', duration_s=1., references=[
                dict(start_s=0., end_s=1., caption='Move a plate.')])]
        refs = dict(dataset=WGO.__dict__, records=rows)
        manifest = dict(schema_version='bir_wgo_preparation_v1', dataset=WGO.__dict__,
            selection_frozen=True, proposed_split='development', selected_count=2,
            selected_ids=['one', 'missing'], max_episodes=3,
            candidates=[dict(id=row['id'], selected=True, profile='human', proposed_split='development',
                native_supported=True, references_valid=True, source_sha256=row['source_sha256'],
                duration_s=1., reference_count=len(row['references'])) for row in rows])
        contract = build_reference_contract(manifest, refs, manifest_sha256='1' * 64, references_sha256='2' * 64)
        annotations = [dict(episode_id='one', input_sha256={'main': 'a' * 64}, duration_s=1.,
            profile='Human', camera_roles=['main'], segments=[
            dict(start_s=0., end_s=1., caption=text, status='complete')
            for text in ('Move a book.', 'Move a cup.')])]
        self.prepare_kwargs = dict(contract=contract,
            gate_receipt=dict(marker_sha256='3' * 64, annotation_artifacts=[dict(episode_id='one', sha256='4' * 64)],
                input_validation=[dict(source_sha256='a' * 64, status='inputs_validated', profile='Human',
                    yam32=False, duration_s=1., instruction_sha256=contract['episodes'][0]['instruction_sha256'])]),
            input_receipts=dict(initial_manifest_sha256='1' * 64, private_reference_input_sha256='2' * 64,
                annotation_files=[dict(episode_id='one', file_sha256='4' * 64, content_sha256=digest(annotations[0]))]))
        self.manifest = manifest
        cards, mapping = prepare(annotations, refs, **self.prepare_kwargs)
        reviews = []
        for index in range(2):
            ratings = []
            for card in cards:
                same = card['reference_caption'] == card['candidate_caption']
                ratings.append(dict(card_id=card['card_id'], action_match='yes',
                    object_match='yes' if same else 'no', result_match='not_applicable',
                    contradiction='no' if same else 'yes', unsupported_extra_claims=False,
                    rationale='The object agrees.' if same else 'The object differs.'))
            reviews.append(dict(review_id=str(index), evaluator_type='AI', evaluator_model='test fixture',
                rubric_sha256=RUBRIC_SHA256, new_paid_api_calls=0, ratings=ratings))
        return refs, annotations, cards, mapping, reviews

    def test_joint_matching_uses_semantic_edges_and_keeps_missing_episode(self):
        _, _, cards, mapping, reviews = self.fixture()
        self.assertEqual(len(cards), 4)
        result = score(mapping, reviews)
        # Re-matching accepted semantic edges gives two matches. Pruning the
        # initial temporal assignment would incorrectly retain zero.
        joint = result['aggregate']['joint_ai_reference_event']
        self.assertEqual(joint['true_positive'], 2)
        self.assertEqual(joint['false_negative'], 1)
        self.assertEqual(joint['micro_f1'], .8)
        self.assertEqual(joint['macro_f1'], .5)
        self.assertFalse(result['is_human_evaluation'])
        self.assertFalse(result['official_wgo_replication'])

    def test_source_identity_review_coverage_and_evidence_integrity(self):
        refs, annotations, _, mapping, reviews = self.fixture()
        annotations[0]['input_sha256']['main'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'source differ'):
            prepare(annotations, refs, **self.prepare_kwargs)
        bad = copy.deepcopy(reviews)
        bad[0]['ratings'].pop()
        with self.assertRaisesRegex(ValueError, 'omitted'):
            score(mapping, bad)
        mapping['iou_threshold'] = .1
        with self.assertRaisesRegex(ValueError, 'changed'):
            score(mapping, reviews)

    def test_exact_reference_content_and_receipts_are_frozen(self):
        refs, annotations, _, _, _ = self.fixture()
        changed = copy.deepcopy(refs)
        changed['records'][0]['references'][0]['caption'] = 'Replace the reference after seeing predictions.'
        with self.assertRaisesRegex(ValueError, 'references differ'):
            prepare(annotations, changed, **self.prepare_kwargs)
        changed = copy.deepcopy(self.prepare_kwargs)
        changed['input_receipts']['private_reference_input_sha256'] = 'not-a-file-checksum'
        with self.assertRaisesRegex(ValueError, 'input receipts are invalid'):
            prepare(annotations, refs, **changed)
        annotations[0]['segments'][0]['caption'] = 'Changed saved annotation.'
        with self.assertRaisesRegex(ValueError, 'Gate C artifact receipt'):
            prepare(annotations, refs, **self.prepare_kwargs)

    def test_freeze_rejects_final_or_unbounded_selections_and_changed_source_counts(self):
        refs, *_ = self.fixture()
        for field, value in [('proposed_split', 'final'), ('selection_frozen', False), ('selected_count', 4)]:
            changed = dict(self.manifest, **{field: value})
            with self.assertRaisesRegex(ValueError, 'initial development selection'):
                build_reference_contract(changed, refs, manifest_sha256='1' * 64, references_sha256='2' * 64)
        changed = copy.deepcopy(self.manifest)
        changed['candidates'][0]['reference_count'] = 1
        with self.assertRaisesRegex(ValueError, 'reference count differs'):
            build_reference_contract(changed, refs, manifest_sha256='1' * 64, references_sha256='2' * 64)

    def test_public_result_contains_receipts_and_flags_but_no_private_strings(self):
        _, _, cards, mapping, reviews = self.fixture()
        reviews[0]['ratings'][0]['unsupported_extra_claims'] = True
        reviews[0]['ratings'][0]['rationale'] = '/private/reviewer/rationale-do-not-publish'
        result = score(mapping, reviews)
        self.assertEqual(result['unsupported_extra_claim_cards'], 1)
        self.assertEqual(result['provenance'], mapping['provenance'])
        self.assertEqual(result['reviewers'][0]['review_content_sha256'], digest(reviews[0]))
        public = json.dumps(result)
        for card in cards:
            self.assertNotIn(card['reference_caption'], public)
            self.assertNotIn(card['candidate_caption'], public)
        self.assertNotIn('/private/reviewer', public)
        self.assertNotIn('rationale', public)

    def test_duplicate_extra_and_malformed_review_rows_fail(self):
        _, _, _, mapping, reviews = self.fixture()
        for mode in ('duplicate', 'extra', 'malformed'):
            changed = copy.deepcopy(reviews)
            if mode == 'duplicate':
                changed[0]['ratings'].append(copy.deepcopy(changed[0]['ratings'][0]))
            elif mode == 'extra':
                extra = dict(changed[0]['ratings'][0], card_id='unknown-card')
                changed[0]['ratings'].append(extra)
            else:
                changed[0]['ratings'][0]['action_match'] = 'invented-score'
            with self.assertRaises(ValueError):
                score(mapping, changed)

    def test_cli_freezes_once_and_rejects_changed_references_before_gate_or_cards(self):
        refs, *_ = self.fixture()
        script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_native_annotations.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, references, contract = [root / name for name in ('manifest.json', 'references.json', 'contract.json')]
            manifest.write_text(json.dumps(self.manifest))
            references.write_text(json.dumps(refs))
            freeze = [sys.executable, str(script), 'freeze', '--initial-manifest', str(manifest),
                      '--references', str(references), '--output', str(contract)]
            completed = subprocess.run(freeze, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            original = contract.read_bytes()
            repeated = subprocess.run(freeze, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(contract.read_bytes(), original)
            refs['records'][0]['references'][0]['caption'] = 'Changed private reference.'
            references.write_text(json.dumps(refs))
            (root / 'unused-annotations').mkdir()
            rejected = subprocess.run([sys.executable, str(script), 'prepare',
                '--initial-manifest', str(manifest), '--references', str(references),
                '--reference-contract', str(contract), '--annotations', str(root / 'unused-annotations'),
                '--gate-c-marker', str(root / 'missing-gate'), '--output-dir', str(root / 'cards')],
                capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('changed after the review freeze', rejected.stderr)
            self.assertFalse((root / 'cards').exists())

    def test_rehashed_mapping_cannot_change_frozen_references_or_graph_routes(self):
        _, _, _, mapping, reviews = self.fixture()
        for mode in ('reference', 'edge'):
            changed = copy.deepcopy(mapping)
            if mode == 'reference':
                changed['episodes'][0]['references'][0][1] = .9
            else:
                changed['edges'][0]['prediction_index'] = -1
            changed['mapping_sha256'] = digest({k: v for k, v in changed.items() if k != 'mapping_sha256'})
            with self.assertRaisesRegex(ValueError, 'frozen source|semantic edge'):
                score(changed, reviews)

    def test_all_missing_predictions_keep_every_reference_in_denominators(self):
        refs, _, _, _, reviews = self.fixture()
        inputs = copy.deepcopy(self.prepare_kwargs)
        inputs['input_receipts']['annotation_files'] = []
        cards, mapping = prepare([], refs, **inputs)
        self.assertEqual(cards, [])
        for review in reviews:
            review['ratings'] = []
        result = score(mapping, reviews)
        for metric in ('temporal', 'joint_ai_reference_event'):
            self.assertEqual(result['aggregate'][metric]['false_negative'], 3)
            self.assertEqual(result['aggregate'][metric]['micro_f1'], 0)
            self.assertEqual(result['aggregate'][metric]['macro_f1'], 0)

    def test_fresh_private_index_preserves_content_contract_with_new_file_receipts(self):
        refs, annotations, _, _, _ = self.fixture()
        original = self.prepare_kwargs['contract']
        regenerated = copy.deepcopy(refs)
        regenerated['private_path'] = '/another/private/extraction/path'
        regenerated['records'][0]['metadata'] = {'private_source_path': '/another/private/source'}
        current = build_reference_contract(self.manifest, regenerated,
            manifest_sha256='5' * 64, references_sha256='6' * 64)
        self.assertEqual(current['content_contract_sha256'], original['content_contract_sha256'])
        self.assertNotEqual(current['contract_sha256'], original['contract_sha256'])
        args = copy.deepcopy(self.prepare_kwargs)
        args['input_receipts'].update(initial_manifest_sha256='5' * 64, private_reference_input_sha256='6' * 64)
        _, mapping = prepare(annotations, regenerated, **args)
        self.assertEqual(mapping['provenance']['reference_contract'], original)
        self.assertEqual(mapping['provenance']['input_receipts']['private_reference_input_sha256'], '6' * 64)
        self.assertNotIn('/another/private', json.dumps(mapping))

    def test_gate_instruction_and_annotation_profile_duration_are_bound(self):
        refs, annotations, _, _, _ = self.fixture()
        changed = copy.deepcopy(self.prepare_kwargs)
        changed['gate_receipt']['input_validation'][0]['instruction_sha256'] = '9' * 64
        with self.assertRaisesRegex(ValueError, 'inference instruction'):
            prepare(annotations, refs, **changed)
        for key, value in [('duration_s', 2.), ('profile', 'YAM'), ('camera_roles', ['main', 'left_wrist'])]:
            output = copy.deepcopy(annotations)
            output[0][key] = value
            with self.assertRaisesRegex(ValueError, 'duration, profile, or camera roles'):
                prepare(output, refs, **self.prepare_kwargs)
        marker = dict(input_validation=[dict(input_sha256={'main': 'a' * 64}, status='inputs_validated',
            profile='Human', yam32=False, duration_s=1., instruction_sha256='b' * 64,
            private_video_path='/private/original/video.mp4', cameras={'private': 'not public'})])
        sanitized = gate_input_receipts(marker)
        self.assertEqual(sanitized[0]['source_sha256'], 'a' * 64)
        self.assertNotIn('/private', json.dumps(sanitized))
        self.assertNotIn('cameras', sanitized[0])


if __name__ == '__main__':
    unittest.main()
