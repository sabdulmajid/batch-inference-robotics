"""Independent evidence-review regressions; these tests do not run inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import evaluate_adaptive as evaluator


def fixture():
    references, candidates = [], []
    for identity in ('development', 'final-a', 'final-b'):
        source = hashlib.sha256(identity.encode()).hexdigest()
        references.append(dict(id=identity, source_sha256=source, instruction='', references=[
            dict(start_s=float(i), end_s=float(i + 1), caption='Put the cup on the table.')
            for i in range(2)]))
        candidates.append(dict(id=identity, source_sha256=source, duration_s=2., group_id=identity,
                               native_supported=True, references_valid=True, reference_count=2))
    protocol = dict(frozen=True, maximum_adaptive_development_revisions=0,
        development_ids=['development'], final_ids=['final-a', 'final-b'],
        reviewed_task_groups={row['id']: row['id'] for row in references})
    freeze = evaluator.create_freeze(dict(candidates=candidates, dataset={'fixture': True}), references, protocol)
    rows = []
    for (episode_id, index), target in evaluator.targets_for(freeze, 'final').items():
        for policy in evaluator.POLICIES:
            row = dict(schema_version='bir_reference_caption_v1', episode_id=episode_id,
                reference_index=index, policy=policy, freeze_sha256=freeze['freeze_sha256'],
                execution_contract_sha256='c' * 64, input_sha256={'main': target['episode']['source_sha256']},
                start_s=target['start_s'], end_s=target['end_s'], status='complete',
                caption='Place the cup on the table.', policy_wall_seconds=2.,
                attempts=[dict(category=policy, status='complete', wall_seconds=2.)])
            row['result_sha256'] = evaluator.digest(row)
            rows.append(row)
    return freeze, references, rows


class AdaptiveReviewRegressions(unittest.TestCase):
    def test_early_rejected_invocation_preserves_completed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'existing-run'
            run.mkdir()
            index = run / 'outcomes.jsonl'
            summary = run / 'run-summary.json'
            index.write_text('{"preserved":"outcome index"}\n')
            summary.write_text('{"status":"complete","preserved":"last accepted run"}\n')
            before = {path: path.read_bytes() for path in (index, summary)}
            result = subprocess.run([sys.executable, 'scripts/adaptive_benchmark.py',
                '--source-root', str(root / 'unused'), '--manifest', str(root / 'unused.jsonl'),
                '--gate-evidence', str(root / 'unused-gate.json'), '--split', 'final',
                '--output-dir', str(run), '--resume'], capture_output=True, text=True,
                env=dict(os.environ, SLURM_RESTART_COUNT='1'))
            self.assertNotEqual(result.returncode, 0)
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload, path.name)
            receipts = sorted(run.glob('invocation-*.json'))
            self.assertEqual(len(receipts), 1)
            failed = json.loads(receipts[0].read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['fresh_items'], [])

    def test_review_rejects_an_edited_caption_with_a_stale_result_checksum(self):
        freeze, references, rows = fixture()
        rows[0]['caption'] = 'A different action from another run.'
        with self.assertRaisesRegex(ValueError, 'checksum|digest'):
            evaluator.prepare_review(freeze, references, rows, 'final')

    def test_review_dedup_retains_interval_identity_and_failure_denominators(self):
        freeze, references, rows = fixture()
        # One failed and one missing item must not remove their frozen intervals.
        rows[0].update(status='failed', caption=None)
        rows[0]['result_sha256'] = evaluator.digest({k: v for k, v in rows[0].items() if k != 'result_sha256'})
        rows.pop(1)
        cards, mapping = evaluator.prepare_review(freeze, references, rows, 'final')
        self.assertEqual(len(cards), 4)  # Equal text, but four distinct reference intervals.
        self.assertEqual(len(mapping['mapping']), 12)
        failures = [row for row in mapping['mapping'] if row['card_id'] is None]
        self.assertEqual(len(failures), 2)
        self.assertEqual({row['status'] for row in failures}, {'failed', 'missing'})
        self.assertTrue(all(row['automatic_failure_score'] == 0 for row in failures))
        self.assertTrue(all('policy' not in card and 'episode_id' not in card for card in cards))
        by_target = {}
        for row in mapping['mapping']:
            by_target.setdefault((row['episode_id'], row['reference_index']), set())
            if row['card_id']:
                by_target[(row['episode_id'], row['reference_index'])].add(row['card_id'])
        self.assertTrue(all(len(identities) == 1 for identities in by_target.values()))
        self.assertEqual(len(set.union(*by_target.values())), 4)


if __name__ == '__main__':
    unittest.main()
