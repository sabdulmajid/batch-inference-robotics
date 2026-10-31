"""CPU tests for frozen adaptive execution guards, identity, and cost limits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.adaptive_benchmark import (POLICIES, captioner_placement_signature, digest, export_outcomes, final_projection, item_key,
                                         policy_items, verify_outcome)
from scripts.evaluate_adaptive import outcome_evidence, read_outcomes


class AdaptiveBenchmarkTests(unittest.TestCase):
    def test_atomic_outcome_export_is_consumed_by_the_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'outcomes.jsonl'
            rows = {'second': {'episode_id': 'two', 'caption': 'line one\nline two'},
                    'first': {'episode_id': 'one', 'status': 'failed'}}
            export_outcomes(path, rows)
            self.assertEqual(read_outcomes([path]), [rows[key] for key in sorted(rows)])
            export_outcomes(path, {'first': rows['first']})
            self.assertEqual(read_outcomes([path]), [rows['first']])

    def fixture(self, root, seconds=1.0):
        freeze = dict(freeze_sha256='a' * 64,
            protocol={'adaptive': {'final_job_minutes_ceiling': 90}}, episodes=[])
        for split, name, count, duration in [('development', 'dev', 2, 10), ('final', 'heldout', 3, 20)]:
            freeze['episodes'].append(dict(episode_id=name, split=split, source_sha256=name + '-hash',
                duration_s=duration, targets=[dict(reference_index=i, start_s=float(i), end_s=float(i + 1))
                                              for i in range(count)]))
        contract = dict(base_engine={'code': 'pinned'}, script_sha256={'runner': 'pinned'},
                        freeze_sha256=freeze['freeze_sha256'], captioner_placement_signature='c' * 64)
        development = dict(status='complete', split='development', freeze_sha256=freeze['freeze_sha256'],
            contract=contract, contract_sha256='b' * 64, validated_captioner_placement_signature='c' * 64,
            cumulative_shared_work_seconds={
                'model_load': 4, 'setup': 2, 'preparation': 3, 'cleanup': 1})
        selection = dict(schema_version='bir_adaptive_review_result_v1', split='development',
            freeze_sha256=freeze['freeze_sha256'], best_fixed_policy='native',
            policy_metrics={p: {'policy_application_seconds': 2 * seconds} for p in POLICIES})
        directory = root / 'outcomes'
        directory.mkdir()
        rows = []
        for episode_id, index, policy, target in policy_items(freeze, 'development'):
            row = dict(execution_contract_sha256=development['contract_sha256'],
                freeze_sha256=freeze['freeze_sha256'], episode_id=episode_id,
                reference_index=index, policy=policy, start_s=target['start_s'], end_s=target['end_s'],
                input_sha256={'main': target['episode']['source_sha256']}, policy_wall_seconds=seconds,
                status='complete', caption='fixture', attempts=[])
            row['result_sha256'] = digest(row)
            rows.append(row)
            (directory / (item_key(episode_id, index, policy) + '.json')).write_text(json.dumps(row))
        selection.update(outcome_evidence(rows))
        (root / 'selection.json').write_text(json.dumps(selection))
        (root / 'run-summary.json').write_text(json.dumps(development))
        return freeze, contract, development, selection

    def test_policy_order_retains_every_reference_and_rotates(self):
        with tempfile.TemporaryDirectory() as temporary:
            freeze, *_ = self.fixture(Path(temporary))
            items = list(policy_items(freeze, 'development'))
            self.assertEqual([i[2] for i in items], ['native', 'compact', 'adaptive', 'compact', 'adaptive', 'native'])
            self.assertEqual(len({item_key(*i[:3]) for i in items}), 6)

    def test_resume_checks_signed_measured_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze, _, development, _ = self.fixture(root)
            episode_id, index, policy, target = next(policy_items(freeze, 'development'))
            path = root / 'outcomes' / (item_key(episode_id, index, policy) + '.json')
            args = dict(contract_sha256=development['contract_sha256'], freeze_sha256=freeze['freeze_sha256'],
                        episode_id=episode_id, index=index, policy=policy, target=target)
            original = verify_outcome(path, **args)
            self.assertEqual(original['policy_wall_seconds'], 1.0)
            with self.assertRaisesRegex(ValueError, 'another input'):
                verify_outcome(path, **dict(args, policy='adaptive'))
            original['caption'] = 'edited'
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_outcome(path, **args)

    def test_final_projection_counts_maxima_all_policies_and_shared_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze, contract, *_ = self.fixture(root)
            result = final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)
            # Adaptive projection allows compact1 + native fallback1, even
            # though all observed adaptive policies took only one second.
            self.assertEqual(result['projected_seconds'], 1.5 * (7 + 6 + 12))
            with self.assertRaisesRegex(ValueError, 'Code or model'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json',
                                 dict(contract, base_engine={'changed': True}))

    def test_final_rejects_missing_costs_interrupted_evidence_and_excess_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze, contract, development, selection = self.fixture(root, seconds=1000)
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)
            development['unmeasured_interrupted_invocations'] = 1
            (root / 'run-summary.json').write_text(json.dumps(development))
            with self.assertRaisesRegex(ValueError, 'completed measured'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)
            development['unmeasured_interrupted_invocations'] = 0
            (root / 'run-summary.json').write_text(json.dumps(development))
            selection['policy_metrics']['compact']['policy_application_seconds'] = 0
            (root / 'selection.json').write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'costs differ'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)

    def test_final_rejects_changed_outcomes_and_uncommitted_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze, contract, development, selection = self.fixture(root)
            development['uncommitted_item_attempts'] = 1
            (root / 'run-summary.json').write_text(json.dumps(development))
            with self.assertRaisesRegex(ValueError, 'completed measured'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)
            development['uncommitted_item_attempts'] = 0
            (root / 'run-summary.json').write_text(json.dumps(development))
            path = next((root / 'outcomes').glob('*.json'))
            changed = json.loads(path.read_text())
            changed['caption'] = 'changed with a new checksum'
            changed['result_sha256'] = digest({k: v for k, v in changed.items() if k != 'result_sha256'})
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'different measured outcome'):
                final_projection(freeze, root / 'selection.json', root / 'run-summary.json', contract)

    def test_placement_signature_changes_with_actual_parameter_distribution(self):
        residency = {'captioner': {'all_parameters_and_nonempty_buffers_on_cuda': True,
                     'parameter_bytes_by_device': {'cuda:0': 10, 'cuda:1': 20},
                     'parameter_dtypes': {'torch.bfloat16': 2}, 'buffer_devices': {}, 'device_map': {'': '0'}}}
        first = captioner_placement_signature(residency)
        residency['captioner']['parameter_bytes_by_device'] = {'cuda:0': 20, 'cuda:1': 10}
        self.assertNotEqual(first, captioner_placement_signature(residency))
        residency['captioner']['all_parameters_and_nonempty_buffers_on_cuda'] = False
        with self.assertRaisesRegex(ValueError, 'verified per-device'):
            captioner_placement_signature(residency)

    def test_restarted_job_stops_before_source_or_framework_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run([sys.executable, 'scripts/adaptive_benchmark.py',
                '--source-root', str(root / 'missing'), '--manifest', str(root / 'missing.jsonl'),
                '--gate-evidence', str(root / 'missing-gate.json'), '--split', 'development',
                '--output-dir', str(root / 'run')], capture_output=True, text=True,
                env=dict(os.environ, SLURM_RESTART_COUNT='1'))
            self.assertEqual(result.returncode, 1)
            summary = json.loads((root / 'run/run-summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('reconcile', summary['error'])
            self.assertEqual(summary['fresh_items'], [])
            self.assertEqual(summary['gpu_memory'], [])


if __name__ == '__main__':
    unittest.main()
