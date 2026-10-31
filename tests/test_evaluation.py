import unittest

from bir.evaluation import Interval, adaptive_verdict, cost_summary, coverage, interval_metrics, match_intervals, paired_quality, performance_summary


class EvaluationTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_repairs_greedy_choice(self):
        # P0 can match either reference; P1 can match only R0. Greedy matching
        # can lose the second match. Maximum matching must recover both.
        predictions = [Interval(0, 3), Interval(0, 1)]
        references = [Interval(0, 2), Interval(1, 3)]
        self.assertEqual(match_intervals(predictions, references, .5), [(0, 1), (1, 0)])
        duplicated = interval_metrics([Interval(0, 1)] * 2, [Interval(0, 1)])
        self.assertEqual(duplicated["true_positive"], 1)
        self.assertEqual(duplicated["false_positive"], 1)

    def test_missing_predictions_count_as_false_negatives_and_empty_is_not_perfect(self):
        result = interval_metrics([], [Interval(0, 1)])
        self.assertEqual(result["f1"], 0)
        self.assertEqual(result["false_negative"], 1)
        self.assertIsNone(interval_metrics([], [])["f1"])
        self.assertEqual(coverage(["a", "b", "c"], ["a"], ["b"])["missing_ids"], ["c"])

    def test_joint_matching_requires_independent_semantic_acceptance(self):
        intervals = [Interval(0, 1), Interval(2, 3)]
        result = interval_metrics(intervals, intervals, semantic_match=lambda i, j: i == j == 0)
        self.assertEqual(result["f1"], .5)

    def test_source_hours_ignore_cache_replays_and_camera_or_retry_duplicates(self):
        rows = [dict(episode_id="a", duration_seconds=3600, completed=True, fresh_inference=True, latency_seconds=20),
                dict(episode_id="a", duration_seconds=3600, completed=True, fresh_inference=True, latency_seconds=20),
                dict(episode_id="b", duration_seconds=3600, completed=True, fresh_inference=False, latency_seconds=.1),
                dict(episode_id="c", duration_seconds=3600, completed=False, fresh_inference=True, latency_seconds=30)]
        result = performance_summary(rows, 120, 4 / 30)
        self.assertEqual(result["source_hours"], 1)
        self.assertEqual(result["source_hours_per_wall_hour"], 30)
        self.assertEqual(result["source_hours_per_allocated_gpu_hour"], 7.5)
        self.assertEqual(result["failed_fresh_attempts"], 1)

    def test_adaptive_cost_counts_fallback_audit_and_failed_work(self):
        records = [dict(work_id=str(i), category=category, wall_seconds=seconds, allocated_gpu_hours=seconds / 900)
                   for i, (category, seconds) in enumerate([("compact", 10), ("gate", 1), ("fallback", 20), ("audit", 5), ("failed", 4)])]
        self.assertEqual(cost_summary(records)["wall_seconds"], 40)
        with self.assertRaises(ValueError):
            cost_summary(records + records[:1])

    def test_small_pilot_cannot_establish_one_percent_margin(self):
        result = paired_quality({"a": 1, "b": 1}, {"a": 1, "b": 1})
        self.assertEqual(result["quality_verdict"], "inconclusive")
        verdict = adaptive_verdict(result, 1, 2, complete_cost_accounting=True, evaluator_frozen=True)
        self.assertEqual(verdict["verdict"], "inconclusive")

    def test_paired_results_require_identical_coverage_and_report_degradation(self):
        with self.assertRaises(ValueError):
            paired_quality({"a": 1}, {"b": 1})
        reference = {str(i): 1 for i in range(20)}
        candidate = {str(i): .9 for i in range(20)}
        result = paired_quality(reference, candidate, resamples=100)
        self.assertEqual(result["quality_verdict"], "outside_margin")
        verdict = adaptive_verdict(result, 1, 2, complete_cost_accounting=True, evaluator_frozen=True)
        self.assertEqual(verdict["verdict"], "unsupported")

    def test_identical_pilot_scores_do_not_prove_a_narrow_population_margin(self):
        scores = {str(i): 1 for i in range(100)}
        result = paired_quality(scores, scores)
        self.assertEqual(result["mean_delta"], 0)
        self.assertEqual(result["quality_verdict"], "inconclusive")
        self.assertIsNone(result["interval_95"])


if __name__ == "__main__":
    unittest.main()
