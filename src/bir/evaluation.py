"""Deterministic metrics; native agreement is not annotation quality."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Mapping, Sequence


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self):
        _finite_nonnegative(self.start, "start")
        _finite_nonnegative(self.end, "end")
        if self.end <= self.start:
            raise ValueError("Intervals must have positive duration")


def temporal_iou(a: Interval, b: Interval) -> float:
    """Temporal intersection divided by union; touching intervals score zero."""
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - overlap
    return overlap / union


def match_intervals(predictions: Sequence[Interval], references: Sequence[Interval],
                    threshold: float = 0.5,
                    semantic_match: Callable[[int, int], bool] | None = None) -> list[tuple[int, int]]:
    """Maximum-cardinality one-to-one matching at the declared IoU threshold.

    Ties are deterministic: predictions follow input order; neighbors follow
    descending overlap then reference index. Optional semantic decisions must
    come from a declared independent evaluator. This is not FLORA's WGO score.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    adjacency = []
    for i, prediction in enumerate(predictions):
        candidates = [(temporal_iou(prediction, reference), j) for j, reference in enumerate(references)]
        adjacency.append([j for overlap, j in sorted(candidates, key=lambda p: (-p[0], p[1]))
                          if overlap >= threshold and (semantic_match is None or semantic_match(i, j))])
    reference_owner: dict[int, int] = {}

    def assign(i: int, visited: set[int]) -> bool:
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in reference_owner or assign(reference_owner[j], visited):
                reference_owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        assign(i, set())
    return sorted((i, j) for j, i in reference_owner.items())


def interval_metrics(predictions: Sequence[Interval], references: Sequence[Interval],
                     threshold: float = 0.5,
                     semantic_match: Callable[[int, int], bool] | None = None) -> dict:
    matches = match_intervals(predictions, references, threshold, semantic_match)
    tp = len(matches)
    fp, fn = len(predictions) - tp, len(references) - tp
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / len(predictions) if predictions else None,
            "recall": tp / len(references) if references else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "matches": matches, "iou_threshold": threshold,
            "matching": "maximum_cardinality_one_to_one"}


def coverage(expected_ids: Iterable[str], completed_ids: Iterable[str], failed_ids: Iterable[str] = ()) -> dict:
    expected_list = list(expected_ids)
    if len(set(expected_list)) != len(expected_list):
        raise ValueError("Expected episode IDs must be unique")
    expected, complete, failed = set(expected_list), set(completed_ids), set(failed_ids)
    if (complete | failed) - expected:
        raise ValueError("Results contain unknown episode IDs")
    # A recovered earlier failure can coexist with a completed episode.
    unresolved = failed - complete
    missing = expected - complete - unresolved
    return {"expected": len(expected), "completed": len(complete), "failed": len(unresolved),
            "missing": len(missing), "completion_fraction": len(complete) / len(expected) if expected else None,
            "failed_ids": sorted(unresolved), "missing_ids": sorted(missing)}


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Linear interpolation between sorted observations (type 7 quantile)."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def performance_summary(episodes: Sequence[Mapping], wall_seconds: float, allocated_gpu_hours: float) -> dict:
    """Count unique fresh completed source time, once across synchronized views.

    Each row has episode_id, duration_seconds, completed, fresh_inference, and
    latency_seconds. Cached result retrieval contributes no inferred source time.
    Wall time is the measured condition boundary; GPU hours include every failed
    and idle allocation assigned to that condition.
    """
    _finite_nonnegative(wall_seconds, "wall_seconds")
    _finite_nonnegative(allocated_gpu_hours, "allocated_gpu_hours")
    completed: dict[str, float] = {}
    durations: dict[str, float] = {}
    latencies = []
    failures = cache_replays = 0
    for row in episodes:
        episode_id = row["episode_id"]
        duration = _finite_nonnegative(row["duration_seconds"], "duration_seconds")
        latency = _finite_nonnegative(row["latency_seconds"], "latency_seconds")
        if episode_id in durations and duration != durations[episode_id]:
            raise ValueError("One episode ID has conflicting source durations")
        durations[episode_id] = duration
        if not row["fresh_inference"]:
            cache_replays += 1
            continue
        if not row["completed"]:
            failures += 1
            continue
        if episode_id not in completed:
            completed[episode_id] = duration
            latencies.append(latency)
    source_hours = sum(completed.values()) / 3600
    return {"unique_completed_episodes": len(completed), "source_hours": source_hours,
            "wall_seconds": wall_seconds, "allocated_gpu_hours": allocated_gpu_hours,
            "source_hours_per_wall_hour": source_hours / (wall_seconds / 3600) if wall_seconds else None,
            "source_hours_per_allocated_gpu_hour": source_hours / allocated_gpu_hours if allocated_gpu_hours else None,
            "latency_median_seconds": percentile(latencies, 0.5), "latency_p95_seconds": percentile(latencies, 0.95),
            "failed_fresh_attempts": failures, "cached_retrievals": cache_replays,
            "latency_population": "first fresh successful attempt per unique episode"}


def cost_summary(records: Sequence[Mapping]) -> dict:
    """Sum measured compact/gate/fallback/audit/failed work without estimates.

    Each record has unique work_id, category, wall_seconds, allocated_gpu_hours.
    Allocation costs must be apportioned once; do not also supply their job total.
    Application time and allocation GPU hours are separate measurements.
    """
    seen = set()
    by_category: dict[str, dict[str, float]] = {}
    for row in records:
        if row["work_id"] in seen:
            raise ValueError("Duplicate cost record")
        seen.add(row["work_id"])
        category = str(row["category"])
        total = by_category.setdefault(category, {"wall_seconds": 0.0, "allocated_gpu_hours": 0.0})
        for name in total:
            total[name] += _finite_nonnegative(row[name], name)
    return {"by_category": by_category,
            "wall_seconds": sum(v["wall_seconds"] for v in by_category.values()),
            "allocated_gpu_hours": sum(v["allocated_gpu_hours"] for v in by_category.values()),
            "work_items": len(seen)}


def paired_quality(reference_scores: Mapping[str, float], candidate_scores: Mapping[str, float], *,
                   margin: float = 0.01, seed: int = 20261031, resamples: int = 10000,
                   minimum_episodes: int = 20) -> dict:
    """Episode-paired bootstrap of candidate minus reference mean score.

    The reference here is the fixed-budget comparator scored against independent
    annotation targets, not the native output used as a target. Scores must be in
    [0, 1]. Every frozen evaluation episode must appear in both mappings; failures
    receive the predeclared failure score instead of being dropped.
    """
    if set(reference_scores) != set(candidate_scores):
        raise ValueError("Paired evaluation requires identical episode coverage")
    if not 0 <= margin <= 1 or not math.isfinite(margin):
        raise ValueError("Quality margin must be in [0, 1]")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    if isinstance(minimum_episodes, bool) or not isinstance(minimum_episodes, int) or minimum_episodes < 2:
        raise ValueError("At least two episodes are required for uncertainty estimation")
    for score in (*reference_scores.values(), *candidate_scores.values()):
        _finite_nonnegative(score, "quality score")
        if score > 1:
            raise ValueError("Quality scores must be in [0, 1]")
    ids = sorted(reference_scores)
    differences = [candidate_scores[i] - reference_scores[i] for i in ids]
    observed = sum(differences) / len(differences) if differences else None
    result = {"episodes": len(ids), "mean_delta": observed, "margin": margin,
              "interval_95": None, "quality_verdict": "inconclusive",
              "method": "paired_episode_percentile_bootstrap", "seed": seed, "resamples": resamples}
    if len(ids) < minimum_episodes:
        result["reason"] = "Too few independent episodes for the predeclared pilot uncertainty rule"
        return result
    if max(differences) - min(differences) < 1e-12 and observed >= -margin:
        # A zero-width empirical interval cannot measure unseen rare losses.
        result["reason"] = "A degenerate paired sample cannot establish the risk of unseen quality losses"
        return result
    rng = random.Random(seed)
    means = [sum(rng.choice(differences) for _ in ids) / len(ids) for _ in range(resamples)]
    lower, upper = percentile(means, .025), percentile(means, .975)
    result["interval_95"] = [lower, upper]
    if lower >= -margin:
        result["quality_verdict"] = "within_margin"
    elif upper < -margin:
        result["quality_verdict"] = "outside_margin"
    else:
        result["reason"] = "Uncertainty crosses the allowed quality loss"
    return result


def adaptive_verdict(quality: Mapping, candidate_cost: float, best_fixed_cost: float, *,
                     complete_cost_accounting: bool, evaluator_frozen: bool) -> dict:
    """Resolve only the declared sample and metric; this is no general guarantee."""
    _finite_nonnegative(candidate_cost, "candidate_cost")
    _finite_nonnegative(best_fixed_cost, "best_fixed_cost")
    if not complete_cost_accounting or not evaluator_frozen or best_fixed_cost == 0:
        return {"verdict": "inconclusive", "reason": "Missing complete costs, frozen evaluation, or a nonzero comparator"}
    saving = 1 - candidate_cost / best_fixed_cost
    if saving <= 0:
        verdict = "unsupported"
    elif quality["quality_verdict"] == "outside_margin":
        verdict = "unsupported"
    elif quality["quality_verdict"] == "within_margin":
        verdict = "supported_on_declared_sample"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "cost_saving_fraction": saving, "quality": dict(quality)}
