# Adaptive caption experiment results

The development experiment completed all 21 planned items on three RTX 3090
GPUs. The compact policy took 7.94% less interval-policy time than native and
produced the same seven captions. This is one development episode. It does not
establish held-out quality or full-pipeline speedup.

Final evaluation was not admitted. The frozen cost check estimated 117.693
minutes for all 81 final items, above its 90-minute phase limit. No final GPU
job was submitted. The final targets, routing, margins, and limits remain
unchanged. The adaptive quality conclusion is **inconclusive**.

## Measured development result

The fixed intervals are seven independent dataset references from `homer_41`.
They are not TempSeg predictions. Policy order rotates by interval. Each policy
uses the same full source video, temporal evidence, prompt, native model, and
generation settings. The compact preset changes only the spatial pixel budget.

| Policy | Completed intervals | Full interval-policy time | Generation time, included at left | Input tokens | AI reference agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 7 | 264.754 s | 76.300 s | 57,449 | 3/7 |
| Compact | 7 | 243.728 s | 56.050 s | 40,865 | 3/7 |
| Adaptive | 7 | 243.608 s | 56.093 s | 40,865 | 3/7 |

The full policy timer includes decoding, image transport, processing, generation,
and gate or fallback work. Generation time is nested; do not add it again.
Shared setup, model loading, preparation, and cleanup took another 100.229 s.
The application took 856.621 s. The complete allocation took 859 s and used
0.715833 GPU-hours. That allocation is counted once for all policies and all
overhead. Per-policy allocation shares were not measured.

All 21 items completed without a failure or retry. All seven adaptive gates
accepted the compact output; no native fallback ran. The small adaptive versus
compact time difference does not demonstrate a separate speed benefit.

Paired frame indices, source timestamps, and prompt audits match native.
Compact and adaptive input tensors match each other for every interval. The
compact policy reduced seven target blocks and kept 14 context blocks unchanged.
Changed block areas were 47.62% or 75.40% of native. Temporal grid sizes stayed
unchanged. These are actual processor and input checks, not a claim that all
future inputs retain quality.

## Quality review and selection

Two new AI reviewers each assessed seven blinded cards. Identical captions for
the same interval share a card; all 21 policy items and their costs remain in
the denominator. Both reviewers accepted three of seven reference events.
They had no binary score disagreements. Both exact model identifiers were
unavailable. Their separately measured review times total 103.568 s and can
overlap in wall time. No new paid API call was made.

All seven cards had added-claim flags. These indicate details that the text
reference cannot verify; they do not prove visual errors. This is AI text
reference agreement, not human annotation or visual ground truth.

The frozen rule selected **compact** as the fixed comparator. Its score matched
native, and it had the lowest measured time among eligible fixed policies.
The adaptive format gate also accepted captions with zero reference-agreement
scores. A nonempty caption is not a semantic quality check.

## Why final evaluation did not run

The existing admission function uses the largest measured interval cost for
each policy. For adaptive, it allows a compact pass plus a native fallback,
even though development observed no fallback. It adds scaled preparation and
shared setup, then applies the frozen 1.5 headroom factor:

```text
1.5 × (92.4845 + 18.0287 + 27 × (44.1720 + 40.9614 + 85.1334))
= 7,061.573 seconds > 5,400 seconds
```

This is a conservative planning estimate, not measured final runtime. The
project-wide GPU budget still had room, but it does not override the frozen
phase limit. The runner rejects this projection. No targets were removed and
no rule was tuned to obtain admission. Even a completed final set would have
only two task groups, too few to establish the one-percentage-point population
quality margin.

## Evidence and reproduction

- [Exact scored development comparator](../reports/adaptive-development.json)
- [Execution, fidelity, cost, and admission evidence](../reports/adaptive-experiment.json)
- [Frozen protocol and source identities](../manifests/adaptive-freeze.json)
- [Runner commands and timing boundaries](ADAPTIVE.md#bounded-runner)

Inference used immutable runtime `d2701b0`. Use the pinned environment and
prepared source manifests described in [reproduction instructions](REPRODUCTION.md).
Keep videos, raw outcomes, references, review cards, and reviewer responses in
private project storage.

After development inference, prepare blinded cards and score two separate
reviews with the existing evaluator:

```sh
python scripts/evaluate_adaptive.py prepare-review \
  --freeze manifests/adaptive-freeze.json --split development \
  --references "$DEV_REFERENCES" --outcomes "$DEV_RUN/outcomes.jsonl" \
  --output-dir "$DEV_REVIEW"
python scripts/evaluate_adaptive.py score \
  --freeze manifests/adaptive-freeze.json --mapping "$DEV_REVIEW/mapping.json" \
  --reviews "$REVIEW_ONE" "$REVIEW_TWO" --output "$DEV_SELECTION"
```

Reviewers receive only the rubric, cards, and response template. Commit the
reviewed comparator before any admitted final inference. Apply the unchanged
`final_projection` function from `scripts/adaptive_benchmark.py` to that
selection and the complete development receipt before reserving final GPUs.
For the recorded evidence, rejection is the expected result.
