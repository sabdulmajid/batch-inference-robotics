# Review native annotations against independent references

This bounded review measures temporal matching and AI reference-event agreement
on the three initial development episodes. It uses 40 independent WGO reference
intervals. It is not a held-out test, human evaluation, visual factual accuracy,
or a reproduction of the official WGO scoring method.

The completed development review found temporal micro F1 of **0.6118** and joint
temporal/AI reference-event micro F1 of **0.0941**. The joint rule produced four
one-to-one matches from 45 predictions and 40 references. The two reviewers
agreed on acceptance for all 27 review cards; five accepted cards produced those
four matches. They flagged added details on all cards that the reference text
could not verify. These flags do not prove visual errors. Exact reviewer model
identifiers were unavailable. The low observed joint agreement applies to this
small development sample and fixed rubric. See the
[sanitized result](../reports/native-reference-review.json) for counts, timing,
provenance, and limits. The rubric and metrics were not changed after scoring.

The [reference contract](../manifests/native-development-review.json) was saved
before Captioner execution. It fixes the selected source videos, dataset revision,
instructions, reference intervals and caption hashes, rubric, and temporal IoU
threshold of 0.5. IoU is the overlap duration divided by the combined duration.
For example, an interval from 1 to 3 seconds and one from 2 to 4 seconds have
IoU 1/3 and do not match at this threshold.

The preparation command requires a complete native Gate C record and matching
annotation file hashes. It makes a private review card for every prediction and
reference pair that meets the temporal threshold. Two separate AI reviewers see
the fixed rubric and cards. They do not see model conditions or each other's
ratings. Missing captions receive no accepted semantic edges.

Temporal F1 uses maximum-cardinality one-to-one matching. Joint F1 uses the same
matching rule after retaining only pairs that both AI reviewers accept. It
recomputes the matching on that graph. It does not remove rejected pairs from
an earlier temporal assignment. Each prediction and reference counts at most
once. Missing expected episodes remain in the denominator. Reports include
both aggregate counts and the mean of episode F1 scores.

Keep reference text, cards, mappings, raw annotations and reviewer responses in
private task storage. Review the resulting score report before publication.
The report contains hashes, counts and scores, without raw caption text.

```bash
python scripts/evaluate_native_annotations.py prepare \
  --annotations runs/native/annotations \
  --references runs/prepared/private-index.json \
  --initial-manifest manifests/initial-native.json \
  --reference-contract manifests/native-development-review.json \
  --gate-c-marker runs/native-caption-probe/probe-summary.json \
  --output-dir runs/native-review

python scripts/evaluate_native_annotations.py score \
  --mapping runs/native-review/mapping.json \
  --reviews runs/native-review/reviewer-1.json runs/native-review/reviewer-2.json \
  --output runs/native-review/scores.json
```

Use fresh reviewers with only the saved rubric, cards, and response schema.
Record the actual model identifier, or explicitly state when it is unavailable.
Record each review's elapsed time. No new paid API calls are permitted. The
rubric and rating fields are shared with [the adaptive review](EVALUATION.md).

The contract records original receipt hashes for provenance. Portable validation
uses the exact selected content, so a newly prepared index can use a different
private storage path. Its actual receipt hash is recorded for that reproduction.
None of these checks establish population-level accuracy from three episodes.
