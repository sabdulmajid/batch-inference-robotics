# Evaluation and resource accounting

This protocol measures two separate results. Native agreement tests whether the
engine preserves FLORA behavior. Independent annotations test annotation quality.
For example, an optimized caption that matches the reference output proves
agreement for that input. It does not prove that the caption is correct.

No measured inference, performance, or adaptive-quality result is recorded in
this document. `configs/evaluation.yaml` is a protocol proposal with `frozen:
false`. It must not be reported as a completed experiment.

## Freeze before evaluation

Commit the dataset revision, episode IDs, exclusions, development/final split,
scene grouping, evaluator version, prompts or rubric, quality margin, and routing
rule before final outputs are inspected. Pin the source, both model revisions,
processor revisions, environment, placement, and inference settings separately.
Record a hash of this configuration in each run.

Use development episodes to choose the fixed compact budget and the adaptive
rule. The initial compact proposal uses half the native image pixel area. It
preserves temporal frames, timestamps, aspect ratio, camera geometry, prompts,
generation settings, and processor alignment. Log the actual image dimensions and
visual token counts. The initial gate requests native fallback when compact
parsing fails. This rule can miss plausible but wrong captions. Audit accepted
compact captions against independent references, including small objects,
occlusion, and short actions. At most one development revision is allowed.

Do not use a full native caption as an input to the runtime gate. Paired native
and compact outputs may be used during development. Freeze the gate before the
final set is processed. Choose the best fixed comparator on development data.

## Metrics and uncertainty

`bir.evaluation.interval_metrics` computes project temporal F1 at intersection
over union (IoU) 0.5. IoU is the shared interval duration divided by the combined
duration. Maximum-cardinality one-to-one matching prevents two predictions from
claiming the same reference. It also avoids loss from an early greedy match.
Matching order is deterministic. This is a project metric, not a claim to
reproduce the official WGO scoring protocol.

For joint temporal and semantic F1, supply independently evaluated semantic
acceptance for each candidate pair. For caption quality, compare captions on
fixed reference intervals. Select a reproducible evaluator before the final run.
Use independent human-provided dataset references as targets. Label all agent
reviews as AI evaluations. Exact caption string equality is not a factual
correctness metric. If no suitable evaluator is available, report quality as
unvalidated.

`paired_quality` accepts one score in [0, 1] for each episode in each condition.
Include every frozen episode. Use the declared failure score of zero for missing
or failed annotations. Do not drop failures to improve the average. Weight each
episode equally, so episodes with more intervals do not dominate the estimate.

The proposed acceptable loss is 0.01 in absolute mean episode score. A score
change from 0.80 to 0.79 is a loss of one percentage point. The paired bootstrap
resamples episodes with replacement. It reports the 2.5th and 97.5th percentiles
of 10,000 mean differences, with a fixed seed. Related clips must stay together
as one independent group. Pass group scores instead of separate clip scores
when multiple episodes share a scene or task instance.

Fewer than 20 independent groups produce an inconclusive margin decision under
this pilot protocol. Twenty groups is a predeclared minimum, not a guarantee of
enough statistical power. A narrow interval on a small or unrepresentative set
does not establish broad quality. If the lower interval bound is at least
-0.01, the sample supports the margin. If the upper bound is below -0.01, the
sample does not support it. Otherwise, the quality result is inconclusive.

If all observed paired differences are equal and within the margin, the
bootstrap would return a zero-width interval. The implementation reports
inconclusive in that case. A sample with no observed difference cannot measure
the risk of rare, unseen losses. Do not use an artificial zero-width interval
to certify the one-percentage-point margin.

`adaptive_verdict` requires complete cost accounting and a frozen evaluator.
The adaptive method must cost less than the selected fixed comparator and meet
the quality margin. A supported result applies only to the declared sample,
metric, hardware, and configuration. Insufficient evidence is inconclusive.
No completed adaptive run means **not run**, not an inconclusive experiment.

## Performance and cost

Compare the placement-adapted reference, persistent native execution, and fully
optimized native execution on the same inputs and hardware. Keep model placement,
source/model revisions, generation settings, and requested outputs equal.
Separate startup, first compilation, warm inference, and cached result retrieval.

`performance_summary` counts unique fresh completed source duration. Multiple
synchronized cameras count as one episode duration. A repeated completed episode
does not increase the numerator. Cached annotation retrieval does not count as
fresh model work. Record both throughput per wall hour and throughput per
allocated GPU hour. Four GPUs allocated for 15 minutes cost one GPU-hour, even
if three GPUs are idle or the application fails.

The wall interval must include all work assigned to that comparison. Report the
arrival/queue condition and exact timing boundaries. The library's latency
summary uses the first successful fresh attempt per episode. Report retries,
failed attempts, and total wall time separately. Do not use this successful-only
latency statistic to hide recovery cost. Use measured request and end-to-end
timestamps when queue-inclusive latency is required.

`cost_summary` includes measured decoding, segmentation, compact inference, gate
work, actual native fallbacks, runtime audits, failures, and retries. Unique work
IDs prevent duplicate records. It sums supplied application times, which can
overlap; this sum is not an end-to-end wall measurement. Supply allocation GPU
hours once per allocation or as nonoverlapping attributed portions. Never add
both an allocation total and its step totals. `MaxRSS` measures host RAM, not GPU
memory. Record GPU peaks separately for each device.

## Private resource ledger API

`bir.resource_ledger` uses only the Python standard library. It never submits or
cancels a job. The coordinator must keep its database on verified persistent
private storage with functioning file locks. All submitters must use the same
database and the same designated coordinator identity. A different identity or
changed resource policy fails closed. Keep the identity stable across sessions;
do not use a shared Unix username as proof of job ownership.

The coordinator workflow is:

1. Construct `ResourceLedger(path, policy_mapping, coordinator_id)` with the
   parsed `configs/resource_policy.yaml` mapping.
2. Create a `ResourceRequest` with explicit resources, phase, and provenance.
   Record the code commit, source/model revisions, environment fingerprint,
   configuration hash, input manifest, expected outputs, job name/comment, and
   private persistent log paths in `provenance`.
3. Call `reserve(run_id, request)` before submission. Reservations include full
   requested time. A GPU shard counts as one physical device in this budget.
4. Validate the changed request with `sbatch --test-only`. Call
   `begin_submission(run_id)` immediately before the real submission. This
   state blocks a second submission if the reply is lost.
5. Attach the exact allocation ID with `attach_job(run_id, job_id)`. If the reply
   is ambiguous, query scheduler state using the unique run name/comment. Do not
   retry first. `abandon_unsubmitted` requires evidence that no allocation exists.
6. Call `reject_restarted_execution()` at job startup before model work. Set
   `--no-requeue` in the script. If a restart still occurs, call `mark_restart`
   and reconcile every allocation attempt. A restart is not a new free budget.
7. Reconcile final allocation records with `reconcile(run_id, attempts)`. Use
   one `AllocationAttempt` per restart, starting at index zero. Include failed
   and idle elapsed time. Do not provide `.batch`, `.extern`, array, or duplicate
   step records. Use the generic GPU allocation count once; do not add it again
   to the typed count in `AllocTRES`.
8. Inspect `snapshot()`. Actual costs replace requested reservations after final
   reconciliation. Overspend is retained and prevents further work. Identical
   reconciliation is safe; changed final accounting requires explicit correction
   evidence and must not silently replace old records.

Before optional work, call `set_release_reserve(gpu_hours, cpu_core_hours, slots,
pilot_evidence)` using measured reproduction costs. This protects the reserve
and at least one final submission slot. The `release` phase may spend that
reserve. Call `assert_owned_job(job_id)` before cancellation. It returns the
recorded run ID or refuses an unrelated ID. The coordinator must also verify
the scheduler job identity; the ledger is not a scheduler or access control.

`check_storage` applies two independent checks: projected new-project usage must
stay at or below 250 GiB, and available storage must cover at least 120% of the
full remaining planned new project footprint. Include pending writes in that
plan. The user approved this project-sized headroom check on 2026-10-31. It
replaces the original whole-filesystem 20%-free rule. It does not increase the
250 GiB project cap or any compute limit. Use space available to the
account, such as `statvfs().f_bavail * f_frsize`. Verify mount health, write
permission, account quota, and free space before download. A successful capacity
check alone does not prove those requirements. Recheck before substantial writes
because other users can consume shared storage.

## Tests

From the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_resource_ledger.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_evaluation.py' -v
```

These tests exercise concurrency, ambiguous submissions, exact job ownership,
restart accounting, failed allocations, release reserves, storage limits,
one-to-one temporal matches, duplicate source duration, and incomplete quality
evidence. They are software checks. They do not validate real GPU execution or
FLORA annotation quality.
