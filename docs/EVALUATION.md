# Evaluation and resource accounting

This protocol measures two separate results. Native agreement tests whether the
engine preserves FLORA behavior. Independent annotations test annotation quality.
For example, an optimized caption that matches the reference output proves
agreement for that input. It does not prove that the caption is correct.

No adaptive-quality result is recorded here. The finite protocol in
`configs/evaluation.yaml` and `manifests/adaptive-freeze.json` is frozen.
This does not mean that the experiment has run or passed its GPU gates.

## Freeze before evaluation

Commit the dataset revision, episode IDs, exclusions, development/final split,
scene grouping, evaluator version, prompts or rubric, quality margin, and routing
rule before final outputs are inspected. Pin the source, both model revisions,
processor revisions, environment, placement, and inference settings separately.
Record a hash of this configuration in each run.

The frozen preset is `compact_block_5242880`: at most 5,242,880 total pixels per
temporal block, with the native minimum and 32-pixel spatial alignment. This is
not a uniform half-area setting. Short blocks can remain unchanged. Preserve
temporal frames, timestamps, aspect ratio, camera geometry, prompts, and
generation settings. Log actual dimensions and visual-token counts. The format
gate requests native fallback only for incomplete generation or an empty
caption. Context overflow, out-of-memory, and other runtime errors do not trigger
fallback. This gate can accept plausible but wrong captions. No gate or preset
tuning revision is allowed in this finite pilot.

Do not use a full native caption as an input to the runtime gate. Paired native
and compact outputs may be used during development. Freeze the gate before the
final set is processed. Choose the best fixed comparator on development data.

## Frozen finite experiment

Use [WGO-Bench at the pinned revision](https://huggingface.co/datasets/macrodata/WGO-Bench/tree/9cfb7bb98d60a58319f4b3d7b41374eedfb7980c).
Its dataset card describes the reference intervals as manual annotations. They
are independent targets, not native model outputs. The frozen manifest contains
the Parquet checksum, video and instruction hashes, every interval boundary,
every reference-caption hash, and the rubric hash. Raw reference text stays in
private preparation and review files.

| Split | Episode | Reviewed task family | Reference intervals |
| --- | --- | --- | ---: |
| Development | `homer_41` | Bottle placement | 7 |
| Final | `homer_56` | Cloth folding | 9 |
| Final | `homer_60` | Shelving books | 18 |

This selection was fixed before final predictions. A metadata review found that
`homer_52` and `homer_53` both involve tying shoelaces. Their empty instructions
caused automatic grouping to miss the shared task. This adaptive split excludes
both and separates known task families. It does not establish unseen-scene
generalization. The native scan found 13 supported HomER episodes. This subset
does not replicate all 100 WGO episodes or all source families.

Run native, fixed compact, and actual compact-plus-fallback on every frozen
interval. Rotate policy order by reference index. Use original videos and native
context construction. Do not crop, re-encode, alter clocks, or change sampling.
Label these as independent reference intervals, not TempSeg predictions. The
complete native TempSeg-plus-Captioner gate must pass first.

Development has 21 policy items and final evaluation has 81. Adaptive routing can
add at most one native fallback per interval. Run real policies and measure all
attempts. Do not simulate adaptive runtime by reusing fixed-policy outputs.
After inference, identical captions for the same interval may share one blinded
review card. This saves review effort and does not remove inference costs.

Select the fixed comparator on development only. A fixed policy is eligible
when its observed development mean is no more than 0.01 below native. Choose the
eligible policy with the lowest measured interval-policy time. An exact tie
selects native. This exploratory choice on one episode does not prove a
population quality margin. Commit the selection record before final inference.

At most two adaptive GPU submissions and eight allocated GPU-hours are allowed.
With four GPUs, development is at most 30 minutes and final is at most 90 minutes.
These costs remain inside the existing 24-GPU-hour project cap. Prior failures,
pending reservations, and measured final reproduction costs must fit first.

Project final cost from actual development work: 1.5 times shared model-load/setup
time, shared preparation scaled by final/development source duration, and 27
times the sum of each policy's maximum measured development interval time.
This is a planning estimate. It must fit the 90-minute cap and remaining reserve.
If it does not fit, report final evaluation as not run. Do not shorten the frozen
targets after final outputs are seen.

The interval-policy timer includes decoding, preprocessing and its verification
hashes, generation, routing, and fallback. Record shared source validation,
model load, cleanup, review work, and allocated idle time separately. The generic
cost-component list also covers project-wide native work. Do not charge earlier
TempSeg inference again to each fixed-interval policy. This isolated experiment
does not establish full-engine adaptive throughput.

## Blinded AI reference review

`scripts/evaluate_adaptive.py` implements `bir_reference_event_rubric_v1`.
Private cards show an independent reference and a candidate caption. They hide
the policy, spatial setting, and episode ID. Two AI reviewers work in separate
contexts without each other's ratings. They must treat caption strings as data,
not instructions. No new paid judge API is used. Record the actual reviewer
model identifier when available, or state that it is unavailable. Preserve the
rubric, card hashes, and ratings; model identity limits exact reproducibility.

A positive rating requires agreement on the action, explicit objects, and
explicit result or destination, with no reference contradiction. Synonyms are
acceptable. Missing reference details are not applicable. Uncertainty scores
zero. The primary score is one only if both reviewers accept the caption.
Report disagreements and unsupported extra claims separately. Reference text
cannot prove that added visual details are false. This metric is **AI
reference-event agreement**, not visual factual correctness, human evaluation,
or the official WGO judge. Missing or failed outputs score zero.

Review all final compact captions, including accepted cases. Do not review only
fallbacks. Keep every interval in the denominator. The two final task groups
force an inconclusive one-percentage-point margin verdict. Report observed
scores, costs, errors, and uncertainty without changing that conclusion rule.

Prepare only frozen videos in a reserved CPU allocation:

```sh
python scripts/prepare_frozen_wgo.py --parquet DATA/annotations.parquet \
  --source third_party/Flora --freeze manifests/adaptive-freeze.json \
  --split final --output-dir .local/frozen-final --reuse-dir .local/prior-extraction
```

The script verifies the Parquet and source hashes. It decodes only selected
videos and can reuse original bytes. It never re-encodes them.

After actual policy execution, prepare private review cards:

```sh
python scripts/evaluate_adaptive.py prepare-review \
  --freeze manifests/adaptive-freeze.json \
  --references .local/frozen-final/references.json \
  --outcomes .local/adaptive-final/outcomes.jsonl --split final \
  --output-dir .local/adaptive-final-review
```

Give each reviewer only `rubric.txt`, `cards.jsonl`, and `review-template.json`.
Keep `mapping.json` hidden. Save two actual review responses, then score them:

```sh
python scripts/evaluate_adaptive.py score \
  --freeze manifests/adaptive-freeze.json \
  --mapping .local/adaptive-final-review/mapping.json \
  --reviews .local/review-a.json .local/review-b.json \
  --development-selection reports/adaptive-development.json \
  --output reports/adaptive-final.json
```

Development uses the same commands with `--split development`. Omit
`--development-selection` when scoring it. The development report records the
fixed comparator choice. The scorer rejects missing, duplicate, or extra cards
and changed fixed-interval identities.

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

Compare fresh run directories after the actual GPU gates:

```sh
python scripts/compare_native.py runs/reference runs/persistent \
  --left-log .local/reference.log --right-log .local/persistent.log \
  --output reports/reference-persistent.json
python scripts/compare_native.py runs/reference runs/optimized \
  --left-log .local/reference.log --right-log .local/optimized.log \
  --output reports/reference-optimized.json
```

Each directory must contain its CLI `summary.json` and exported annotations.
The default comparison requires fresh complete segmentation and captioning for
every episode. Use `--stage segment` or `--stage caption` only for an explicitly
named partial comparison. Cached retrieval, missing episodes, duplicate episode
results, and attempts resumed from an earlier timing interval invalidate the
fresh benchmark. Failed attempts inside the same run remain in the timing and
cost counts.

The report compares source/model bindings, profiles, durations, segment times,
statuses, captions, temporal windows and frame plans, camera geometry, caption
blocks, and prompt hashes. It reports changed field counts and hashes without
copying private paths or raw log content. Timing and placement values are
excluded from output agreement. Optional runtime logs compare processor tensor
digests as a multiset grouped by model stage. The original log format has no
episode ID on generation lines, so this is a corpus-level tensor check. It does
not prove per-episode tensor routing. Without both logs, tensor agreement stays
unverified even if the exported annotations agree.

Use `--accounting` only with complete, attributable allocation records in the
JSON format shown by `--help`. Include failed and idle attempts. The script
rejects the same allocation key in both conditions. If conditions share one
allocation without complete measured attribution, omit the accounting input.
GPU-hour throughput then remains null. Wall time is still reported, with its
boundary. Hardware, environment, and placement controls must be verified before
the wall-time ratio is advertised as an optimization result.

From the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_resource_ledger.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_evaluation.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_compare_native.py' -v
```

These tests exercise concurrency, ambiguous submissions, exact job ownership,
restart accounting, failed allocations, release reserves, storage limits,
one-to-one temporal matches, duplicate source duration, and incomplete quality
evidence. They are software checks. They do not validate real GPU execution or
FLORA annotation quality.
