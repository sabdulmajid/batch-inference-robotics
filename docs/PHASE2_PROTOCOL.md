# Phase-two native acceleration protocol

This phase measures a new native execution path against the previous optimized
engine. It also adds a practical annotation and review workflow. Performance is
the primary experiment. Native agreement measures compatibility, not useful or
visually correct labels.

The previous release is `e51aaa723a5de9e5e532e178e4041a78e335f207`. Its measured
inference modules are pinned at `d2701b08a08b88d18887abed81f71c8373478d7b`.
The baseline condition is `optimized`; the new condition is `accelerated`.
The old native conditions and the phase-one adaptive result remain available.

## Finite resource plan

The project retains its original 24 aggregate GPU-hour and 16 GPU-submission
limits. Phase one used 8.773056 GPU-hours and eleven submissions. The first
phase-two profile submission was cancelled while pending after an input-path
error was found. It used zero GPU-hours but consumed one submission. Four
submissions remain at this observation; they are not a new allowance.

| Remaining allocation | Purpose | Limit before admission |
| --- | --- | --- |
| 1 | One original development episode under the immutable baseline profiler, then the accelerated original-three-episode pilot without profiling | At most 60 minutes, three GPUs |
| 2 | Frozen block 0, baseline then candidate | Measured projection; at most 120 minutes, three GPUs |
| 3 | Frozen block 1, candidate then baseline | Measured projection; at most 120 minutes, three GPUs |
| 4 | Fresh final reproduction and recovery check | Protect at least 1.5 GPU-hours and one slot first; coordinator sizes the final request |

These are maximum admission limits, not four simultaneous reservations. The
coordinator reconciles each completed parent allocation before sizing later
ones. CPU-only source extraction has a separate reservation under the existing
32-core-hour cap. Every allocation uses the existing ledger. Requeues and
automatic experiment retries remain disabled. No optional retry slot exists.

The final comparison is admitted only after a working accelerated pilot and an
unchanged native compatibility check. The cost estimate uses the accepted
unprofiled v1 optimized baseline (1,328.396727 seconds for 138.767089 source
seconds) and the fresh unprofiled accelerated pilot. The separately profiled
episode does not supply a comparison timing or a cost rate.

For each frozen block, reserve the ceiling in minutes of
`1.2 × block_source_seconds × (baseline_rate + candidate_rate) + 120 seconds`.
Rates are measured whole-workflow seconds per source second. The 1.2 factor and
120-second controller allowance are explicit estimates. Both block requests
must fit within 120 minutes and the remaining aggregate allowance after the
reproduction reserve. Profiling, preparation, failed work, and idle allocated
time remain charged. A failed projection does not authorize a smaller workload
after final predictions have begun.

## Source-only selection, before final predictions

[`manifests/phase2-native.json`](../manifests/phase2-native.json) freezes the
source byte hashes, complete durations, geometry, profile, task labels, empty
instructions, and two execution orders. The pinned WGO index contains fourteen
native-supported human episodes. Three were used for development. The remaining
eleven are all included, with 726.897933 unique source seconds. This falls below
the requested 12–20 episode range because only eleven unseen episodes are
available in the validated pinned pool. No development repeat is relabeled as
an unseen example.

The block assignment enumerates all 462 five-episode subsets and their
six-episode complements. Each block must contain at least three source task
labels. The selected partition minimizes the absolute source-duration
difference using exact decimal arithmetic. Ties use the SHA256 of the canonical
JSON value `["phase2-balance-v2", sorted_five_episode_ids]`. The unchanged fixed
hash rule orders episodes inside each block. Task labels come from source
instructions and source-video metadata, without inspecting model predictions.
Both engines receive the same empty instruction and original single-view video.

This source-only scheduling change was made before any phase-two GPU profile,
pilot, or final prediction. The earlier greedy assignment differed by
46.514999 source seconds between blocks. The exact partition reduces that
difference to 0.336067 seconds. The manifest records its predecessor digest.
Every episode row, the five/six counts, and the AB/BA orders remain unchanged.
The 1.2 cost margin, controller allowance, and resource limits are unchanged.
The original extracted videos remain valid; only block manifests need rewriting.

| Block | Fixed episode order | Source seconds | Condition order |
| --- | --- | ---: | --- |
| 0 | 59, 60, 29, 48, 38 | 363.617000 | v1 optimized, accelerated |
| 1 | 39, 56, 33, 53, 37, 40 | 363.280933 | accelerated, v1 optimized |

The numeric episode suffixes refer to `homer_` IDs. The source tasks include
chopping, table setting, surface cleaning, container stacking, shoelaces, cloth
handling, and shelf actions. Duration and image size describe input variation;
they are not validated measures of visual difficulty. Episode 39 has valid
native input but invalid independent reference boundaries. It remains in the
performance denominator. This phase does not propose a text-reference quality
score for it.

Each GPU process handles five or six episodes. Thus this design alone cannot
measure the old eight-entry metadata-cache failure. A separate twelve-item CPU
regression must expose that failure and verify the repair. The broader corpus
and that regression answer different questions.

## Matching and measurement boundaries

Run both versions in separate fresh subprocesses with the same common worker.
Use three same-node RTX 3090 GPUs, four CPUs, 128 GiB host RAM, the pinned native
environment, unchanged model revisions/dtypes, four GiB placement headroom,
batch one, and the same complete source inputs. Compare actual placement,
device capacities, and dependency versions with the admitted pilot and within
each pair. No completed annotation or stage result can enter a fresh trial.

Full native input hashing and parameter checks stay enabled in both versions.
Optional detailed profiling stays disabled. The baseline receives lightweight
episode/stage/source/request-ordinal markers from the external harness. The
candidate has equivalent built-in markers. This does not modify immutable
baseline files or any generation argument. Input tensor digests must match by
episode, stage, and ordinal, as well as in the existing native comparison.

Primary time is parent-observed subprocess wall time. It includes imports,
environment and input checks, model loading, execution, output writing, and
worker cleanup. The engine and CLI timers have narrower, nested boundaries;
do not add them. Keep CPU work sums, generation timers, and input preparation
separate because some intervals overlap.

Report full cold-model-load time only when both versions supply a complete
successful load event for each model. Also report the derived load-excluded
workflow throughput as a sustained-cost diagnostic. It retains other startup
overhead and can subtract a load interval that overlapped CPU preparation. It
is not an independently timed steady-state trial or the primary speedup.

For each version, report subprocess time, unique source duration, source minutes
per wall minute, peak allocated/reserved GPU bytes, cache/preparation work,
stage attempts, failures, and native agreement. Report GPU-seconds per source
minute as `visible_GPU_count × subprocess_seconds / source_minutes`. This is
the GPU capacity held during that worker window, not GPU utilization or a
separate scheduler bill. Reconcile the full shared allocation once, including
both versions, controller overhead, failures, and idle time. Source preparation
allocation costs are separate and are not hidden in a claimed algorithm gain.

## Denominators, replication, and decision rule

All eleven frozen episodes remain in both denominators. Missing, timed-out,
failed, or unstarted conditions are explicit. Do not compute a successful-only
speedup and do not drop a slow video. Stop the current pair after a failure and
retain its receipt, elapsed work, and partial artifacts. Do not silently retry.

There are two disjoint counterbalanced blocks and one pair per episode. Report
both block ratios and the pooled ratio `sum(v1 wall) / sum(candidate wall)`.
The corpus is small and blocks differ in content. Two blocks do not justify a
repeat-based confidence interval or a population-wide guarantee.

The success criterion is fixed at **1.5× full-workflow throughput versus the
previous optimized v1 path**, before any phase-two GPU profile, pilot, or final
prediction. Roughly 2× remains the ambitious development objective. With the
same three GPUs, reaching 1.5× means at least one-third less GPU capacity-time
per source minute during the matched worker windows. This does not remove
profiling, idle time, or other campaign allocations from the resource bill.

The development rationale is a measured CPU preparation reduction from about
17.69 to 7.53 seconds for the checked pair, with identical processor inputs.
The earlier optimized CLI took about 1,324.7 seconds; its generation and full
cold-model-load diagnostic timers were about 546.6 and 134.9 seconds. These
boundaries are not additive. They support investigating a 1.5–1.8× whole-run
gain as a development hypothesis, not a prediction or result. The GPU pilot
will test the mechanism and determine cost admission. It will not change the
success criterion. The final execution freeze must use `--target-speedup 1.5`.

A passing performance claim requires all eleven paired episodes to complete,
unchanged native inputs and outputs, matching hardware/environment/placement,
and the full workflow metric. Report a missed target plainly. Any changed
output needs investigation against the actual video before a quality statement.
Native equality is not evidence of visual correctness. Do not tune the
criterion on final outputs or final timing.

## Reproducible commands

The following paths are placeholders for verified project storage. Only the
coordinator submits SLURM allocations. These tools never submit a job.

```sh
# Source-only selection. Refuses an existing output.
python scripts/phase2_benchmark.py select --index private-index.json \
  --output selected-sources.json

# Inside the combined allocation, after the separate baseline profile.
# DEV_METADATA is a JSON list of episode_id/source_sha256/duration_s/
# instruction_sha256 for the original development episodes, in manifest order.
python scripts/phase2_benchmark.py pilot --candidate-only \
  --episodes DEV_METADATA.json --manifest DEV_JOBS.jsonl \
  --baseline-root V1_ROOT --candidate-root NEW_ROOT --source FLORA_ROOT \
  --model-lock native-models.json --output FRESH_PILOT --minutes REMAINING_MINUTES

# After the pilot, locally freeze exact code and measured cost admission.
python scripts/phase2_benchmark.py freeze --selection selected-sources.json \
  --pilot pilot-cost-evidence.json --baseline-root V1_ROOT --candidate-root NEW_ROOT \
  --model-lock native-models.json --available-gpu-hours REMAINING_HOURS \
  --available-slots 3 --release-gpu-hours 1.5 --target-speedup 1.5 \
  --output frozen-execution.json

# Run once per allocation, using its exact frozen block manifest and time cap.
python scripts/phase2_benchmark.py run --plan frozen-execution.json --block 0 \
  --baseline-root V1_ROOT --candidate-root NEW_ROOT --source FLORA_ROOT \
  --manifest block-0.jsonl --model-lock native-models.json --output FRESH_BLOCK_0
# Repeat for --block 1 with block-1.jsonl and FRESH_BLOCK_1.

python scripts/phase2_benchmark.py summarize --plan frozen-execution.json \
  --pairs FRESH_BLOCK_0/pair-summary.json FRESH_BLOCK_1/pair-summary.json \
  --accounting final-parent-attempts.json --output comparison.json
```

The coordinator's `bir_phase2_cost_evidence_v1` receipt uses `phase: pilot`,
`status: complete`, the source duration, exact `code_sha256`,
`model_lock_sha256`, and `engine_conditions` bindings. It names the baseline
measurement `historical_unprofiled_v1_optimized` and candidate measurement
`fresh_unprofiled_accelerated_pilot`, with immutable receipt hashes. Each
`conditions.baseline` / `conditions.candidate` row has `status: fresh_complete`,
`gpu_count: 3`, `profile_enabled: false`, measured `subprocess_wall_seconds`,
and matching hardware/environment/placement `parity`. The `comparison` must
record native compatibility and tensor agreement from the actual pilot check.
These measurements size the final run; they are not themselves a matched
performance result. The final freeze validates baseline files against the
immutable Git objects. Remote workers verify the frozen file hashes.

Raw logs, paths, outputs, reference text, and correction records stay private.
Publish only sanitized measured results, code/configuration identities, and
honest scope limits after the final allocation and artifact checks complete.
