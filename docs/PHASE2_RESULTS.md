# Phase-two results: faster native annotation and a review workflow

The tool can annotate a folder or JSONL manifest, show stage progress and
failures, resume completed work, play caption intervals in a local browser,
record separate corrections, and export timestamped JSONL labels. Start with
the [quickstart](QUICKSTART.md). Its precomputed example contains 15 actual
captions and needs no model weights or GPU.

The new execution path achieved **1.57× throughput** on one complete,
preselected six-episode comparison: **37.4 minutes instead of 58.9 minutes** on
the same three GPUs. All 121 captions and 170 native model inputs matched.
The full eleven-episode success criterion was **not met**. The previous
optimized baseline failed on two episodes in the other block, so a complete
eleven-episode speedup cannot be reported.

## What changed

The previous Captioner path repeatedly encoded frames as lossless PNG images,
then read and decoded them before processing. A real profile of one development
episode recorded 613 PNG-save calls with 187.955 cumulative seconds. This
identified substantial repeated CPU work. Profile timers can contain nested
work and waits; the profiled run is not a speedup measurement.

The `accelerated` path passes the same RGB frames directly to the native
processor and prepares one next request while GPU generation runs. It preserves
frame selection and timestamps, resize behavior, camera geometry, prompts,
processor tensors, model values and dtypes, generation limits, parsing, and
stitching. TempSeg transport is unchanged. Both models persist across work
items, with one model stage resident at a time. Production inference remains
batch one, with no quantization or inference-weight offload.

Two supporting changes improve larger runs and recovery. A bounded metadata
spill cache retains reusable metadata beyond the previous eight-entry memory
cache. Runtime dependency versions now enter the execution identity, so changed
dependencies cannot silently resume an incompatible run. The twelve-item CPU
regression covers the cache limit; the five- and six-item GPU blocks do not.

The direct path bounds retained RGB and final CPU tensor payloads at 1 GiB per
request. It keeps at most one preparing/ready request beside the active request,
plus a separate 256 MiB decoded-frame cache. These bounds exclude temporary
processor allocations and are not a whole-process RAM limit.

See the [native pilot](../reports/phase2-native-pilot.json) and
[45-request CPU input check](../reports/phase2-preparation-check.json).

## Complete six-episode comparison

The frozen block contains `homer_39`, `homer_56`, `homer_33`, `homer_53`,
`homer_37`, and `homer_40`: **363.280933 source seconds**. Both conditions used
all six complete original videos, twelve fresh stage attempts, and no resumed
stages. The accelerated condition ran first, then the previous optimized
release, in separate fresh processes within one allocation.

| Measure | Previous optimized | Accelerated |
| --- | ---: | ---: |
| Full subprocess wall time | 3,534.320 s | 2,246.460 s |
| Full subprocess wall time | 58.905 min | 37.441 min |
| Source minutes per wall minute | 0.102787 | 0.161713 |
| GPU capacity-seconds per source minute | 1,751.200 | 1,113.086 |
| Full cold-model-load time | 135.587 s | 135.678 s |
| Derived workflow time after subtracting model loads | 3,398.733 s | 2,110.782 s |
| Complete captions | 121 | 121 |
| TempSeg / Captioner generation calls | 49 / 121 | 49 / 121 |
| Fresh stage attempts / failed attempts | 12 / 0 | 12 / 0 |

Throughput is **1.573284×** and full wall time is **36.438699% lower** in this
block. For these 6.05 minutes of video, the saved elapsed time was 21.46 minutes.
The unchanged three-GPU count gives the same percentage reduction in held GPU
capacity-time during each condition's measured process window. This does not
measure GPU utilization or a separate scheduler bill for each condition.

The parent allocation completed successfully in **5,782 seconds**, using
**4.818333 GPU-hours**. This charges both conditions, controller work, and idle
time once. Source preparation is separate: eleven videos required 132 seconds
on four CPU cores. The earlier 45-request CPU compatibility check used another
1,008 seconds on four cores. These costs remain in the project ledger.
The [machine-readable comparison](../reports/phase2-comparison.json) preserves
both blocks and their full-denominator outcome.

Primary timing is parent-observed subprocess time. It includes imports, input
and environment checks, model loading, preparation, execution, serialization,
and worker cleanup. Both conditions use the same audit level and common worker;
detailed profiling is disabled. CPU preparation and GPU work can overlap, so
their timers must not be added to elapsed time. The load-excluded time is a
derived diagnostic, not a separately measured steady-state trial.

The exact code versions are baseline
`d2701b08a08b88d18887abed81f71c8373478d7b` and candidate
`87fff61d812b2e9cd9954fec5e25d4c953eeb330`. Native parameter checks, source
hashes, actual device placement, dependency versions, and input tensors matched.
Request comparisons bind each tensor record to its episode, stage, and ordinal.
All caption text, interval boundaries, and stitching agree. This establishes
native compatibility, not correct visual descriptions.

## Failed block and full-workload decision

The source-only plan froze eleven unseen episodes, **726.897933 seconds**,
across eight task groups. The pinned supported pool had only eleven unseen
episodes after the three development clips were excluded. This is below the
requested 12–20 range. The workload was divided into two balanced blocks with
opposite execution orders before final predictions.

| Block | Planned episodes | Outcome | Allocated GPU-hours |
| --- | ---: | --- | ---: |
| 0: previous optimized, then accelerated | 5 | Baseline failed on two episodes; accelerated did not start | 2.794167 |
| 1: accelerated, then previous optimized | 6 | Both conditions completed all six episodes | 4.818333 |

In block 0, Captioner exhausted GPU memory on `homer_59` and `homer_60`.
All five TempSeg stages passed. The remaining three episodes produced 65
serialized captions; partial work from failed episodes is not counted as
complete output. The failed allocation used 3,353 seconds on three GPUs, all
charged. Identical input shapes succeeded elsewhere. That observation does
not prove allocator fragmentation or establish a universal input-size limit.
See [the preserved failure evidence](../reports/phase2-block0-failure.json).

All eleven episodes remain in the denominator. Only six have a complete paired
measurement. The experiment's fixed success rule required complete paired
coverage and at least **1.5×** throughput across all eleven episodes. It did not
pass. There is no pooled full-workload ratio and no successful-only substitute.
The six-episode number above describes its predeclared block only. No failed
episode was dropped, retried, or used to tune the final settings.

There was one paired run per episode. The failed block removes the planned
opposite-order comparison, so order effects remain unresolved. No repeat-based
confidence interval or population-wide speedup is supported. Roughly 2× was
the ambitious objective; it was not demonstrated.

## Hardware and memory limits

Both conditions used three same-node RTX 3090 GPUs, four CPU cores, and 128 GiB
requested host RAM. The environment included Python 3.12.12, PyTorch
2.11.0+cu128, CUDA 12.8, and Transformers 5.8.1. The native Captioner parameter
payload is about 50.96 GiB. One or two 24 GiB GPUs cannot hold this configuration.

Both complete conditions recorded the same per-device peaks:

| Logical GPU | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: |
| 0 | 20.921 | 23.207 |
| 1 | 22.332 | 23.176 |
| 2 | 22.687 | 23.180 |

Framework reserve approached device capacity, leaving about 0.35–0.38 GiB
between peak reserve and reported capacity. This is not a guaranteed free-memory
margin throughout inference. The earlier failures show that three 24 GiB cards
are not a reliable fit for every tested episode. Different hardware, longer
videos, larger contexts, and other camera profiles need separate validation.

## Researcher workflow and reproduction

Use `bir annotate --videos VIDEO_FOLDER --source FLORA_SOURCE --run-dir RUN`
or replace `--videos` with `--manifest JOBS.jsonl`. Then use `bir status`,
`bir review`, and `bir export`. Repeat the same annotation command to resume.
The [quickstart](QUICKSTART.md) gives complete setup, example, and export commands.

Review stores corrections separately from generated labels. JSONL export keeps
source hashes, timestamps, request identities, generated and reviewed text,
review provenance, and explicit failed or pending records. Actual browser
verification played an interval from the real precomputed example and saved a
correction. CLI export preserved all 15 generated labels and that separate edit.
The viewer runs on localhost; it is not a hosted service.

**Final fresh annotation, isolated CPU resume, status, and export reproduction:
pending.** The protected final run uses the original three development episodes
and the unchanged candidate. Its result will be added after final scheduler and
artifact verification.

The [frozen protocol](PHASE2_PROTOCOL.md#reproducible-commands) contains the
comparison commands, timing boundaries, and decision rule. The
[execution plan](../manifests/phase2-execution.json) binds the exact code,
model configuration, sources, and budget admission. Prior
[phase-one results](FINAL_REPORT.md), including the inconclusive adaptive
experiment and failed batching admission, remain unchanged.
