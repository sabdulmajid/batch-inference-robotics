# Bounded native benchmark

`scripts/native_benchmark.py` compares reference, persistent, and optimized
native execution on the same manifest. It runs each condition in a fresh process
inside one GPU allocation. It also checks cache-only resume for each condition.
For example, three supported short episodes can show whether repeated model
loading or CPU preparation dominates this workload.

## Measured result

The completed 2026-10-31 trial used release
`d2701b08a08b88d18887abed81f71c8373478d7b`, three same-node RTX 3090 GPUs,
four CPU cores, and the pinned Python 3.12 / PyTorch 2.11.0+cu128 environment.
The inputs were `homer_52`, `homer_50`, and `homer_41`: 138.767089 seconds of
single-view human video. All conditions used request batch size one and the same
native model placement, inputs, and parameter checks.

The optimized condition took 1,328.397 seconds of subprocess wall time, versus
1,487.463 seconds for reference. This is an observed **1.120x speedup**, or
10.694% less wall time, in one fixed-order trial. All 45 captions and native
input/output projections matched. Each comparison also matched all 65 generation
input tensor digests as a corpus multiset: 20 TempSeg and 45 Captioner calls.
This establishes native compatibility, not annotation quality.

| Condition | Subprocess seconds | CLI seconds | Source hours per subprocess hour | TempSeg / Captioner loads |
| --- | ---: | ---: | ---: | ---: |
| Reference | 1,487.463 | 1,482.223 | 0.093291 | 3 / 3 |
| Persistent | 1,371.814 | 1,368.256 | 0.101156 | 1 / 1 |
| Optimized | 1,328.397 | 1,324.739 | 0.104462 | 1 / 1 |

These are whole-workflow comparisons. They do not isolate one optimization.
Reference prepared native episode metadata three times; persistent prepared it
six times; optimized prepared it three times. Measured preparation work was
19.455, 38.561, and 20.092 seconds, respectively. Metadata reuse, traversal order,
and preparation overlap differ between conditions. Fixed trial order can also
affect filesystem caches and machine state. No repeated-trial uncertainty
estimate is available.

The optimized frame cache recorded 241 hits, 2,591 misses, and 2,548 evictions.
Its peak retained RGB data was 268,429,152 bytes, within the 268,435,456-byte
(256 MiB) limit. It retained 43 entries at the final snapshot. The measured CPU
worker and stage intervals imply at least 8.959 seconds of overlap. This is not
a GPU utilization measurement or an isolated prefetch speedup.

| Condition | GPU 0 peak allocated GiB | GPU 1 peak allocated GiB | GPU 2 peak allocated GiB | Result latency median / p95 seconds |
| --- | ---: | ---: | ---: | ---: |
| Reference | 20.927 | 22.314 | 22.695 | 1,482.013 / 1,482.039 |
| Persistent | 20.908 | 22.314 | 22.687 | 1,368.046 / 1,368.070 |
| Optimized | 20.907 | 22.314 | 22.687 | 1,324.525 / 1,324.552 |

Result latency starts when all jobs enter the engine and ends when each export
completes. All exports follow inference and model cleanup, so these values are
close to total engine time. They exclude scheduler wait and CLI setup.

All three separate resume checks passed with six resumed stages, zero new
attempts, unchanged annotations, and no model loading or generation. Their
subprocess times were 3.959, 3.698, and 3.781 seconds. These times are separate
from fresh-inference throughput.

The full allocation completed with exit code zero in 4,200 seconds. It consumed
**3.5 allocated GPU-hours**, charged once for all conditions, resume checks,
comparison work, cleanup, and idle time. Per-condition allocated GPU-hours were
not measured separately. The scheduler batch-step host-memory sample was
11.205 GiB; it does not establish complete per-worker memory peaks. Project GPU
use after this benchmark was 6.787222 hours, including earlier failed work.
The final reproduction allowance remained 2.25 GPU-hours and one submission slot.

The [sanitized measured report](../reports/native-benchmark.json) contains exact
times, per-episode latency, allocated and reserved GPU-memory peaks, preparation
and cache counters, model-load and generation timings, source hashes, and evidence
checksums. GPU microbatching and adaptive quality remain unvalidated by this
benchmark. The adaptive experiment is pending.

## Reproduction and measurement boundaries

Run this only after a complete native TempSeg-plus-Captioner probe has passed.
The coordinator must confirm its final SLURM state and reserve the benchmark's
resources first. The script submits no jobs. It rejects restarted allocations,
missing Gate C evidence, and manifests with more than 12 episodes.

```bash
python scripts/native_benchmark.py \
  --source-root third_party/Flora \
  --manifest runs/prepared/jobs.jsonl \
  --gate-c-marker runs/native-caption-probe/probe-summary.json \
  --model-lock configs/native-models.json \
  --captioner-headroom-gib 4 \
  --output-dir runs/native-benchmark-reproduction
```

Run the measured release in an isolated checkout. The prepared manifest must
contain the same three episode IDs and source hashes listed in the report.
Use the pinned Linux Python 3.12 environment. Set `HF_HOME` and
`FLORA_MODEL_CACHE` to the already prepared project storage. The benchmark forces
offline model access. It preserves the scheduler's device visibility. The output
directory must be fresh. Keep all generated files private until review.

The Gate C marker must report a complete native caption probe. Its exported
annotations must include both models' verification and GPU residency receipts,
temporal and caption input audits, and complete nonempty captions. The script
checks these artifacts. Scheduler completion remains the coordinator's check.

Each condition has a separate state directory, runtime log, probe summary, fresh
summary, and preserved `fresh/` annotation snapshot. A second process checks
resume against the original state. The original fresh summary is retained even
though the run store's current summary becomes a resume summary. Resume must
show zero new attempts and no model-load or generation events.

A deterministic native model-load failure stops the current engine invocation.
Completed checkpoints remain available. Later fresh stages are marked blocked
without another model load. Input-specific generation failures and bounded
transient I/O retries retain their separate behavior. Lifecycle events measure
all-device free, allocated, and reserved memory before and after synchronized
upstream cleanup. Failed placement checks retain their measured memory limits.
These diagnostics help distinguish a capacity failure from a generated-input
out-of-memory error; they do not relax the placement rule.

The script compares actual device placement, parameter dtype/byte distribution,
hardware, package versions, source files, and input hashes across conditions.
The plan pins the benchmark/comparison scripts and Gate C artifact hashes before
the first worker. Each later worker checks the same plan before inference.
It uses `compare_native.py` to compare native outputs and processor tensor
digests. It stops if a condition fails, a retry occurs, inputs or outputs differ,
or hardware/placement changes. Earlier evidence remains available for diagnosis.

`benchmark-summary.json` reports source hours once per episode, fresh condition
work, per-stage attempt times, all-device peak memory, and resume work. It keeps
three timing boundaries separate:

- Subprocess wall time includes process startup and worker checks.
- CLI workflow time includes manifest hashing, preparation, inference, export,
  and model cleanup. Worker framework checks occur before this timer. Final
  summary writing and SQLite close occur after it, inside the subprocess timer.
- Stage attempt time excludes input preparation, attempt creation, and final
  checkpoint commit. It includes model loading when that stage triggers it.

Native runtime events also separate model loading from generation. The upstream
parameter-verification event measures time through parameter checks. It excludes
processor finalization. The upstream model-ready event has no duration field.
The adapter records a separate full load time, including processor construction,
audit and residency checks, and synchronization of every GPU. Failed load work
is included. Older traces without this event retain an unknown full load time.
Generation events exclude input preparation, hashing, and device transfer.
These event intervals overlap. They are already included in wall time.

Each fresh run also retains engine telemetry. CPU counters separate source
verification, native input preparation, and prepared-metadata reuse. Their times
sum worker work, which can overlap inference. They are not elapsed wall time.
Frame-cache records contain hits, misses, evictions, current and peak retained
RGB bytes, and configured limits. The engine saves them before cache cleanup.
Cache byte counts exclude Python objects and other process memory.

Episode latency assumes that all manifest jobs arrive at the start of the engine
run. The result becomes available when its serialized export completes. Exports
occur after all inference and model cleanup, in manifest order. Their latencies
can therefore be close. Reports give the median and p95 only for a wholly fresh,
complete condition. They retain failed, partial, cached, and mixed results with
explicit labels. Percentiles use linear interpolation at index `(n - 1) * p`.
This boundary excludes CLI setup and the scheduler queue.

The benchmark timer includes all three conditions, resume checks, and comparison
work. Allocated GPU-hours remain unset until the coordinator reconciles the
complete scheduler allocation. Do not charge the whole shared allocation to
each condition. Include failed work, idle time, and any restart attempts once.

The fixed reference/persistent/optimized order can affect filesystem caches and
machine state. Report this limit with any observed speedup. One bounded run does
not establish a general performance guarantee. Native agreement measures
compatibility; it does not establish annotation quality.
