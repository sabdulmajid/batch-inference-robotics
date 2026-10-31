# Bounded native benchmark

`scripts/native_benchmark.py` compares reference, persistent, and optimized
native execution on the same manifest. It runs each condition in a fresh process
inside one GPU allocation. It also checks cache-only resume for each condition.
For example, three supported short episodes can show whether repeated model
loading or CPU preparation dominates this workload.

Run this only after a complete native TempSeg-plus-Captioner probe has passed.
The coordinator must confirm its final SLURM state and reserve the benchmark's
resources first. The script submits no jobs. It rejects restarted allocations,
missing Gate C evidence, and manifests with more than 12 episodes.

```bash
python scripts/native_benchmark.py \
  --source-root third_party/Flora \
  --manifest runs/prepared/jobs.jsonl \
  --gate-c-marker runs/native-caption-probe/probe-summary.json \
  --output-dir runs/native-benchmark-01
```

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
  and engine cleanup. Worker framework checks occur before this timer.
- Stage attempt time excludes input preparation.

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
