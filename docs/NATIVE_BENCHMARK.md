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

Native runtime events also separate model loading from generation. Load events
include asset and parameter verification. Generation events exclude input
preparation, hashing, and device transfer. These rounded event times explain the
workflow; they are already included in wall time.

The benchmark timer includes all three conditions, resume checks, and comparison
work. Allocated GPU-hours remain unset until the coordinator reconciles the
complete scheduler allocation. Do not charge the whole shared allocation to
each condition. Include failed work, idle time, and any restart attempts once.

The fixed reference/persistent/optimized order can affect filesystem caches and
machine state. Report this limit with any observed speedup. One bounded run does
not establish a general performance guarantee. Native agreement measures
compatibility; it does not establish annotation quality.
