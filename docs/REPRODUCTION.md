# Reproduce and inspect the engine

These commands separate CPU verification from GPU evidence. A successful CPU
test does not prove that either model fits the chosen GPUs.

## 1. Source and environment

Clone this project. Create a Python environment and install `.[test]` as shown
in the README. Clone the pinned FLORA source into `third_party/Flora`; keep its
tracked files unchanged. Source and model pins are in `configs/native-models.json`.

For native execution, use Linux x86_64 and Python 3.12.12. In a scheduled CPU
allocation, `scripts/prepare_environment.sh` creates a task-owned environment,
installs the CUDA 12.8 PyTorch build and native dependencies, downloads the exact
model and dataset revisions, and checks the released model manifests. Set
`BIR_PROJECT_ROOT` to a verified private project directory first. The script
requires a SLURM allocation and rejects a restarted job. It never submits jobs.

Before setup, the coordinator must reserve CPU time and record storage estimates
with `bir.resource_ledger.ResourceLedger` and `check_storage`. The current setup
plan allows 180 GiB of new storage against the unchanged 250 GiB cap. No script
is permission to exceed the resource policy. Preserve the runtime environment
freeze file and private setup logs.

## 2. Native gates

Set `HF_HOME` and `FLORA_MODEL_CACHE` to the prepared paths. Do not change
`CUDA_VISIBLE_DEVICES` to expose devices outside the scheduler grant.

Within a budget-reserved GPU allocation:

```bash
python scripts/framework_smoke.py --output runs/framework.json
```

Then use `scripts/native_probe.py --help` for the segmentation and caption
probes. Segmentation and captioning can share the same state directory across
separate allocations. Keep the manifest, code, native settings, and request
identities unchanged. Start with at most three supported episodes. Use a native
CPU `bir inspect` check before model inference.

The initial TempSeg candidate is one 24 GiB GPU. Captioner needs a justified
same-node placement across up to four GPUs. These are candidates until an actual
probe establishes fit, device residency, output completeness, and memory use.
Never substitute quantization or CPU/disk inference offload for this gate.

## 3. Resume and benchmark conditions

```bash
bir run --manifest jobs.jsonl --source third_party/Flora --run-dir runs/reference --condition reference
bir run --manifest jobs.jsonl --source third_party/Flora --run-dir runs/persistent --condition persistent
bir run --manifest jobs.jsonl --source third_party/Flora --run-dir runs/optimized --condition optimized
```

Use identical inputs, hardware allocation, placement settings, revisions, and
requested outputs. Use fresh directories for comparisons. A second invocation
of a completed directory measures result retrieval, not inference. Inspect
`fresh_stage_attempts`, resumed stages, partial statuses, and failures.

The engine run timer includes preparation, inference, exports, and session
cleanup. The CLI workflow timer also covers its setup. Stage attempt timers
exclude preparation. An annotation's summed stage time can include segmentation
from a prior invocation. Do not use that field as current end-to-end wall time.
Allocated GPU-hours come from scheduler allocation records, once per attempt;
include failures, idle allocation time, and restarts.

After a successful native probe, verify its resume path in a small reserved CPU
allocation. This command makes an isolated database backup and copies the saved
stage artifacts. It runs the exact engine code that produced the original state.
It checks every cached stage, zero new attempts, unchanged output hashes, and
absence of model loading or generation. It preserves the original evidence.

```bash
python scripts/verify_native_resume.py \
  --python "$NATIVE_PYTHON" --code-root "$PROBE_CODE" \
  --source-root "$FLORA_SOURCE" --manifest "$MANIFEST" \
  --state-dir "$PROBE_STATE" --gate-c-marker "$GATE_C_RECEIPT" \
  --output-dir "$RESUME_EVIDENCE"
```

The output directory must be new. The tool hides GPUs from its subprocess and
never submits jobs. Keep its raw receipts and logs private. A local test of the
tool uses synthetic stored results; the real probe state still needs this check.

## 4. What to retain

Keep input hashes, request identities, model verification receipts, processor
tensor digests, prompt hashes, stage results, environment lock, source revision,
all-device GPU peaks, scheduler final state, and exact commands. Store raw
operational records privately. Publish only reviewed, sanitized measurements.

A final adaptive experiment requires frozen splits, evaluator, spatial presets,
routing rule, and quality margin. The planned protocol is not a measured result.
