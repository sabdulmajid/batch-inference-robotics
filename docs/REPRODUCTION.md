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

For initial setup, place this repository at `$BIR_PROJECT_ROOT/code`. The setup
script uses that exact layout. It creates the pinned upstream checkout at
`$BIR_PROJECT_ROOT/Flora`. Use a fresh task-owned directory for this setup:

```bash
git clone https://github.com/sabdulmajid/batch-inference-robotics.git "$BIR_PROJECT_ROOT/code"
cd "$BIR_PROJECT_ROOT/code"
# Inside the reserved CPU allocation:
bash scripts/prepare_environment.sh
```

Before setup, the coordinator must reserve CPU time and record storage estimates
with `bir.resource_ledger.ResourceLedger` and `check_storage`. The current setup
plan allows 180 GiB of new storage against the unchanged 250 GiB cap. No script
is permission to exceed the resource policy. Preserve the runtime environment
freeze file and private setup logs.

In the prepared CPU environment, extract the initial three-video fixture:

```bash
export PATH="$BIR_PROJECT_ROOT/venv/bin:$PATH"
export PYTHONPATH="$BIR_PROJECT_ROOT/code/src"
export HF_HOME="$BIR_PROJECT_ROOT/cache/huggingface"
export FLORA_MODEL_CACHE="$BIR_PROJECT_ROOT/models"
python scripts/prepare_wgo.py prepare \
  --parquet "$BIR_PROJECT_ROOT/datasets/wgo/data/annotations.parquet" \
  --source "$BIR_PROJECT_ROOT/Flora" \
  --output-dir "$BIR_PROJECT_ROOT/datasets/initial" \
  --max-episodes 3 --split development
```

The output directory must be new. The expected IDs, in order, are `homer_52`,
`homer_50`, and `homer_41`. Their source hashes and selection evidence are in
[the initial manifest](../manifests/initial-native.json). Preparation copies the
original video bytes. It does not generate model outputs. The automated split
proposal from this command is only for the initial fixture; the later adaptive
experiment uses its separately frozen task groups.

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

## 5. Verified release reproduction

The final reproduction completed with three same-node RTX 3090 GPUs, four CPUs,
128 GiB requested RAM, 20 GiB temporary disk, and a 45-minute limit. Its parent
allocation completed in 1,358 seconds with exit code zero. It produced 45 native
captions and then resumed six stages without new inference. See
[the reproduction receipt](../reports/final-reproduction.json) and
[the complete release report](FINAL_REPORT.md).

The commands below express the executed path with portable project variables.
They require the existing prepared environment, pinned upstream source, models,
and initial manifest from sections 1–2. The coordinator must reserve the full
allocation cost before submission, disable requeue, and preserve scheduler GPU
visibility. These commands do not submit a job.

Create immutable source trees from the repository history. Use new directories;
do not overwrite an existing experiment:

```bash
export BIR_NATIVE_CODE="$BIR_PROJECT_ROOT/native-release"
export BIR_RESUME_CODE="$BIR_PROJECT_ROOT/resume-release"
mkdir "$BIR_NATIVE_CODE" "$BIR_RESUME_CODE"
git -C "$BIR_PROJECT_ROOT/code" archive \
  d2701b08a08b88d18887abed81f71c8373478d7b | tar -x -C "$BIR_NATIVE_CODE"
git -C "$BIR_PROJECT_ROOT/code" archive \
  aad9dcfa7ee0ec398e007ad3766a2441d6337602 | tar -x -C "$BIR_RESUME_CODE"
```

The second tree supplies the corrected resume verification helper. Its native
inference modules are byte-identical to the first tree. Inside the authorized
GPU allocation, run:

```bash
set -euo pipefail
test "${SLURM_RESTART_COUNT:-0}" = 0
export PATH="$BIR_PROJECT_ROOT/venv/bin:$PATH"
export PYTHONPATH="$BIR_NATIVE_CODE/src"
export PYTHONUNBUFFERED=1
export HF_HOME="$BIR_PROJECT_ROOT/cache/huggingface"
export FLORA_MODEL_CACHE="$BIR_PROJECT_ROOT/models"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$BIR_PROJECT_ROOT/cache"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export BIR_REPRO_STATE="$BIR_PROJECT_ROOT/runs/reproduce"
export BIR_REPRO_PROBE="$BIR_PROJECT_ROOT/ops/reproduce"
export BIR_REPRO_RESUME="$BIR_PROJECT_ROOT/ops/reproduce-resume"
test ! -e "$BIR_REPRO_STATE"
test ! -e "$BIR_REPRO_PROBE"
test ! -e "$BIR_REPRO_RESUME"
cd "$BIR_NATIVE_CODE"

python "$BIR_NATIVE_CODE/scripts/native_probe.py" \
  --source-root "$BIR_PROJECT_ROOT/Flora" \
  --manifest "$BIR_PROJECT_ROOT/datasets/initial/jobs.jsonl" \
  --state-dir "$BIR_REPRO_STATE" --output-dir "$BIR_REPRO_PROBE" \
  --stages all --condition optimized

python "$BIR_RESUME_CODE/scripts/verify_native_resume.py" \
  --python "$BIR_PROJECT_ROOT/venv/bin/python" --code-root "$BIR_NATIVE_CODE" \
  --source-root "$BIR_PROJECT_ROOT/Flora" \
  --manifest "$BIR_PROJECT_ROOT/datasets/initial/jobs.jsonl" \
  --state-dir "$BIR_REPRO_STATE" \
  --gate-c-marker "$BIR_REPRO_PROBE/probe-summary.json" \
  --output-dir "$BIR_REPRO_RESUME"
```

Expected evidence is `probe-summary.json` with `native_caption_gate=true`,
`fresh_inference=true`, and `complete_caption_count=45`; six complete first
stage attempts in the engine summary; and `receipt.json` in the resume directory
with `resumed_stages=6`, `fresh_stage_attempts=0`, and
`original_evidence_unchanged=true`. Verify the final scheduler state separately.
The resume subprocess hides GPUs, but its time remains inside the parent GPU
allocation and is counted there once.

The controlled performance command is in [NATIVE_BENCHMARK.md](NATIVE_BENCHMARK.md).
The frozen adaptive result and rejected final projection are in
[ADAPTIVE_RESULTS.md](ADAPTIVE_RESULTS.md). Do not treat a cache-only resume,
an unadmitted batch, or an estimated final runtime as fresh inference evidence.

## 6. Phase-two researcher workflow reproduction

The phase-two candidate is `87fff61d812b2e9cd9954fec5e25d4c953eeb330`.
Its resume verifier is `88f9b85c238400685771f95b08d64b99e238247f`.
The fresh verification run is pending. These are its exact commands, with
portable paths. See [the phase-two results](PHASE2_RESULTS.md) for its eventual
verified outcome. The phase-one reproduction above remains historical evidence.

Reuse the prepared environment, pinned source, models, and original three-video
manifest. Create two new source directories from the repository history:

```bash
export BIR_PHASE2_CODE="$BIR_PROJECT_ROOT/phase2-release"
export BIR_PHASE2_HELPER="$BIR_PROJECT_ROOT/phase2-resume-helper"
mkdir "$BIR_PHASE2_CODE" "$BIR_PHASE2_HELPER"
git -C "$BIR_PROJECT_ROOT/code" archive \
  87fff61d812b2e9cd9954fec5e25d4c953eeb330 | tar -x -C "$BIR_PHASE2_CODE"
git -C "$BIR_PROJECT_ROOT/code" archive \
  88f9b85c238400685771f95b08d64b99e238247f | tar -x -C "$BIR_PHASE2_HELPER"
```

Inside the reserved allocation, use the prepared environment. The recorded
request is thirty minutes on three same-node RTX 3090 GPUs, four CPU cores,
128 GiB host RAM, and 20 GiB temporary disk, without automatic requeue.
Preserve the scheduler's GPU visibility. These commands do not submit a job.

```bash
set -euo pipefail
test "${SLURM_RESTART_COUNT:-0}" = 0
export PATH="$BIR_PROJECT_ROOT/venv/bin:$PATH"
export PYTHONPATH="$BIR_PHASE2_CODE/src"
export HF_HOME="$BIR_PROJECT_ROOT/cache/huggingface"
export FLORA_MODEL_CACHE="$BIR_PROJECT_ROOT/models"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export BIR_PHASE2_STATE="$BIR_PROJECT_ROOT/runs/phase2-reproduce"
export BIR_PHASE2_RESUME="$BIR_PROJECT_ROOT/ops/phase2-reproduce-resume"
export BIR_PHASE2_STATUS="$BIR_PROJECT_ROOT/ops/phase2-reproduce-status.json"
export BIR_PHASE2_LABELS="$BIR_PROJECT_ROOT/ops/phase2-reproduce-labels.jsonl"
test ! -e "$BIR_PHASE2_STATE"
test ! -e "$BIR_PHASE2_RESUME"
test ! -e "$BIR_PHASE2_STATUS"
test ! -L "$BIR_PHASE2_STATUS"
test ! -e "$BIR_PHASE2_LABELS"
test ! -L "$BIR_PHASE2_LABELS"
set -C
cd "$BIR_PHASE2_CODE"
python -m bir.cli annotate \
  --manifest "$BIR_PROJECT_ROOT/datasets/initial/jobs.jsonl" \
  --source "$BIR_PROJECT_ROOT/Flora" --run-dir "$BIR_PHASE2_STATE"
python "$BIR_PHASE2_HELPER/scripts/verify_workflow_resume.py" \
  --python "$BIR_PROJECT_ROOT/venv/bin/python" --code-root "$BIR_PHASE2_CODE" \
  --source-root "$BIR_PROJECT_ROOT/Flora" \
  --manifest "$BIR_PROJECT_ROOT/datasets/initial/jobs.jsonl" \
  --state-dir "$BIR_PHASE2_STATE" --output-dir "$BIR_PHASE2_RESUME"
python -m bir.cli status --run-dir "$BIR_PHASE2_STATE" > "$BIR_PHASE2_STATUS"
python -m bir.cli export --run-dir "$BIR_PHASE2_STATE" --output "$BIR_PHASE2_LABELS"
```

The verifier backs up the completed state, hides GPUs from the child process,
and invokes the same `annotate` command on the copy. It checks six reused stages,
zero new model calls or stage attempts, unchanged request identities and labels,
and unchanged original evidence. This is a completed-stage recovery test; it
does not resume a partially generated caption. Keep the original fresh run,
resume receipt, status output, exported labels, and scheduler outcome.
