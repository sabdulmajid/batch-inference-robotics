# Annotate, review, and export video labels

Use this tool to turn short manipulation videos into timestamped FLORA captions.
Check the video and correct labels before using them in research. A completed
model response is not proof that its description is correct.

The researcher commands are `bir annotate`, `bir status`, `bir review`, and
`bir export`. Review and export need no GPU or model weights. The review view
currently supports human single-view videos.

## Try real precomputed output without downloading models

Run these commands from an editable checkout. Python 3.11 or later is enough
for this demonstration. Examples and model-lock files live in the checkout;
they are not bundled into a standalone wheel.

```bash
git clone https://github.com/sabdulmajid/batch-inference-robotics.git
cd batch-inference-robotics
python3 -m venv .venv
. .venv/bin/activate
pip install -e . 'pyarrow==21.0.0' 'huggingface-hub==1.33.0'

hf download macrodata/WGO-Bench data/annotations.parquet \
  --repo-type dataset --revision 9cfb7bb98d60a58319f4b3d7b41374eedfb7980c \
  --local-dir .local/demo-data
bir demo --parquet .local/demo-data/data/annotations.parquet \
  --output-dir .local/demo
bir review --run-dir .local/demo --open
```

The public dataset download is about 1.4 GB. `bir demo` checks the pinned dataset
checksum, copies only the original `homer_41` video bytes, and installs 15 real
precomputed captions. It performs **zero model calls**. No reference labels are
presented as predictions. If you already have the exact original video, replace
`--parquet ...` with `--video /path/to/homer_41.mp4`; its checksum must match.
The output directory must be new.

The example comes from [MacroData's pinned WGO-Bench release](https://huggingface.co/datasets/macrodata/WGO-Bench/blob/9cfb7bb98d60a58319f4b3d7b41374eedfb7980c/README.md).
Its dataset card lists [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The published example contains generated text, source hashes, and attribution.
This repository does not distribute the video. FLORA retains its
[NVIDIA research and evaluation restriction](../LICENSE-NVIDIA).

In the review page, choose an interval and press **Play interval**. The player
seeks to the saved source time and pauses near the interval end. Browser playback
is for inspection; the exported timestamps retain the exact recorded values.
Edit **Reviewed caption**, enter a reviewer identifier, and press **Save correction**.
The original caption stays visible. **Restore generated label** records a reversal
without removing the edit history. The demo banner remains explicit after edits.

Use the page's **Download JSONL** link, or export from another terminal:

```bash
bir export --run-dir .local/demo --output .local/demo-labels.jsonl
```

Choose a new export filename each time. Existing files, source videos, database
sidecars, generated artifacts, and review journals are never overwritten.

## Prepare the native environment and hardware

Native inference needs the pinned Linux environment and downloaded models.
Follow [the environment and asset setup](REPRODUCTION.md#1-source-and-environment).
The verified configuration uses Python 3.12.12, PyTorch 2.11.0+cu128,
Transformers 5.8.1, three same-node RTX 3090 GPUs, four CPU cores, and 128 GiB
requested host RAM. The Captioner does not fit one or two 24 GiB cards under
the native placement contract. Different hardware or larger inputs need their
own fit check. The normal inference path remains batch one.

Inside an authorized GPU allocation, from the project checkout with its prepared
environment activated:

```bash
export FLORA_SOURCE="$BIR_PROJECT_ROOT/Flora"
export HF_HOME="$BIR_PROJECT_ROOT/cache/huggingface"
export FLORA_MODEL_CACHE="$BIR_PROJECT_ROOT/models"
export HF_HUB_OFFLINE=1
python -c 'import sys, torch, transformers; print(sys.version); print(torch.__version__, torch.version.cuda, transformers.__version__)'
nvidia-smi
bir source-check --source "$FLORA_SOURCE"
python scripts/framework_smoke.py --output "$BIR_PROJECT_ROOT/ops/framework-check.json"
```

The framework check performs a real CUDA operation. GPU visibility alone is
not a successful inference test. Preserve the scheduler's GPU visibility and
resource limits. These commands do not reserve or submit compute.

## Annotate your videos and resume

For a folder of single-view human videos:

```bash
bir annotate --videos /path/to/videos --source "$FLORA_SOURCE" \
  --run-dir /path/to/private/run
```

The command includes subfolders, sorts their relative paths, and uses those
paths as episode IDs. It freezes the selected files and their hashes. Use at
most 100 files per run and at most 256 MiB per camera file. Native validation
applies further source-duration, frame, and profile constraints. Use a new run
directory when the input list, source content, or native settings change.

A JSONL manifest gives explicit IDs and instructions. Paths are relative to
the manifest:

```json
{"id":"shelf-001","video":"videos/shelf-001.mp4","profile":"human","instruction":""}
```

```bash
bir inspect --manifest jobs.jsonl --source "$FLORA_SOURCE"
bir annotate --manifest jobs.jsonl --source "$FLORA_SOURCE" --run-dir runs/research
```

`annotate` selects `--condition accelerated`. This is the phase-two execution
path; its GPU compatibility and throughput evidence are recorded separately
from the already verified v1 baseline. The baseline remains available with
`--condition optimized`. Use separate run directories for different conditions.
Do not interpret a new command name or CPU test as an acceleration measurement.

Progress shows completed stages, failures, and the current episode/stage.
Detailed native output is appended to `inference.log`. Completed segmentation
and caption stages are committed separately. Repeat the **same annotation
command** to resume. The result reports fresh and reused stages separately.
Resume never needs to repeat a completed model stage.

```bash
bir status --run-dir runs/research
bir review --run-dir runs/research --open
bir export --run-dir runs/research --output labels-v1.jsonl
```

`status` returns per-stage states and actions for failures. Check the allocation
or input named by the error, rather than deleting checkpoints. Transient retries
remain bounded. A deterministic failure needs a fix and, if the execution
contract changes, a new run directory. Preserve the earlier failed run as evidence.

For a low-level `bir run` result created before the researcher workflow, supply
the original manifest when opening review:

```bash
bir review --run-dir runs/older-result --manifest jobs.jsonl
```

The viewer verifies source hashes before serving media. It binds only to
`127.0.0.1` and prints a private local URL. It is not a hosted service. Run it
on the machine with the original videos. A browser may not decode every source
codec; use a compatible local player if needed. Do not change source bytes to
make a completed inference result appear to match another video.

## Export format and correction provenance

Each JSONL row uses `schema_version: bir_label_v1` and contains:

| Field | Meaning |
| --- | --- |
| `episode_id`, `request_id`, `record_id` | Stable source request and generated-record identities |
| `source_sha256`, `profile` | Exact original camera-file hashes and native profile |
| `start_s`, `end_s` | Original interval boundaries in seconds; null when no interval exists |
| `generated_caption` | Unchanged generated text, or null when generation did not complete |
| `reviewed_caption`, `effective_caption` | Separate correction, and the selected corrected/original text |
| `inference_status`, `failure` | Original inference state and actionable failure information |
| `generation` | Execution-contract digest, saved payload identity, and explicit precomputed flag |
| `review` | Reviewer identifier, UTC time, revision, prior journal hash, and generated-record binding |

Failed or pending episodes remain in the export. Completed caption stages with
no intervals produce an explicit `no_intervals` record, not a completed label.
Changing a reviewed caption never turns failed inference into successful
inference. Reviewer identifiers are user-supplied provenance, not authenticated
human ground truth. Review changes are appended to `reviews.jsonl`; the generated
SQLite store and caption exports remain unchanged. Concurrent stale edits are
rejected so one browser view cannot silently replace another view's correction.

The workflow tests cover source and checkpoint identity, preserved timestamps,
failed-item coverage, stale corrections, export file safety, localhost video
range requests, and correction/export round trips. Actual browser verification
also played the public `homer_41` interval and saved a separate test correction;
CLI export preserved all 15 generated labels and the correction. This workflow
verification runs no inference and makes no annotation-quality claim.
