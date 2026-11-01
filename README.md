# FLORA video annotation for researchers

Turn a folder of manipulation videos into timestamped action labels. Inspect
each interval beside its video, keep corrections separate from model output,
and export JSONL for analysis or dataset preparation. Interrupted runs retain
completed stages and resume with the same command.

This is an offline research tool around NVIDIA FLORA TempSeg and Captioner.
For example, a researcher can annotate short assembly videos, inspect uncertain
labels, and export generated and reviewed text with source hashes and timestamps.

**Phase two is under GPU validation.** The new direct image preparation path
and researcher workflow are implemented. CPU tests establish input equivalence
on tested requests. They do not establish a complete-pipeline speedup. The
[previous verified release](docs/FINAL_REPORT.md) remains the comparison
baseline. See [current progress](docs/STATUS.md) and the
[frozen comparison protocol](docs/PHASE2_PROTOCOL.md).

## Try the review workflow without model weights

The [quickstart](docs/QUICKSTART.md) includes a pinned public-data download and
a precomputed example with 15 real captions. The example is labeled as
precomputed throughout the interface and export. It does not run inference.

From this checkout, after installing the lightweight package and obtaining the
pinned public dataset:

```sh
pip install -e '.[test]'
bir demo --parquet annotations.parquet --output-dir runs/demo
bir review --run-dir runs/demo --open
bir export --run-dir runs/demo --output labels.jsonl
```

Select an interval, play it, and save a correction with your reviewer name.
Export retains the original caption, correction history, source identity,
timestamps, and failures. Model agreement does not establish visual accuracy;
inspect labels before using them as training data.

## Annotate your videos

Use the [pinned native environment and model setup](docs/QUICKSTART.md).
Run these commands from the checkout with the prepared FLORA source and assets:

```sh
bir source-check --source third_party/Flora
bir annotate --videos /path/to/videos --source third_party/Flora \
  --run-dir runs/my-dataset
bir status --run-dir runs/my-dataset
bir review --run-dir runs/my-dataset --open
bir export --run-dir runs/my-dataset --output labels.jsonl
```

Repeat the same `annotate` command to resume. A manifest can replace `--videos`:

```json
{"id":"episode-001","video":"videos/episode-001.mp4","profile":"human"}
```

Paths are relative to the manifest. Folder mode uses the human single-view
profile and includes subfolders. Source bytes and settings are bound to each
request. Changed runtime dependencies or native settings require a new run
directory. Failures remain visible; they are not exported as successful labels.

The default `accelerated` condition removes lossless PNG serialization and
readback from Captioner preparation. It passes the same native RGB pixels,
frame clocks, prompts, and processor tensors directly, and prepares one next
request while generation runs. TempSeg's native transport stays unchanged.
Metadata can spill to bounded temporary storage, so the second stage can reuse
validated metadata beyond eight episodes. Models remain loaded across episodes.

The lower-level `bir run` interface retains `reference`, `persistent`, and
`optimized` conditions. Use `--condition optimized` to select the previous
preparation path. The immutable v1 comparison uses the exact prior code release,
not a relabeled new implementation.

## Hardware and scope

- Verified v1 inference used three same-node RTX 3090 GPUs, four CPU cores,
  and 128 GiB host RAM. Full Captioner parameter storage is about 50.96 GiB.
  One or two 24 GiB GPUs cannot hold this native configuration.
- The new path targets short human single-view videos. Its retained source RGB
  and final CPU tensor payload is bounded at 1 GiB per request, with at most one
  preparing/ready request beside the active request and a separate 256 MiB
  decoded-frame cache. This is not a whole-process RAM bound.
- Native values, precision, temporal evidence, generation limits, parsing, and
  stitching remain unchanged. Production inference stays batch one. No new
  batching or approximate-model claim is made.
- The broader comparison contains eleven complete unseen episodes across eight
  task groups. Its GPU results are pending. Prior native agreement and adaptive
  results are preserved in the [baseline report](docs/FINAL_REPORT.md).

## Development and provenance

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

See the [native contract](docs/NATIVE_CONTRACT.md),
[resource policy](configs/resource_policy.yaml),
[WATcloud overlay](docs/WATCLOUD_RUNBOOK.md), and
[prior reproduction commands](docs/REPRODUCTION.md).

FLORA source and weights retain NVIDIA's non-commercial research and evaluation
license. The example includes public-dataset attribution. See
[third-party notices](THIRD_PARTY_NOTICES.md) and [the NVIDIA license](LICENSE-NVIDIA).
This tool does not train or control a robot.
