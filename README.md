# Batch inference for robotics

This project runs offline video annotation with NVIDIA FLORA TempSeg and
Captioner. Give it a JSONL manifest of videos. It keeps a model loaded across
work items, saves segmentation before captioning, and resumes completed work.
For example, a researcher can annotate several short manipulation videos and
resume after a worker stops without repeating completed segments.

**Status:** the complete native pipeline passed on three single-view human
videos. TempSeg used one RTX 3090; Captioner used three on one node. It exported
45 captions and resumed all six completed stages without new inference. The
model reload correction also passed on two videos, with 30 captions that exactly
matched the prior native outputs. The full benchmark is queued for shared
capacity. No complete performance comparison or adaptive quality result is
available yet.
See [current status](docs/STATUS.md) and
[native end-to-end evidence](reports/native-end-to-end.json), plus
[reload verification](reports/native-reload-verification.json).

## Local checks

Use Python 3.12 for the native environment. Python 3.11 or later supports the
orchestration package.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
git clone https://github.com/NVIDIA/Flora.git third_party/Flora
git -C third_party/Flora checkout --detach 129a2ada74c526ee4692f3d1b111bea6ca19d5b7
pytest -q
bir source-check --source third_party/Flora
```

Tests include real synthetic-video decoding and native frame planning. They do
not establish model inference or annotation quality.

## Video manifest and native run

Paths are relative to the manifest file. Keep real videos and raw outputs in
private project storage. `human` uses one camera. Other native profile names are
`yam`, `dex`, and `tianji`; multi-camera profiles remain subject to upstream
synchronization checks and separate GPU validation.

```json
{"id":"episode-001","video":"videos/episode-001.mp4","profile":"human"}
```

```bash
bir inspect --manifest jobs.jsonl --source third_party/Flora
bir run --manifest jobs.jsonl --source third_party/Flora \
  --run-dir runs/native --condition optimized
```

Repeat the same run command to resume. Use a different run directory for each
benchmark condition: `reference`, `persistent`, or `optimized`. `reference`
loads a new session per episode. `persistent` retains each model for its stage.
`optimized` also enables bounded decoded-frame reuse and CPU input preparation.
Only batch size one is enabled until real GPU measurements establish a safe
larger batch.

The native environment requires the separate pinned dependencies in
[configs/requirements-native.txt](configs/requirements-native.txt). Set
`HF_HOME` and `FLORA_MODEL_CACHE` to verified persistent project storage before
model work. [Reproduction instructions](docs/REPRODUCTION.md) describe the full
sequence and measurement boundaries.

## Contracts and limits

- [Native fidelity and placement](docs/NATIVE_CONTRACT.md)
- [Evaluation design](docs/EVALUATION.md)
- [Resource policy](configs/resource_policy.yaml)
- [Project plan](docs/PROJECT_PLAN.md)
- [WATcloud project overlay](docs/WATCLOUD_RUNBOOK.md)

FLORA source and weights are for non-commercial research and evaluation.
See [third-party notices](THIRD_PARTY_NOTICES.md) and
[the NVIDIA license](LICENSE-NVIDIA). This project does not control a robot.
