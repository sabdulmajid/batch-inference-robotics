# Review recorded rollouts and export to IQL

This guide connects a real simulator trajectory to browser review and an existing
offline-RL trainer. Video annotations, recorded actions, environment rewards,
FLORA outcomes, and review corrections remain separate. Native Rewarder execution
is pending; no GPU work is authorized in this phase.

## 1. Open the public examples

Open [the static demo](https://flora-rollout-review.exact-leaf-7339.chatgpt.site).
Verified in a clean Incognito window without a ChatGPT login.
No account, model download, or inference endpoint is used.

The human-video example contains 15 genuine precomputed FLORA captions and no
robot actions. The robot example contains 59 recorded Panda Lift transitions and
a replay rendered from saved simulator states. It has no native FLORA captions
or outcomes. Source task text and environment reward are not FLORA predictions.

In the human example, select an interval, play it, and inspect the generated text. Enter a reviewer
identifier and save a correction. **Download bundle** retains the original
prediction payload and all correction events. Import that file again to verify
the history. Keep it with its matching source media; the JSON bundle does not
embed video bytes. Each example retains its dataset attribution and restrictions.

## 2. Install and inspect an existing run

Use Python 3.11 or 3.12 for the lightweight package. The `data` extra adds the
numeric and HDF5 tools needed for trajectory import/export; it does not install
FLORA, PyTorch, or a simulator.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install 'batch-inference-robotics[data] @ git+https://github.com/sabdulmajid/batch-inference-robotics.git@v0.3.0'
bir resources --output-dir bir-resources
bir bundle-check --bundle bir-resources/web/examples/robot.bir.json
```

The resource directory must be new. It contains packaged examples, configuration,
helper scripts, and the static review app. A wheel installation works outside an
editable source checkout. To run the same static app locally:

```sh
python -m http.server 8080 --bind 127.0.0.1 --directory bir-resources/web
```

Open `http://127.0.0.1:8080`. Use **Import bundle** and explicitly select matching
local media. Bundle and source checksums must pass. Selected media, filenames,
labels, and corrections stay in your browser. There is no upload or analytics
request. Public example assets are the only fetched data. Local draft recovery
uses browser storage and reports storage failures; download a bundle to keep a
portable copy. Clearing browser storage does not change a downloaded bundle.

For the existing local annotation-run viewer, keep using:

```sh
bir review --run-dir runs/my-data --open
bir export --run-dir runs/my-data --output labels.jsonl
```

This local server binds to loopback. It is separate from the public static app.
Annotation JSONL contains the latest applicable correction, not the full journal.
Full history remains in `reviews.jsonl` and portable run bundles. New corrections
require a completed generated caption. Historical edits to changed or incomplete
records remain visible and do not block unrelated records or move to new captions.

To package an existing annotation run, supply its measured media durations:

```json
{"episode-001":{"task":"Lift the object.","media":[{"camera":"main","filename":"episode-001.mp4","source_sha256":"REPLACE_WITH_ORIGINAL_64_HEX_SHA256","duration_s":12.0}]}}
```

```sh
bir bundle-pack --run-dir runs/my-data --media-info media-info.json \
  --output annotated.bir.json
bir bundle-check --bundle annotated.bir.json
```

Use the exact media-info structure accepted by `bir bundle-pack --help`; source
hashes must match the recorded run; do not substitute a transcoded file. A precomputed example
is not fresh inference. Completed run data must not be relabeled as a different
source to make an attachment pass.

## 3. Import real trajectories and preserve environment rewards

The first adapter accepts the pinned official robomimic Lift / proficient-human /
low-dimensional dataset for robosuite 1.5. It has real recorded simulator control
commands. It is not a converter for arbitrary CSV, robot logs, or video-only data.

```sh
python bir-resources/scripts/prepare_rollout_example.py \
  --dataset data/lift.hdf5 --download-only
bir trajectory-import --dataset data/lift.hdf5 \
  --demos demo_0 demo_1 --output lift.bir.json
bir bundle-check --bundle lift.bir.json
bir training-export --bundle lift.bir.json --output lift-iql.hdf5 \
  --reward-source environment --boundary-mode source-mask
```

The pinned download is 21,084,088 bytes. The importer verifies its SHA-256 before
selecting two demonstrations: 59 and 58 transitions. It preserves observations,
next observations, seven-dimensional controller actions, environment rewards,
source bootstrap masks, transition identities, and recorded simulator clocks.
It retains action units and controller metadata. No missing action, reward,
next observation, or physical terminal flag is invented.

`source-mask` explicitly retains the dataset's `dones` behavior for bootstrapping.
The source mask means success or final stored sample. Physical termination and
time-limit truncation remain unknown. Read [the verified trainer contract](TRAINER_INTEGRATION.md#boundary-and-reward-semantics)
before interpreting these values. Choosing `--boundary-mode terminated` requires
real recorded termination information; it cannot repair missing evidence.

Every export requires a new output path. Its export metadata records episode membership,
exclusions, source and configuration identities, alignment parameters, and reward
selection. A changed trajectory or reward configuration creates a different
dependent export; it does not rewrite completed visual inference.

To export a downloaded robot bundle, validate it and select environment rewards:

```sh
bir bundle-check --bundle downloaded-robot.bir.json
bir training-export --bundle downloaded-robot.bir.json --output reviewed-robot.hdf5 \
  --reward-source environment --boundary-mode source-mask
```

The packaged two-demo bundle and ready-to-load HDF5 are described in
[the real examples](../examples/rl/README.md).

The packaged `robot.bir.json` already attaches the real replay video with its
explicit clock mapping. To build a new replay on a Linux CPU host with OSMesa,
follow [the rendering recipe](TRAINER_INTEGRATION.md#replay-and-render-verification).
Pass a JSON object keyed by demonstration ID containing each `replay.json` receipt
to `trajectory-import --replay-receipts receipts.json`. A mapping uses
`video_seconds = trajectory_seconds * scale + offset_seconds`; equal array index
alone does not establish alignment. Replay renders saved states; policy evaluation
starts from fresh simulator resets.

Existing annotation results can receive an independently prepared trajectory or
outcome without rerunning the visual engine:

```sh
bir bundle-attach --bundle annotated.bir.json --episode-id episode-001 \
  --trajectory trajectory.json --output attached.bir.json
```

Identity, camera/source, clock, shape, and boundary checks must pass. Use
`--outcome result.json` for a separately validated Rewarder artifact. An unrelated
recording or task cannot be attached merely by changing its display name.

## 4. Run the CPU IQL integration

Keep this environment separate from native annotation and Rewarder dependencies.
The tested CPU profile uses Python 3.11, PyTorch 2.5.1, MuJoCo 3.2.7, pinned
robomimic 0.5.0, and pinned robosuite 1.5.1. No CUDA or Apple MPS device is selected.

```sh
python3.11 -m venv .venv-rl
. .venv-rl/bin/activate
pip install cmake==3.31.6
```

On Linux, select the official CPU wheels before the requirements install if the
package manager would otherwise fetch CUDA libraries:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.5.1' 'torchvision==0.20.1'
```

```sh
pip install -r bir-resources/configs/rl/requirements-cpu.txt
python bir-resources/scripts/rl_cpu_smoke.py --dataset lift-iql.hdf5 \
  --output iql-smoke --steps 20 --rollouts 2 --horizon 400 --threads 2
```

The command uses two threads, zero loader subprocesses, at most two evaluation
episodes, and a 600-second execution ceiling. Heavy shared-cluster CPU work still
belongs in an authorized, bounded CPU allocation. Installation and downloaded
assets also count toward the applicable storage policy.

The real example completed 20 IQL updates on 117 transitions. Actor, critic, and
value parameters changed. The checkpoint restored the exact policy action and
optimizer state, then completed one further update. Two fresh 400-step simulator
episodes reported **0/2 task successes**. This proves the integration runs; it
does not show learned competence or policy improvement. See the [CPU receipt](../reports/rl-cpu-smoke.json).

The new output directory contains `result.json`, `config.json`,
`iql-checkpoint.pth`, and `iql-resumed.pth`. The script reloads its own checkpoints;
do not load an untrusted PyTorch pickle. The policy receives state observations
only. It does not receive captions, future outcomes, or retrospective FLORA scores.

## 5. Native annotation and optional Rewarder setup

The [native annotation quickstart](QUICKSTART.md) and [reproduction guide](REPRODUCTION.md)
retain the validated source, model, runtime, and setup instructions. Check the
hardware and approved resource allocation before heavy dependencies or model
downloads. The historical verified setup used three same-node RTX 3090 GPUs,
four CPU cores, and 128 GiB requested host RAM. One or two 24 GiB GPUs cannot
hold the full native Captioner. Three GPUs did not fit every tested episode.

Before downloading native models, inspect available GPU names and memory with `nvidia-smi`, confirm the documented hardware profile and free project storage, and obtain the required resource authorization. The installer does not perform these checks or allocate hardware.

Once an independently authorized native environment is ready:

```sh
bir source-check --source FLORA_SOURCE
bir annotate --videos VIDEO_FOLDER --source FLORA_SOURCE --run-dir runs/native
bir status --run-dir runs/native
bir export --run-dir runs/native --output native-labels.jsonl
```

The same `annotate` command resumes completed stages when the engine, runtime, and configuration are unchanged. Resume older inference runs with their recorded release. Saved results remain inspectable and exportable after an upgrade. A JSONL video manifest can
replace the folder input. A trajectory sidecar is additive; the old video manifest
and existing run identities remain valid.

The terminal Rewarder worker is independent. This CPU command prepares the exact
source/task request and does not download weights or run the model:

```sh
bir reward-prepare --episode-id rollout-001 --task 'Lift the cube off the table.' \
  --video rollout-001.mp4 --camera front --clock-offset 0 --whole-episode \
  --output reward-request.private.json
```

The request includes local paths; keep it private. `bir reward-score` defaults to
native execution disabled. **Do not invoke native scoring during this project's
GPU pause.** Its separate H100 80 GB profile, model/runtime pins, and one bounded
later validation proposal are in [REWARDER.md](REWARDER.md). Captioner fit and
speed do not establish Rewarder fit or speed.

For future validated native outcomes, `training-export --reward-source
flora-terminal` selects the separately versioned `(expected_score - 1) / 4`
training mapping. It preserves raw judgments and original environment rewards.
It rejects fixtures, failed/unknown results, source/task mismatches, and missing
actual episode-ending evidence. The current Lift examples have unknown physical
endings, so this option cannot silently treat their recording cuts as terminals.

No BC/environment-IQL/FLORA-IQL learning comparison has run. A later comparison
must hold trajectories and training settings fixed, use genuine native scores,
and evaluate multiple seeds on fresh held-out simulator episodes. The historical
[six-video 1.57× result](PHASE2_RESULTS.md) measures annotation throughput only.
