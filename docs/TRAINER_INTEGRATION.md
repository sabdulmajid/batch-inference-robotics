# Real rollout to CPU IQL

The first supported training profile is the official robomimic **Lift / proficient-human / low-dimensional / robosuite 1.5** dataset. It contains recorded Panda robot control actions in a simulator. The policy uses state observations. It does not use captions, rendered frames, or outcome judgments as observations.

The small recipe tests the data loader, actual IQL optimizer updates, checkpoint recovery, and policy execution in a fresh simulator. It is not a learned-competence or policy-improvement result.

The released example passed that test on macOS arm64 with Python 3.11.12 and two CPU threads. Both episodes preserved all 46 source arrays, including their dtypes. The final smoke made 20 IQL updates, recovered the checkpoint and optimizer exactly, and made one more update. Fresh policy success was **0/2** at 400 steps per episode. Its measured wall time was 7.98 seconds with installed dependencies and a warm simulator cache; this is not a training-speed benchmark. See the [CPU receipt](../reports/rl-cpu-smoke.json), [array and clock checks](../reports/rl-data-roundtrip.json), and [small checkpoint artifacts](../examples/rl/lift-iql-smoke/README.md). Linux replay rendering is a separate validation result.

## Supported data and exact versions

| Component | Pinned version |
| --- | --- |
| Dataset | `robomimic/robomimic_datasets`, revision `74fa018461f479cd9fd15b924a16103012096203` |
| Dataset file | `v1.5/lift/ph/low_dim_v15.hdf5`, 21,084,088 bytes |
| Dataset SHA-256 | `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` |
| robomimic | 0.5.0, commit `d309eaecc18acf4152a830a895a6984b8ac71b05` |
| robosuite | 1.5.1, commit `a071383d53568ab798eb315c0e95357911be922d` |
| Python / PyTorch / MuJoCo | 3.11 / 2.5.1 / 3.2.7 |

The complete source file has 200 demonstrations and 9,666 transitions. The small example selects `demo_0` and `demo_1`: 59 and 58 transitions. The [official dataset card](https://huggingface.co/datasets/robomimic/robomimic_datasets) declares MIT licensing. Retain the source attribution and license when redistributing its examples. See also the [dataset documentation](https://robomimic.github.io/docs/datasets/overview.html).

The recorded seven-dimensional action is a **normalized controller command**, not a joint-position observation. The Panda uses the `BASIC` composite controller with right-arm `OSC_POSE`. The first three components are world-frame translation deltas, scaled by the controller to ±0.05 metres. The next three are rotation deltas, scaled to ±0.5 radians. The last component is the normalized gripper command. Keep the original values and the full controller metadata.

The controller runs at 20 Hz. The first element of each saved MuJoCo state is its recorded simulation time. The replay helper checks these actual times against the controller rate. Its receipt records every frame time and maps the video clock to those times. It does not infer video alignment from equal array indices.

## Boundary and reward semantics

Preserve the original environment rewards and `dones` exactly for the explicit `source-mask` export mode. In this source convention, `dones` means **success at the next state, or the last stored sample**. It does not identify a physical terminal event or a time-limit truncation. The source has five success-marked samples near each example's end. The trajectory bundle therefore keeps physical termination and truncation unknown.

The [pinned conversion code](https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/robomimic/scripts/dataset_states_to_obs.py) documents and implements this convention. The [pinned IQL implementation](https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/robomimic/algo/iql.py) uses `reward + (1 - done) * discount * next_value`. Thus the preserved source mask is a bootstrapping convention, not an inferred event label.

A FLORA terminal outcome cannot be placed on an arbitrary video cut or an episode with unknown ending semantics. This example's original environment-reward export is usable now. Its FLORA-derived reward export needs genuine native scores **and** a supported, explicit episode-ending contract. A successful test of the environment-reward path does not validate native Rewarder inference.

## Install the isolated CPU profile

Run these commands from a versioned source release. Keep this environment separate from the FLORA inference environment.

```sh
uv venv .venv-rl --python 3.11
uv pip install --python .venv-rl/bin/python cmake==3.31.6
export PATH="$PWD/.venv-rl/bin:$PATH"
uv pip install --python .venv-rl/bin/python -r configs/rl/requirements-cpu.txt
.venv-rl/bin/python scripts/prepare_rollout_example.py \
  --dataset data/lift-ph-low_dim_v15.hdf5 --download-only
```

`requirements-cpu.txt` pins the algorithm, simulator, and main numeric dependencies. `resolved-macos-py311.txt` records the transitive packages installed for the local smoke test; its macOS-specific packages are not a Linux installation recipe. On Linux, install the CPU PyTorch wheel from the official CPU wheel index before the remaining requirements if the package manager would otherwise fetch CUDA libraries. CPU training always uses `torch.device("cpu")`; it never selects CUDA or MPS.

On Linux, the transitive `egl-probe` package can build from source. It needs a C/C++ compiler and CMake on the build process's `PATH`. The first Linux setup attempt failed because CMake was absent. Before installing the main requirements, use a prepared compiler toolchain and:

```sh
uv pip install --python .venv-rl/bin/python cmake==3.31.6
PATH="$PWD/.venv-rl/bin:$PATH" uv pip install \
  --python .venv-rl/bin/python -r configs/rl/requirements-cpu.txt
```

This build prerequisite does not enable GPU inference. Software replay also needs the system OSMesa library described below.

First import the source and export its selected trajectories with the project's trajectory commands in the [rollout quickstart](ROLLOUT_QUICKSTART.md). The training command takes that exported HDF5 file, not a browser label-only JSONL:

```sh
.venv-rl/bin/python scripts/rl_cpu_smoke.py \
  --dataset output/lift-environment.hdf5 \
  --output output/iql-smoke \
  --steps 20 --rollouts 2 --horizon 400 --threads 2
```

The output directory must be new. Results are never replaced. The command writes:

- `result.json`: source file digest, dataset membership, network-update checks, loss values, checkpoint checks, and fresh simulator outcomes.
- `config.json`: the actual robomimic configuration.
- `iql-checkpoint.pth`: model, critic, value, optimizer, and scheduler states after 20 updates.
- `iql-resumed.pth`: the recovered state after one additional optimizer update.

It checks an exact policy action before and after reload and an exact optimizer-state round trip. It then runs the first saved policy in two fresh simulator resets, with seeds 101 and 202. Each rollout has a 400-step evaluation limit. Simulator task success comes from `env.is_success()["task"]`. The command does not load demonstration states for policy evaluation. A short test can correctly report zero successes.

Only load trusted PyTorch checkpoints. Their serialization format can execute Python code. The script reloads only the checkpoint that it just created.

## Replay and render verification

Replay is a separate operation. It renders the source's saved states and does not execute the trained policy. The software-rendering path needs Linux with OSMesa, for example the system `libosmesa6` library. On a shared cluster, use a bounded CPU scheduler allocation.

```sh
.venv-rl/bin/python scripts/prepare_rollout_example.py \
  --dataset data/lift-ph-low_dim_v15.hdf5 \
  --demo demo_0 --output output/lift-replay
```

The helper sets OSMesa software rendering and checks the actual renderer. It writes a 256×256 `replay.mp4` and a `replay.json` receipt. The receipt binds the source hash, video hash, camera, clock, frame count, and render settings. The helper deliberately rejects macOS's accelerated graphics path during the GPU pause. It labels native FLORA captions and Rewarder output unavailable.

## Later controlled learning experiment

No BC-versus-IQL learning comparison is claimed by the smoke test. A later bounded study should use the same frozen trajectories, state observations, train/test seeds, optimizer-step counts, and evaluation horizon for: behavior cloning; IQL with the preserved environment rewards; and IQL with genuine FLORA-derived rewards under a valid terminal mapping. Keep raw scores and reward mappings separate. Use multiple independent training seeds and fresh held-out simulator resets. Publish success counts and uncertainty, including failed runs. Keep that study separate from the historical six-video annotation speed result.
