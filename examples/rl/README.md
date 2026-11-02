# Real Lift rollout examples

`lift-two-demos.bir.json` contains 117 real transitions from two pinned robomimic Lift proficient-human demonstrations. `lift-environment.hdf5` is their ready-to-load robomimic export with original environment rewards and explicitly selected source bootstrap masks. The first episode has a CPU-rendered video in `web/examples/robot.mp4` in the extracted resources. The second has no media attachment.

Observations, next observations, actions, rewards, states and dtypes are unchanged. Public copies omit the upstream model XML because it contains machine-specific asset paths. Its original hash is retained in bundle provenance. Re-render from the pinned source dataset, not from this sanitized HDF5. Physical termination/truncation are unknown. No native FLORA captions or Rewarder predictions exist for these robot examples.

Dataset revision: `74fa018461f479cd9fd15b924a16103012096203`. Source SHA-256: `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540`. Dataset and simulator attribution/license notices are in `web/examples/LICENSES.md` and the adjacent MIT license files.

See [the CPU integration recipe](../../docs/ROLLOUT_QUICKSTART.md) and [measured evidence](../../docs/ROLLOUT_RESULTS.md). Checkpoints in `lift-iql-smoke/` come from 20 actual IQL updates on these source arrays, followed by two fresh simulator episodes with 0/2 successes. They establish integration, not a competent policy.
