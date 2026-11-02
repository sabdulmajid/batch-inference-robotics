# Portable rollout bundles and training exports

A rollout bundle joins visual annotations to recorded robot transitions. You can review the bundle in the browser, download it with its correction history, check it with the CLI, and export its recorded transitions to robomimic. Videos remain separate files. A video-only bundle stays useful for annotation review, but it cannot supply robot actions for training.

The first real-data adapter supports the official robomimic Lift proficient-human low-dimensional dataset at revision `74fa018461f479cd9fd15b924a16103012096203`. Its file is `v1.5/lift/ph/low_dim_v15.hdf5`, SHA-256 `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540`. Other dataset files are rejected by this adapter. This is one supported convention, not a universal robot-data converter.

## CPU commands

Use Python 3.11 or 3.12 and install the lightweight data extra in an isolated environment. From the repository:

```bash
python -m pip install '.[data]'
bir resources --output-dir bir-resources
python bir-resources/scripts/prepare_rollout_example.py \
  --dataset data/lift.hdf5 --download-only
bir trajectory-import --dataset data/lift.hdf5 \
  --demos demo_0 demo_1 --output lift.bundle.json
bir bundle-check --bundle lift.bundle.json
bir training-export --bundle lift.bundle.json \
  --reward-source environment --boundary-mode source-mask \
  --output lift.training.hdf5
```

The dataset download has a pinned byte count and hash. Import copies the selected source observations, actions, next observations, original rewards, controller metadata and source bootstrap mask. It does not invent videos or captions. A video replay requires the separate supported CPU simulator/rendering setup. Replay shows recorded states; it is not a policy evaluation. See the [training recipe](TRAINER_INTEGRATION.md) for optimizer, checkpoint and simulator checks.

To attach a real rendered replay, pass a JSON object keyed by source demo ID with the corresponding `replay.json` receipt as its value:

```bash
bir trajectory-import --dataset data/lift.hdf5 --demos demo_0 \
  --replay-receipts replay-receipts.json --output lift-video.bundle.json
```

The receipt must bind the same dataset hash, demonstration ID and complete recorded simulator-time sequence. It gives the camera, video hash, duration and explicit video-clock mapping. The importer does not align streams by array position alone.

To package an existing annotation run:

```bash
bir bundle-pack --run-dir my-run --media-info media-info.json \
  --output annotations.bundle.json
bir bundle-attach --bundle annotations.bundle.json --episode-id rollout-001 \
  --trajectory rollout-001.trajectory.json --output with-actions.bundle.json
bir bundle-check --bundle with-actions.bundle.json
```

`media-info.json` is an object keyed by the existing annotation episode IDs. Each value has `task` and `media`. Each ordered media entry has `camera`, a plain `filename`, actual `source_sha256`, and measured `duration_s`. Camera names and hashes must match the saved visual results. For a new source, obtain its real duration and hash; do not copy values from an example. Trajectory sidecars must follow the contract below. An attachment writes a new bundle and preserves the original annotation results and review journal.

An outcome attaches separately:

```bash
bir bundle-attach --bundle with-actions.bundle.json --episode-id rollout-001 \
  --outcome reward-results/REQUEST_ID/result.json --output with-outcome.bundle.json
```

The outcome must match the episode, exact task and ordered camera hashes. Native Rewarder execution is still paused and unvalidated. The [Rewarder guide](REWARDER.md) describes CPU request preparation and the later native gate. A fixture or unavailable outcome cannot supply training rewards.

All output bundle and HDF5 destinations must be new. Commands refuse to replace existing files. Use a new destination after changing a trajectory, reward choice or boundary rule. Run `bir bundle-check` on the exact browser-downloaded bundle before training export.

## Bundle envelope

`bir_rollout_bundle_v1` has exactly four fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `bir_rollout_bundle_v1` |
| `payload_json` | Immutable JSON string holding the generated records and attachments |
| `payload_sha256` | SHA-256 of the exact UTF-8 `payload_json` bytes |
| `reviews` | Ordered, separately hashed review journal |

The string preserves the original payload bytes across browser download. Review changes do not require a browser to rewrite floating-point arrays or original predictions. Hashes detect changed content; they do not authenticate a third party's claim about its data or model execution.

The payload has exactly `schema_version: bir_rollout_payload_v1`, `episodes`, and `provenance`. It contains 1–100 episodes. A bundle file is limited to 64 MiB. Each episode has exactly:

- `episode_id`: unique identity used by its annotations and attachments.
- `task`: plain-language episode instruction.
- `media`: zero to four ordered camera entries with `camera`, plain `filename`, `source_sha256` and `duration_s`. A filename is not a path or remote URL.
- `records`: original `bir_label_v1` visual records. Their record IDs bind their contents and original media hashes. Intervals use video seconds.
- `trajectory`: one `bir_trajectory_v1` sidecar or null.
- `outcome`: one independently versioned Rewarder result or null.

A missing trajectory or outcome remains missing. A failed caption remains failed. Existing generated predictions and separate human corrections are not interchangeable.

## Trajectory sidecar

`bir_trajectory_v1` uses explicit transition rows. There are at most 10,000 rows per episode. State observations and actions are finite numeric arrays; this first convention does not put images or text in policy observations.

| Field | Requirement |
| --- | --- |
| `episode_id`, `trajectory_id` | Episode binding and digest of every other sidecar field |
| `source` | Dataset/revision/file SHA-256 and original `demo_id` |
| `transition_ids` | Stable digests of source hash, source demo, original transition index and observation/next-observation times |
| `observations`, `next_observations` | Matching named arrays with one row per transition; next observations must be supplied |
| `actions` | One fixed-width row per transition |
| `observation_times_s`, `action_times_s`, `next_observation_times_s` | Explicit, finite, strictly increasing clocks |
| `terminated`, `truncated` | One boolean or unknown null per transition; do not infer them from a video ending |
| `environment_rewards` | Original rewards, or null if not recorded |
| `source_bootstrap_stop` | Original dataset bootstrap mask, when present; not relabeled as true termination |
| `boundary` | Actual known boundary kind and source-mask semantics where used |
| `action_spec` | Representation, one unit per action axis, controller metadata, and explicit shared lower/upper bounds |
| `environment_metadata` | Original supported environment configuration |
| `clock_mappings` | Explicit mapping for every attached camera |
| `array_dtypes` | Original numeric storage types, such as `<f4`, `<f8` or `<i8` |
| `simulator_states` | Optional recorded simulator states with the same transition count |
| `source_attributes` | Optional source `model_file` or `ep_meta`; reserved export attributes cannot be replaced |

For every transition, observation time must be no later than action time, and action time must precede next-observation time. The next-observation time must meet the following transition's observation time within `1e-7` seconds. A true termination or truncation cannot appear before another transition in the same sidecar. Do not concatenate separate episodes.

The supported robomimic importer reads simulator times from the stored MuJoCo states and checks the 20 Hz controller interval. It obtains next-observation values from the source `next_obs` datasets. The next-observation clock follows the checked simulator interval. It does not make next observations by shifting rows and copying the final observation.

Each camera clock mapping contains `camera`, `video_sha256`, positive `scale`, `offset_s`, and a nonempty `source` explanation. The formula is:

```text
video_seconds = trajectory_seconds * scale + offset_s
```

The mapped trajectory must lie in the bound video. The validator allows at most 0.1 seconds at the video end for the recorded final transition interval and media duration precision. The replay receipt records every rendered source timestamp. The same row number in two arrays is never sufficient evidence of alignment.

The validator checks action bounds, array shapes, finite values and dtype representability. Export refuses a cast that would change a recorded value. This includes fractional actions cast to integers, overflow, and integers too large for a default float64. A float32 source is accepted when its JSON values exactly represent the stored float32 values. Missing dtype entries use a checked float64 default. A missing observation, action or next observation is an error, not a generated value.

## Review history and resumed annotations

Every review journal entry has `schema_version: bir_review_v1`, a consecutive `revision`, the original `record_id` and matching `generated_record_sha256`, correction text or null, action, reviewer, timestamp, previous-entry hash and entry hash. The hash chain preserves order. A null correction means restore the generated caption.

The original generated caption remains in the payload. The latest applicable correction determines the effective caption. The full journal stays in the bundle and in local `reviews.jsonl`. Annotation JSONL exports retain the latest applicable correction; they do not contain the full correction history.

If captioning resumes and changes a generated record identity, an old edit does not transfer automatically. It remains a historical, unapplied review. It is visible as such and does not block unrelated records. A researcher can inspect the changed generated record and create a new correction for its new identity. Bundle attachment preserves historical entries, including entries whose records are no longer current.

Selected local media are separate from the JSON bundle. The browser checks file content against the declared SHA-256 before playback. Importing the bundle and explicitly selecting local video files is the supported own-data path. Keep private operational files, model weights and credentials out of a public bundle.

## Training reward and bootstrap choices

Training export requires both flags. There is no implicit reward source or boundary interpretation.

| Option | Behavior |
| --- | --- |
| `--reward-source environment` | Preserve original recorded environment rewards; reject missing rewards |
| `--reward-source flora-terminal` | Require valid native outcome evidence and an evidenced true episode-ending transition; place the separate mapped value only on that final transition |
| `--boundary-mode source-mask` | Preserve the explicitly selected robomimic source mask and record its exact interpretation |
| `--boundary-mode terminated` | Require known termination and truncation flags; only true termination stops bootstrapping |

The selected trainer uses `reward + discount * (1 - dones) * V(next_obs)`. A true time-limit truncation uses the recorded next observation and continues bootstrapping. The first source dataset's `dones` mark success or the last recorded sample. They do not establish which recordings actually terminated or hit a time limit. Import therefore keeps `terminated` and `truncated` unknown. Select `source-mask` for a source-preserving smoke test. The importer cannot turn these demonstrations into valid terminal-FLORA training examples merely because the videos end.

The HDF5 output follows the existing robomimic loader structure: `data/demo_N/{actions,rewards,dones,obs/*,next_obs/*}`, environment metadata, and `mask/train`. It also stores transition identities, clocks, original environment rewards, source mask, unknown flags as `-1`, source/clock/action metadata, original visual records, outcome evidence, mapping, and full review history under separate `bir` datasets or attributes. Captions and retrospective rewards are not policy observation fields.

Export provenance records bundle/review identities, dataset membership, excluded video-only episodes and reasons, trajectory IDs, selected reward/mapping/boundary rules, and a new export ID. Source or reward changes invalidate the dependent export identity. They do not rewrite completed annotation checkpoints. Mixed environments require separate exports. A successful loader or short CPU optimizer test demonstrates integration, not learned task competence.
