# Optional episode outcome scores

The Rewarder companion worker attaches one video-derived outcome judgment to a complete robot episode. It does not depend on segmentation or captioning. A failed caption run does not prevent a later outcome request. It does not change the annotation database.

**Native Rewarder execution awaits validation. GPU work remains paused.** This release tests the adapter, input identity, parser, failure states, recovery, and separate reward mapping on CPU. Test fixtures are not FLORA predictions. Existing Captioner execution and speed measurements do not validate Rewarder, its memory use, or policy learning.

## Pinned contract

| Part | Pin or behavior |
| --- | --- |
| NVIDIA FLORA source | `129a2ada74c526ee4692f3d1b111bea6ca19d5b7` |
| Model | `nvidia/Flora-Rewarder` at `031299e46e452fc0cedb4e0975eb9fa95fc37783` |
| Model form | Full Qwen3.5-27B fine-tune; upstream BF16 execution; native mixed checkpoint dtypes preserved |
| Model size | Hub metadata lists 27,781,428,976 parameters and about 55.6 GB of files; this is not a runtime memory measurement |
| Runtime | Python 3.12; vLLM 0.24.0; Transformers 5.8.1; qwen-vl-utils 0.0.14; decord 0.6.0; `av>=15`, `numpy<3`, `Pillow>=11,<13` |
| First hardware profile | One H100 80 GB, tensor parallel size 1; upstream reports this configuration, but this project has not executed it |
| Native entry point | `python -m rewarder.batch_utils.terminal_reward` |
| Input | Original complete episode; explicit task; 1–4 synchronized camera videos in training order |
| Frames | Upstream whole-episode uniform sampling, up to 64 frames at up to 10 FPS; even count, at least two |
| Generation | Upstream greedy constrained `Score: 1` through `Score: 5`; no rationale; model length 16,384; seed 0 |
| Configuration | [`configs/rewarder-terminal.json`](../configs/rewarder-terminal.json), `terminal_native_h100_v1` |

The adapter calls upstream code. It does not substitute prompts, change camera geometry, resample source videos, reduce temporal input, quantize weights, or offload weights. Upstream retains its native sampling and collage behavior. Its original tolerances allow about 1% frame-rate or length differences and up to two repeated tail frames. The caller must establish that camera recordings start together. Matching lengths alone do not establish synchronization.

For two cameras use `left, right`; for three use `left, center, right`; for four use `top, front, left, right` or `top, bottom, left, right`. One camera can use a descriptive name such as `front`. Every clock offset must explicitly be zero. This first profile does not accept video cuts or shifted clocks. It does not infer which camera is which.

The weights and model use are subject to NVIDIA's research license. Read the pinned model card, license, and upstream third-party notices before use.

Primary references: [pinned inference guide](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/rewarder/batch_utils/README.md), [pinned terminal implementation](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/rewarder/batch_utils/terminal_reward.py), [pinned model files](https://huggingface.co/nvidia/Flora-Rewarder/tree/031299e46e452fc0cedb4e0975eb9fa95fc37783).

## Prepare a request without a GPU

This API needs only the lightweight package and original local files. It hashes each file and validates the declarations. It does not download or load the model.

```python
import json
from pathlib import Path
from bir.rewarder import prepare_request

request = prepare_request(
    episode_id="rollout-001",
    task="Lift the cube off the table.",
    videos=["rollout-001-front.mp4"],
    camera_order=["front"],
    whole_episode=True,
    clock_offsets_s=[0],
)
Path("reward-request.private.json").write_text(json.dumps(request, indent=2))
```

`local_sources` contains local paths. Keep the request private. The portable `contract` contains the task, episode ID, ordered camera names, source hashes and sizes, clock declarations, and pinned configuration. Its digest is the request ID. Moving a video without changing its bytes preserves that identity. Changing the task, source bytes, cameras, or configuration changes it. An outcome must also match the attached rollout's episode, task and ordered camera hashes before training export.

Video-only input can obtain visual annotations and, after native validation, a terminal judgment. A training export still needs recorded actions, observations, episode boundaries and the trajectory contract. The Rewarder cannot supply missing actions or terminal flags.

## Separate environment and future execution

Do not run this setup as part of the paused release. First obtain an explicit GPU-pause and submission-limit revision. Then check that the assigned hardware fits the supported profile before an expensive installation or model download. Keep the Rewarder environment separate from the annotation environment. Install the pinned upstream `rewarder/batch_utils/requirements.txt` under Python 3.12. Stage the pinned checkpoint and its `SHA256SUMS` in authorized storage before an allocation. This is about 55.6 GB of additional model files before environment and compilation caches; it must fit the unchanged storage budget.

The worker requires a local model directory. It sets Hugging Face and Transformers offline mode. It checks the actual runtime versions, visible H100 memory, clean source revision, and every model/processor checksum against the pinned checksum manifest. These checks do not make the configuration project-validated; a successful native probe is still required.

After the required authorization and setup, the execution API is:

```python
from bir.rewarder import run_terminal

result = run_terminal(
    request, "reward-results",
    python="/prepared/rewarder-env/bin/python",
    flora_source="/prepared/Flora",
    model_dir="/prepared/Flora-Rewarder",
    allow_native=True,   # Explicit execution; default is disabled.
    timeout_s=1800,
)
assert result["status"] == "complete", result
```

Use only a scheduler allocation when required by the compute site. This API does not submit jobs or override project resource policy. The sample paths above are placeholders. The first profile does not claim support for three RTX 3090 GPUs or other hardware. Setup and model requirements are not removed by installing the lightweight package.

## Results, recovery and failure states

Each request owns `reward-results/<request_id>/result.json`. Each attempt has a separate `attempt-0001`, `attempt-0002`, and later directory. The portable result records:

- Request identity, original ordered source hashes, task, camera clock declarations and configuration.
- Execution kind (`native` or `test_fixture`), exact runtime versions and verified model/source identity when native execution starts.
- Raw upstream JSON, score probabilities, expected score, raw text, duration, sampled-frame count, and raw-file checksum.
- Parse validity, explicit status, attempt number, result digest and completion time.

`native.log`, `jobs.jsonl`, `process.private.json`, and `failure.private.json` are operational files. They can contain local paths. Keep them out of public run bundles and Git. The portable result contains no local video paths from the request.

An invalid parse, failed process, unsupported runtime, missing input, timeout, or incomplete output does not become score zero. A failed result has `parse_valid=false` and null score, expected score and progress. Invalid input at request preparation raises an error before inference. The raw output is retained for diagnosis when it exists. The parser rejects extra/missing fields, wrong episode/task, nonfinite numbers, inconsistent scores or probabilities, and malformed answers. It accounts for upstream four-decimal probability rounding.

A completed request resumes without generation. Source bytes and result integrity are checked first. A failed or interrupted attempt requires `retry=True`. The prior attempt remains. An interrupted fixture attempt becomes `interrupted`; a partial output is never promoted automatically. Every interrupted native attempt requires task-allocation reconciliation before a retry in a fresh output directory. A PID alone cannot prove cross-host ownership. The worker does not cancel a process on recovery merely because its PID exists. Reconcile the task-owned scheduler job before retrying. A normal timeout or keyboard interrupt terminates the worker's own subprocess group. No retry is authorized during the current GPU pause.

The portable checks are available as `load_result(path)`, `validate_result(result)` and `validate_binding(result, episode_id=..., task=..., cameras=[{"camera": ..., "source_sha256": ...}])`. Integrity digests detect accidental changes. They are not cryptographic proof of a third party's claimed model execution.

## Separate training-reward mapping

`terminal_reward(result, mapping="expected_progress_v1")` returns a separate mapping artifact. It requires a valid native result and its asset/runtime receipt. Failed, unavailable, unexecuted and test-fixture results are rejected.

The value is `(expected_score - 1) / 4`. It is a retrospective normalized expected score in `[0, 1]`. It is **not** a calibrated success probability, an environment reward, or instantaneous causal feedback. The raw five-class probabilities and judgment remain unchanged. The mapping record names its version and source-result digest.

The trajectory exporter must explicitly select this reward source and place the value only on the corresponding actual episode-ending transition. It must retain the original environment rewards separately. A video cut does not make a terminal transition. Time-limit truncation and bootstrapping remain governed by the recorded trajectory and the trainer's contract. Captions and retrospective scores must not enter policy observations by default.

Dense and rubric scoring are not supported in this release. In particular, the upstream dense example can assign a normal score to an invalid clip. This worker does not use that fallback or expose it as valid evidence.

## Executed CPU checks

`python -m pytest tests/test_rewarder.py -q` tests request/config identity, camera/clock declarations, source mismatches, strict parsing, probability rounding, nonfinite and partial output, fixture provenance, independent resume, interrupted attempts, live-process ambiguity, source changes during execution, portable binding and reward-mapping version checks. A real CPU subprocess writes a clearly labeled test fixture through the output boundary.

The pinned upstream terminal entry point was also executed with `--help`. Its parser accepted the adapter's options without model loading. A two-second, 20-frame generated CPU video was decoded through the actual pinned `core.Collage`, sampled through `terminal_frame_indices`, and parsed through `parse_terminal`: 20 selected frames, tensor shape `[20, 320, 320, 3]`, original 2.0-second clock preserved. This is a preprocessing contract check on a synthetic test video. It is not native model inference or a real robot result.

## One bounded later native validation proposal

Execution needs explicit permission to resume GPU work and increase the exhausted submission limit from 16 to at least 17. Unused GPU-hours alone do not authorize another submission.

Proposed probe: one H100 80 GB, one node, one GPU, at most 30 minutes and 0.5 allocated GPU-hours. Prepare and verify assets before the allocation. Use at most three short, complete, action-bearing simulator episodes with fixed tasks and source hashes, including a known successful and unsuccessful outcome when available. Preserve all native settings. No quantization, frame reduction, CPU/disk inference offload, dense scoring, or parameter search. Final sizing and admission must use the existing ledger and actual site policy.

Expected artifacts: scheduler terminal status and elapsed allocation; source/model/runtime checks; hardware and memory measurements; the native JSONL and portable results; parser/binding checks; a no-generation resume; mapped training export with real episode boundaries; and a clean CPU training-loader check. Pass means every intended episode yields a valid, correctly bound native score, no prohibited adaptation, reproducible raw records within the declared native determinism limits, unchanged resume evidence, and a valid export. An OOM, timeout, invalid parse, missing episode, identity mismatch, or excess allocation is a failed probe. It does not silently count as zero reward.

This probe would establish integration and feasibility only. It would not establish reward calibration, Rewarder acceleration, or improved policy performance. A later controlled BC / environment-reward IQL / genuine-FLORA-reward IQL comparison needs the same trajectories, fixed training settings, held-out simulator episodes and multiple seeds. That comparison has not occurred during the GPU pause.
