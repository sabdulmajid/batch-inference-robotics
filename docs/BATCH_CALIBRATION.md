# Bounded native caption batch calibration

This optional experiment checks whether two compatible native Captioner requests can share a model call. It uses only the frozen development video. It does not use final quality inputs or change the format-only adaptive gate. No GPU calibration result exists yet.

The coordinator must complete Gate C and preserve the measured final reproduction reserve before submitting this experiment. The script does not submit, cancel, or retry jobs. It rejects scheduler restarts and uses prepared assets in offline mode.

## Concrete experiment request

| Item | Proposed bound |
| --- | --- |
| GPUs | Three same-node 24 GiB GPUs matching successful Gate C hardware and actual Captioner placement |
| Wall time | 40 minutes, at most one submission |
| Reserved GPU cost | 2.0 allocated GPU-hours for three GPUs, including setup, failures, and idle time |
| CPU/RAM | Four CPU cores and 128 GiB RAM; Gate C measured 116,037,656 KiB peak host RSS (110.66 GiB) |
| Temporary storage | Existing 20 GiB task temporary-disk allowance |
| New persistent storage | Small private JSON/log receipts; no new weights or source videos |
| Source inputs | Exact frozen development manifest, currently `homer_41`; no final videos |
| Prepared tensors | At most 2 GiB source-plus-combined tensor payload; at most eight candidate shapes |
| Batch | Exactly two requests from one episode with equal native tensor shapes and token lengths |
| Inference | At most five generation calls: two singles, one native batch, one memory stress batch, one collector demonstration batch |
| Retries | Zero automatic retries |

This is a proposal within the existing resource envelope. Submission requires the coordinator's current ledger reservation and live scheduler/storage checks. Use the successful native placement and measured needs if hardware differs. Do not increase the global budget or omit final reproduction to run this optional experiment. The native generation timer is 300 seconds per call; the 40-minute allocation also allows loading and CPU work. The scheduler wall limit remains binding.

## Execution

```sh
PYTHONPATH=src python scripts/calibrate_caption_batch.py \
  --source-root "$FLORA_SOURCE" \
  --manifest "$DEV_MANIFEST" \
  --gate-evidence "$GATE_C_RECEIPT" \
  --output-dir "$CALIBRATION_RUN"
```

The output directory must be fresh. Use the pinned Python 3.12/CUDA 12.8 environment and verified project caches. The script checks Gate C artifacts, package versions, granted logical GPUs, physical GPU UUIDs, and topology. It preserves scheduler visibility. Keep the raw output directory private until review.

The script first runs the unchanged native caption planner and actual pinned processor on frozen development intervals. It substitutes a capture callback only for model generation. Captured placeholders never become saved annotations or inference evidence. CPU preparation occurs before model loading. The bounded search chooses the first two requests with different input digests and identical shapes, token lengths, video grids, profile, and contract. It searches within one episode. If no pair exists within the bounds, it stops without loading weights or writing an admission.

For the smaller measured pair, add these options to the command above:

```sh
  --signed-segmentation "$GATE_C_HOMER41_SEGMENT_CHECKPOINT" \
  --segment-indices 10 13 \
  --batch-headroom-gib 2
```

This optional source uses **signed TempSeg model-predicted intervals**, not independent references. The checkpoint's complete ordered intervals and temporal audit must match the verified Gate C `homer_41` annotation. Its source, instruction, model revisions, native profile, and duration must match the frozen development episode. The calibration records the checkpoint file SHA-256, stage digest, Gate C annotation SHA-256, and exact selected indices. It rechecks these bindings before generation and before admission publication. No adaptive reference, final input, interval boundary, frame plan, or native runtime module changes.

Indices 10 and 13 cover 45.5–46.5 seconds and 48.4–49.4 seconds. Actual CPU preparation measured 4,344 input tokens per request, with 12 BEFORE frames, 8 TARGET frames, and 12 AFTER frames. Each subset plan matched its plan in the full 15-interval episode. The distinct requests form an exact-compatible pair and retain 396,779,040 source-plus-combined tensor bytes. This additional CPU workflow took 23.742 seconds with zero model loads. These measurements establish smaller native inputs, not GPU feasibility.

The default batch safety headroom remains 4 GiB. An explicit 2 GiB pilot margin is a proposed allowance for this smaller pair, separate from the unchanged 4 GiB **weight-placement** allowance. It is not a measured guarantee or permission to skip checks. Both the actual native batch and full 512-token stress must leave at least the selected margin on every GPU, using peak reserved memory. Admission must also pass the current free/live/cache check, and the collector must stay within the measured peak. If any check fails, the script writes no admission. The coordinator must choose the pilot margin before execution from completed Gate C memory evidence and the remaining budget; do not lower it after seeing a failed batch.

For a pair, the script loads and verifies native Captioner once. Runtime processor settings must exactly match capture settings. Actual parameter placement must match Gate C. It then runs each original request through the unchanged native single-request helper and runs one true native batch. Per-row completion, request routing, and exact caption agreement are required. If agreement or physical headroom fails, it skips the stress pass to preserve the budget.

The separate memory stress pass uses a private copy of generation settings and sets `eos_token_id=None`. It keeps greedy generation, the same inputs, a 512-token limit, and the native 300-second timer. Both rows must reach all 512 tokens with exact input prefixes. This pass is **not native generation or quality evidence**. Model, processor, and persistent generation-configuration hashes must remain unchanged afterward.

If those checks pass, an in-memory measured admission is used for one collector demonstration. The unchanged native planner reconstructs the selected two intervals. The collector must execute a batch, preserve original tensor digests and captions, resolve both native event routes, and stay inside the measured memory envelope. The script does not retry a failed batch as singles.

Original captured and combined tensors are released before the demonstration prepares another pair. An unmatched demonstration request fails before any extra single-request inference. This preserves the tensor bound and five-call ceiling.

## Evidence and reuse

`calibration.json` retains input digests, reference or signed-segment indices, native event routes, interval provenance, processor identity, topology, model/residency verification, runtime bindings, captions, generated-token counts, memory baselines/peaks on every device, and all pass durations. It labels CPU capture and non-native stress separately. The coordinator must add final scheduler status and allocated resource consumption, including unsuccessful or idle work.

The script rechecks implementation hashes, its helper scripts, frozen inputs, and Gate C evidence before generation and before admission publication. The evidence identity includes those receipts. A changed staged release fails the calibration.

`admission.json` is written only after all required checks and cleanup succeed. No such file is shipped with the project. A missing pair, incomplete output, changed configuration, caption disagreement, insufficient headroom, or failed demonstration produces no admission. This does not turn on batching in the normal engine or CLI.

`bir.batching_contract.native_batching_contract` supplies a stable identity for later collector use. It pins native source/model identity, relevant adapter/placement/cache/batching implementation files, native geometry and generation rules, the human single-view episode boundary, and explicit collector bounds. Use `native_batching_contract_id` to derive its canonical digest. Actual processor/tokenizer settings, hardware, placement, and model settings are also bound by the runtime fingerprint. Runtime fingerprinting independently hashes the current relevant implementation files; an old caller-supplied contract ID cannot hide a code change.

Episode IDs, output directories, scheduler job IDs, and admission-file hashes are separate. This avoids a circular identity between evidence and the configuration it validates. Relevant runtime code or setting changes require new calibration. A CLI presentation change alone does not redefine model computation.

CPU calibration and independent review tests check bounded pair selection, distinct inputs, episode boundaries, stable contract identity, current code and physical GPU bindings, admission prerequisites, copied stress settings, full token-count requirements, live release changes, and restart rejection. They use synthetic tensors and stub generation. They do not establish GPU feasibility or native batch agreement.

Real CPU preparation also found a pair in the frozen development video. Each
request contains 8,504 input tokens and three temporal blocks. Rebuilding only
those two intervals preserved ordered tensors, prompts, frames, and timestamps.
The retained input and combined tensors use 793,540,128 bytes. The two local CPU
workflows took 111.740 seconds in total. See the
[sanitized preparation report](../reports/caption-batch-cpu-preparation.json).
This establishes input compatibility only. GPU generation and memory admission
remain unverified.
