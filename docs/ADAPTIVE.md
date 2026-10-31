# Caption spatial-budget experiment

This module tests a smaller spatial budget for Captioner while retaining every selected source frame and its timestamp. For example, a 40-frame target can use fewer visual tokens while a short context block stays unchanged. This is an experimental path. It does not replace the native reference.

The experiment has a separate fixed-reference interval API. The native CLI default remains unchanged. Complete the native end-to-end gate before GPU evaluation. No quality or GPU performance result exists yet.

## The exact processor control

The pinned FLORA loader uses `Qwen3VLProcessor` and `Qwen3VLVideoProcessor` from Transformers 5.8.1. Its video configuration sets `shortest_edge=4096` and `longest_edge=25165824`.

Despite these field names, the Qwen3VL video resize function applies the limits to the whole temporal block: frame count × height × width. The spatial dimensions also align to a 32-pixel grid. Reducing the configured limit by half does not necessarily reduce a native input. FLORA has already limited each single-view source frame to 262,144 pixels before the processor. See the [pinned video processor](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py) and [native frame preparation](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/captioner/batch_utils/vendor/flora_unified/unified_input.py).

The initial compact preset is `compact_block_5242880`. It limits each block to **5,242,880 total pixels**. This equals 40 × 131,072 pixels. It is not a promise of half the spatial area for every frame or block. Short blocks can remain unchanged. Grid alignment also changes the realized fraction.

`CaptionSpatialProcessor` supplies one supported **flat** `size` argument to the unchanged processor call:

```python
processor(
    text=text,
    videos=frames,
    video_metadata=metadata,
    do_sample_frames=False,
    return_metadata=True,
    return_tensors="pt",
    size={"shortest_edge": 4096, "longest_edge": 5242880},
)
```

Do not add only `videos_kwargs={"size": ...}` to the native flat call. In Transformers 5.8.1, a nested modality dictionary suppresses flat arguments for that modality. A real CPU test found that this discarded `video_metadata` and `do_sample_frames=False`. It reduced a 40-frame target to 14 sampled frames. The adapter rejected the changed temporal grid. The flat argument preserves the native calling convention. This behavior follows the pinned [argument merge implementation](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/processing_utils.py).

The proxy accepts the native keyword call with RGB PIL frame blocks. It changes no processor object, input image, frame list, source clock, prompt, patch size, or temporal patch size. It rejects temporal resampling and conflicting size settings. Each call records source-frame and clock hashes before and after processing. It checks the actual compact `video_grid_thw` result against the pinned resize geometry.

Audit records separate the native grid predicted from geometry from the actual compact grid. They report actual compact visual-token counts, realized spatial fractions, unchanged blocks, and processor time. A geometry prediction is not an observed native inference measurement.

## CPU validation

Twelve tests passed in an isolated CPU environment. The processor test used the actual Transformers 5.8.1 classes and tokenizer/processor files from Captioner revision `fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6`. No model weights were downloaded for this test. Fixed-reference routing and fallback tests use generation stubs. They do not validate model inference.

The test had two 512 × 512 RGB blocks: four context frames and 40 target frames. Both used explicit 8 FPS source clocks.

| Block | Actual native grid `[T,H,W]` | Actual compact grid `[T,H,W]` | Spatial area fraction |
| --- | --- | --- | --- |
| Context, 4 source frames | `[2,32,32]` | `[2,32,32]` | 1.0 |
| Target, 40 source frames | `[20,32,32]` | `[20,22,22]` | 0.47265625 |

`T` counts temporal patches, each containing two source frames. The tests checked identical `T`, frame bytes, source metadata, and nonvisual token sequences. The original processor configuration also remained unchanged. This validates a CPU preprocessing operation. It does not establish GPU compatibility, caption fidelity, semantic quality, or a runtime speedup.

The test environment used Python 3.13, torch 2.11.0, torchvision 0.26.0, Transformers 5.8.1, and Hugging Face Hub 1.33.0 on a local CPU. This is separate from the pinned Python 3.12/cu128 inference environment. The private operation records retain the full package list. Tests skip the real-processor case when the optional packages or pinned processor assets are absent; they never download assets during a test.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_adaptive.py' -v
```

## Fixed-reference integration

`run_reference_interval` accepts a prepared human single-view episode, an explicit reference interval, its stable index, a frozen-contract SHA256, and one policy: `native`, `compact`, or `adaptive`. The caller must load and time the verified native Captioner first. The function calls the unchanged upstream `caption_single` planner for that interval. A short session proxy returns `CaptionSpatialProcessor` only for a compact pass. The original persistent session, native processor, and model remain unchanged. The proxy delegates `apply_chat_template`, tokenizer access, and other attributes to the same processor.

Reference intervals are labeled `fixed_independent_reference_interval`. They are not TempSeg predictions and cannot establish the complete native engine gate. The unchanged planner selects context from the original full video. No source clip is substituted. Unsupported intervals retain explicit status and remain in the evaluation denominator.

Each result records episode ID, reference index, exact interval, policy, source hashes, freeze hash, caption status, all attempted passes, the format gate, spatial receipts, native input-plan audits, and generation events. The generation events include actual input and generated token counts. `policy_wall_seconds` covers the full synchronized interval path, including decoding, image transport, processor/hash work, generation, gate, and fallback. Shared model loading and episode validation must be recorded separately by the caller. Attempt times describe nested parts of the policy path; do not add their sum to the full policy time.

The function uses a process lock and restores its temporary runtime-log hook after each policy. Every original runtime event still reaches the private log. Native batch admission rejects the spatial proxy class, so a compact experiment cannot reuse a native microbatch memory admission.

Use a new run contract and state directory for each experiment. Record the preset, gate rule, source/model revisions, and fixed intervals. Freeze the development/test manifests, evaluator, margin, and routing rule before the final run. Update the evaluation configuration to describe the total-block budget; its earlier uniform `compact_area_fraction` proposal is not the implemented setting.

## Format-only fallback

`run_compact_then_native` runs compact generation first. The initial gate accepts a complete, nonempty caption string. An incomplete generation or invalid empty output triggers native fallback. Other failures, including out-of-memory failures, stop the work item. They do not trigger an automatic more expensive rerun.

The caller must map the native generation helper's incomplete-output exception to the declared `invalid_output_errors` tuple. A valid string is only a format check. The gate does not assess whether the caption is true. It never receives the native output before its decision.

Every attempted compact pass, gate decision, and native fallback has a measured application duration and an explicit status. Failed work remains in the record. For CUDA use, supply a synchronization callback for every visible device so timed work finishes at the measurement boundary. Scheduler allocation GPU-hours remain a separate measurement and include idle time. Do not add a job total again after allocating it across passes.

Independent references are still required for quality evaluation. Audit accepted compact outputs as well as fallback cases. Charge that audit work separately. If captions remain structurally valid, this baseline gate can have zero fallback even when captions are wrong. A negative or inconclusive quality result is valid.

## Bounded runner

`scripts/adaptive_benchmark.py` runs the frozen split inside a coordinator-owned allocation. It never submits a job. It requires a complete fresh Gate C receipt and the saved annotations with verified TempSeg and Captioner audits. GPU names, memory sizes, and placement headroom must match that receipt. The runner checks the pinned Python, framework, processor, and CUDA environment. Model downloads are disabled.

Prepare a manifest with exactly the frozen split's full source videos and original instructions. The runner checks their hashes against the freeze. It loads Captioner once, disables decoded-frame caching for equal per-policy preparation work, and rotates native/compact/adaptive order by reference index. It records setup, validation, model load, cleanup, and GPU memory separately from interval-policy timers.

```sh
PYTHONPATH=src python scripts/adaptive_benchmark.py \
  --source-root "$FLORA_SOURCE" --manifest "$DEV_MANIFEST" \
  --split development --gate-evidence "$GATE_C_RECEIPT" \
  --output-dir "$DEV_RUN"
```

After the two blinded AI reviews select the development comparator, the final command also requires its score record and development run receipt:

```sh
PYTHONPATH=src python scripts/adaptive_benchmark.py \
  --source-root "$FLORA_SOURCE" --manifest "$FINAL_MANIFEST" \
  --split final --gate-evidence "$GATE_C_RECEIPT" \
  --development-selection "$DEV_SELECTION" \
  --development-run "$DEV_RUN/run-summary.json" \
  --output-dir "$FINAL_RUN"
```

Before allocating final GPUs, the coordinator must check the same projection and the project ledger. The projection uses the largest measured development interval time for each policy, every final interval, scaled source preparation, shared setup/load/cleanup, and a factor of 1.5. A stricter adaptive allowance uses the larger of the observed adaptive maximum and compact maximum plus native maximum plus gate overhead. Thus zero fallback during development does not imply zero fallback work during final evaluation. This remains a planning estimate, not a guaranteed runtime bound. The runner rejects a projection above the frozen 90-minute ceiling. It does not reduce the target set. The coordinator must set SLURM wall limits of at most 30 minutes for development and 90 minutes for final evaluation. The runner also stops before starting another interval after that phase deadline. The scheduler limit bounds an interval already in progress. The coordinator must preserve the final reproduction reserve and the two-submission/eight-GPU-hour experiment cap.

Each interval-policy result is saved atomically under `outcomes/` with a content checksum and execution-contract hash. `run-summary.json` lists these files. Private `invocation-*.json` receipts preserve each invocation, including failures. Explicit `--resume` accepts only the same code, freeze, model configuration, and Gate C evidence. Final contracts and outcomes also bind the selected development comparator. The scorer checks that same comparator. Actual Captioner device placement must match Gate C and development. Resume reuses completed outcomes without changing their measured times. A cache-only invocation is not fresh inference. A rejected invocation preserves the earlier published outcome index and summary. A recorded runtime failure requires diagnosis; it does not trigger more GPU work on resume. Active work that fails before a confirmed durable checkpoint remains in its invocation receipt. Missing or uncommitted work blocks the final measured-cost gate.

`outcomes.jsonl` is an atomic aggregate of the available verified checkpoints, including partial work. Pass it or the individual outcome JSON files to the review tool. CPU runner and independent regression tests cover the evaluator export boundary, frozen order, checkpoint identity, complete cost projection, changed placements, comparator binding, preserved evidence, and scheduler restart rejection. These tests do not establish GPU execution or adaptive quality.
