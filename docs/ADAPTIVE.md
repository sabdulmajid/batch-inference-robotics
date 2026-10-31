# Caption spatial-budget experiment

This module tests a smaller spatial budget for Captioner while retaining every selected source frame and its timestamp. For example, a 40-frame target can use fewer visual tokens while a short context block stays unchanged. This is an experimental path. It does not replace the native reference.

The experiment is not integrated into the native CLI. Complete the native end-to-end gate before integration or GPU evaluation. No quality or GPU performance result exists yet.

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

Seven tests passed in an isolated CPU environment. The processor test used the actual Transformers 5.8.1 classes and tokenizer/processor files from Captioner revision `fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6`. No model weights were downloaded for this test.

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

## Integration after the native gate

The intended hook is the processor returned by the persistent Captioner session. Wrap it with `CaptionSpatialProcessor` only for a named compact experiment. Leave the TempSeg processor and native reference untouched. The proxy delegates `apply_chat_template`, tokenizer access, and other attributes to the same processor. The existing caption planner then supplies the same blocks and clocks.

Use a new run contract and state directory for each experiment. Record the preset, gate rule, source/model revisions, and fixed intervals. Freeze the development/test manifests, evaluator, margin, and routing rule before the final run. Update the evaluation configuration to describe the total-block budget; its earlier uniform `compact_area_fraction` proposal is not the implemented setting.

## Format-only fallback

`run_compact_then_native` runs compact generation first. The initial gate accepts a complete, nonempty caption string. An incomplete generation or invalid empty output triggers native fallback. Other failures, including out-of-memory failures, stop the work item. They do not trigger an automatic more expensive rerun.

The caller must map the native generation helper's incomplete-output exception to the declared `invalid_output_errors` tuple. A valid string is only a format check. The gate does not assess whether the caption is true. It never receives the native output before its decision.

Every attempted compact pass, gate decision, and native fallback has a measured application duration and an explicit status. Failed work remains in the record. For CUDA use, supply a synchronization callback for every visible device so timed work finishes at the measurement boundary. Scheduler allocation GPU-hours remain a separate measurement and include idle time. Do not add a job total again after allocating it across passes.

Independent references are still required for quality evaluation. Audit accepted compact outputs as well as fallback cases. Charge that audit work separately. If captions remain structurally valid, this baseline gate can have zero fallback even when captions are wrong. A negative or inconclusive quality result is valid.
