# Native execution contract

The adapter runs the released NVIDIA FLORA TempSeg and Captioner code. It keeps one model resident while the engine processes that stage. This reduces repeated model loads without changing the source video or the native frame plan.

The implemented path is **placement-adapted native, batch size one**. CUDA model loading and inference remain unvalidated. CPU media validation, frame planning, and decoded-frame cache equality have passed a local integration test. These results do not establish annotation quality or GPU feasibility.

## Source and model identity

The adapter requires an unchanged tracked checkout of NVIDIA/Flora at `129a2ada74c526ee4692f3d1b111bea6ca19d5b7`. It also checks the separately pinned model revisions:

| Asset | Revision |
| --- | --- |
| TempSeg adapter | `114a1076f70eb53f4a58ac045505db3087e79347` |
| Qwen3.5-9B base | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Captioner | `fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6` |

Native mode rejects environment variables that replace the released repositories. The unchanged upstream loader checks the model source manifest. The unchanged upstream bootstrap checks the vendor manifest. Model files, caches, and upstream checkout stay outside public commits.

FLORA source and model weights use the [NVIDIA research license](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/LICENSE). Use is limited to non-commercial research and evaluation. The project imports the upstream interfaces; it does not relicense the upstream code or weights.

## Placement and memory

The native TempSeg loader remains unchanged. It loads the BF16 base, retains the unmerged FP32 adapter, and checks adapter and Gated DeltaNet values before and after placement.

For Captioner, a short, locked adapter changes only the `max_memory` argument to the native `from_pretrained` call. The model identity, mixed dtype policy, attention implementation, checkpoint verification, and generation settings remain unchanged. The adapter uses only logical CUDA devices already visible to the process. It never broadens `CUDA_VISIBLE_DEVICES`.

Each device budget is its measured free memory minus explicit activation headroom. The default is 4 GiB per visible device. This is an initial placement setting, not a measured safe configuration. A later GPU probe must measure the actual per-device peaks for the native input size.

The released Captioner checkpoint uses FP32 storage. Storage bytes therefore exceed the native BF16 base-weight allocation. The adapter reads safetensors headers and estimates runtime weight bytes as BF16 base parameters plus the native FP32 Gated DeltaNet parameters. It preserves integer buffer byte counts. It does not read tensor payloads for this estimate. Header checks reject missing tensors, invalid shapes, truncated shards, or disagreement with the index. The estimate excludes activations, generation caches, and allocator overhead.

The native loader still verifies saved Gated DeltaNet values. After loading, the adapter rejects any parameter or nonempty buffer on CPU, disk, or meta devices. It also rejects quantized models and parameter dtypes outside BF16 and FP32. These checks are strict. A model that fails them is a failed native configuration, even if it could emit text with offload.

## Stage and output identity

`NativeBackend.prepare(job)` calls the upstream CPU input-validation path. It checks all video timestamps, camera synchronization, duration, size, profile, and instruction constraints. It loads no model.

`segment(prepared)` uses the upstream temporal planner, predictor, boundary repair, and stitcher. It returns a JSON checkpoint with source hashes, profile, instruction hash, model revisions, native audit records, and a digest of the stage payload.

`caption(prepared, segmentation)` checks that digest and input binding before using the upstream caption planner and generation helper. A changed source, configuration, or checkpoint is rejected. The engine must store the segmentation payload unchanged. Routing identifiers belong to the store or final output envelope.

The adapter retains the upstream caption output schema and audit records. Unsupported intervals retain their explicit status. They do not become complete captions. Native generation still requires a stop token and nonempty output.

Transformers 5.8.1 returns sets in its model-loading diagnostics. The adapter copies these diagnostics to JSON before inference. Sets become arrays sorted by canonical JSON; tuples become arrays. No diagnostic entries are dropped. The upstream objects remain unchanged. Unknown object types and nonfinite numbers fail at model load. The resulting JSON diagnostics are part of the signed segmentation checkpoint and the final Captioner record.

`PersistentSession` reuses the loaded model when the requested stage is unchanged. A stage change releases the former model through the upstream session. A failed load clears the cached model. The engine runs segmentation across the corpus before captioning across the corpus.

The adapter emits `BIR_RUNTIME model_load_end` after the full load, processor construction, native audit checks, residency checks, and synchronization of every visible CUDA device. It records `kind`, `elapsed_s`, `cold_load`, and success/failure status. The same measurements are retained under `engine.model_load_timings`. Cached model access is not a cold load. Upstream events remain unchanged; their earlier parameter-verification time is not the full load time.

A non-transient failure during loading or verification is retained for that stage within the session. Later work items receive an explicit blocked-load error before another model load. A new session requires a diagnosed configuration. Known transient transport/I/O errors retain the existing bounded retry policy. An error during generation is input-specific and does not set the model-load failure state.

## CPU preparation and frame reuse

Video preparation can run in a CPU thread while the current inference call uses the GPU. Preparation does not acquire the global inference lock. Inference calls are serialized because the upstream modules contain process-wide state.

The optional decoded-frame cache wraps only upstream `decode_selected`. Cache keys include source content identity, frame index, and canonical color-conversion policy. Cached images are copied so downstream processing cannot change stored frames. Native resizing, JPEG transport, timestamp selection, camera layout, prompts, processor calls, and stitching still run upstream.

No vision-embedding or attention-cache reuse is implemented. Native request microbatching above one is disabled until real GPU memory, routing, padding, stopping, and input-equivalence tests pass. The project does not claim a batch speedup.

## Verification and remaining gates

Run the local native tests after installing the optional CPU dependencies and fetching the pinned source:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_native.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_placement.py' -v
```

The CPU integration test encodes a small synthetic video. It passes that file through the actual native input validator, selected-frame decoder, human caption planner, and decoded-frame cache. It compares decoded bytes and checks source frame bounds. This test runs without torch or model weights.

Separate GPU gates remain required: a framework operation; native TempSeg; faithful Captioner loading and inference; complete annotation; reference-versus-persistent fidelity; per-device peak memory; and performance measurements. Multi-camera, long-window, variable-rate, maximum-context, and batched native inference are not validated by the local CPU test.

## Allocation probes

`scripts/framework_smoke.py --output smoke.json` performs a small BF16 matrix multiplication on each visible CUDA device. It checks the exact result and records the framework, runtime, device properties, and peak memory. It requires the pinned PyTorch 2.11.0/cu128 environment and a SLURM allocation.

`scripts/native_probe.py` runs the same contract and state store as `bir run`. It accepts at most three manifest episodes. Use the same `--state-dir` and `--condition` for a segmentation allocation followed by a Captioner allocation. Use a distinct `--output-dir` for each allocation. The probe records private topology, package versions, source validation, final status, and peak memory for every visible GPU.

A successful resume is labeled as a cache retrieval when no new stage executes. An empty annotation cannot pass the native-caption gate. That gate requires a fresh caption stage, nonempty complete captions, and saved Captioner residency verification. Separate-stage runs preserve the TempSeg loader checks inside the segmentation checkpoint.

Both probes reject nonzero `SLURM_RESTART_COUNT` before model work. The coordinator must reconcile a restarted allocation before another attempt. The probes do not submit, cancel, or retry scheduler jobs.

## Asset trust anchors

`source_manifest_identity` in the upstream Captioner configuration is a source
model identity in `MODEL_PROVENANCE.json`. It is not the raw SHA256 of
`SOURCE_MANIFEST.sha256`. `scripts/check_asset_provenance.py` independently pins
both metadata files and checks that identity. Run it before Captioner loading.
The unchanged native loader then verifies every payload against that anchored
manifest. The metadata-only utility does not claim to read or verify weights.

```bash
python scripts/check_asset_provenance.py "$FLORA_MODEL_CACHE/captioner"
```
