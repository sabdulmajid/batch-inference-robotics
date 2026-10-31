# Conservative Captioner microbatches

The optional batching utility combines compatible, already prepared Captioner requests. It retains every native input tensor and routes each generated caption back to its episode and event. For example, two requests with equal token lengths and equal video grids can share one model generation call.

**Batching is not enabled in the engine or CLI.** The current native path remains batch size one. CPU tests passed, but no real Captioner batch has passed GPU memory, output-agreement, or performance checks. No measured admission record is included.

## Input contract

The utility accepts only the actual tensor keys emitted by the pinned Transformers 5.8.1 processor:

| Key | Single-request layout | Combination |
| --- | --- | --- |
| `input_ids` | `[1,N]`, int64 | Add rows in request order |
| `attention_mask` | `[1,N]`, int64, all ones | Add rows in request order |
| `mm_token_type_ids` | `[1,N]`, int64 | Add rows in request order |
| `pixel_values_videos` | `[total patches,1536]`, FP32 | Append flattened patch rows |
| `video_grid_thw` | `[temporal blocks,3]`, int64 | Append block rows |
| `video_metadata` | Native processor metadata | Keep with its request; do not pass to generation |

Requests must have the same profile, execution contract, token length, tensor shapes, and video grid values. The supported caption output budget is 512 tokens. The utility rejects padding, additional keys such as precomputed positions or caches, mixed profiles, duplicate request routes, and a context length above the native limit. It does not add padding, select frames, resize images, or alter prompts.

Qwen3.5 reads flattened video features in prompt-row order. It uses `mm_token_type_ids` to compute multimodal positions. The implementation therefore keeps all five model tensor keys and records the patch/grid slices for each row. See the [pinned model implementation](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

The input audit uses the same dtype, shape, and raw tensor-byte hashing format as the native generation helper. A second integrity check detects changed request tensors, changed combined tensors, and changed output routes before GPU work.

`collate_caption_requests` requires explicit `max_batch_size` and `max_prepared_bytes` limits. The byte limit covers both retained source tensors and their combined copy. The result reports this retained payload and the smaller combined-copy payload separately. The limit is not a total host-RAM ceiling; allow additional space for metadata, processor objects, and temporary hashing buffers. CPU collation does not establish GPU feasibility.

## Measured admission

`generate_caption_batch` requires a `MeasuredBatchAdmission`. A missing or unverified record fails before generation. A valid record must identify a completed real pilot and independently checked native batch-one agreement. It must match the exact batch size, compatibility signature, framework/runtime versions, logical GPU capacities, model placement, and a verified same-node topology receipt. A native-session binding also pins the source/model/processor revisions and verified parameter receipt. The runtime fingerprint includes the actual tokenizer, chat template, processor settings, model configuration, parameter layout, generation configuration, and batching code. A changed model or processor object requires a new verified binding. The prepared video-token ID must match the processor.

Every visible device requires measured baseline allocated/reserved bytes and peak allocated/reserved bytes. The current live allocation must not exceed the pilot baseline. Only reserved memory that has no live tensors counts as reusable cache. The record also specifies positive safety headroom. The runner checks both total device capacity and current free memory. It does not infer capacity from the number of GPUs or scale a batch-one memory value into a larger admission.

An admission record is a checked configuration supplied by the coordinator. It is not proof by itself. Retain the measured pilot artifacts and review them. Initial calibration must use a separate, coordinator-authorized, bounded GPU experiment and retain its native comparison and memory artifacts. This runner cannot create its own first admission. The production guard requires evidence that the complete 512-output-token memory envelope was measured. A short completed caption on the same input shape does not establish that envelope. The maximum observed generated-token count is recorded. Calibration and native-output agreement are separate checks; changed generation used for an explicitly named memory stress probe cannot count as native-output agreement. CPU checks and synthetic test records cannot authorize production batching. The bounded script and concrete experiment request are in [BATCH_CALIBRATION.md](BATCH_CALIBRATION.md).

The runner has no batch-size growth, automatic retry, quantization, CPU/disk weight offload, or visibility override. It uses the verified native resident model. Inference remains serial at the model-call level; concurrent generation on the same model is not supported.

## Generation and completion

Generation retains native greedy decoding, one beam, a 512-token output limit, cache use, the native tokenizer IDs, and `max_time=300`. The time limit applies to the shared generation call. If it stops before a row reaches EOS, that row fails explicitly. See the [pinned generation stopping criteria](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/generation/stopping_criteria.py).

Each output row must preserve its input prefix. Extraction stops at that row's first EOS token. Any later tokens must be padding. A missing EOS, empty caption, or unexpected tokens after EOS produces a failure status. One row's early completion does not establish completion for another row.

The result retains request, episode, and event IDs, input digests, generated-token counts, row statuses, elapsed generation time, collation time, and peak memory for every visible GPU. A generation exception raises `BatchExecutionError` with an attached failure record. Admission and integrity exceptions also retain an attached preflight record, including elapsed work. Results separate collation, admission/integrity, generation, and total runner time. Native decoding/preparation still belongs to the caller’s full workflow measurement. Exceeding a recorded pilot peak is reported separately from exceeding physical device capacity and makes the overall batch result incomplete. Save that record before retry decisions. Allocation accounting remains external and includes failed and idle time.

## CPU verification

Eight tests passed in the isolated processor CPU environment described in [ADAPTIVE.md](ADAPTIVE.md). The integration test used the actual pinned Captioner processor and tokenizer. It prepared two requests with different RGB pixels. Each request had a four-frame BEFORE block and an eight-frame TARGET block.

The collated tensors exactly matched all five tensors from one actual joint processor call. This checks visual row order, temporal block order, modality IDs, and token rows. Other tests checked stable output identity, independent EOS handling, padding, incomplete outputs, mutation detection, explicit memory bounds, and admission rejection. The generation-output tests use synthetic tensors and token decoding. They do not exercise a model.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_batching.py' -v
```

Tests skip optional torch/processor cases when their dependencies or pinned local processor assets are absent. They download nothing.

## Opt-in native collector

`caption_with_microbatch` in `bir.caption_batching` is an explicit alternative to `NativeBackend.caption`. The normal engine and CLI do not call it. The caller must supply at least one measured admission, a pinned execution-contract identity, a topology fingerprint, and byte/request limits. No admission is provided by the project.

This collector pilot accepts only prepared human single-view episodes. It rejects other profiles before model work. Native robot single-view planners can use custom or six-digit event IDs; the current collector deliberately does not interpret those conventions. The regular batch-one native path remains available for those profiles.

The wrapper holds the native execution lock and temporarily intercepts only the Captioner generation helper. The unchanged upstream planner, frame preparation, prompt, and processor still produce each request. The collector retains exact CPU tensors and non-JSON request handles. Half its byte limit bounds retained source tensors; the other half reserves space for the combined tensor copy. Native frame/image preparation, metadata, and hash temporaries are additional memory. Inputs are hashed when collected and checked again before execution.

Only exact groups with an explicit measured admission use batched generation. Requests with no matching admission and final incomplete groups use the original native batch-one helper. Reaching a collection bound flushes pending work. Admission or batch failures stop the stage; they are not automatically retried as singles. TempSeg interception is rejected because its algorithm needs each response immediately.

After upstream caption planning returns, the wrapper flushes pending work and binds each handle to the actual native event ID in its caption-input audit. It verifies order and cardinality, preserves unsupported intervals, and replaces every handle with its completed string. An unresolved or misrouted handle fails the stage. The temporary generation hook is always restored. Collection ends at the episode boundary; cross-episode batching is not implemented.

The result records batch and single-request work, original input digests, native event routes, bounds, and the complete stage time including the final flush. It also captures GPU peaks before each batch utility resets its peak counters. A caller must use those snapshots and batch records to report the whole stage peak. Any failure has an attached `collector_record`; preserve it with the failed attempt.

Seven CPU collector tests passed. They check exact grouping, single-request fallback, bounds, mutation rejection, deferred output routing with an unsupported interval, early unsupported-profile rejection, TempSeg rejection, and hook restoration. They use a stub generator and synthetic test admissions. They establish software routing only. GPU admission still requires measured memory and native batch-one comparisons on the same placement and inputs. Equal tensor inputs do not guarantee identical generated text under a different numerical batching order.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_caption_batching.py' -v
```
