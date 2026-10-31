# Verified native inference release

The engine can annotate a manifest of short, single-view human videos with
native FLORA TempSeg and Captioner. It keeps each model loaded across episodes,
reuses a bounded frame cache, overlaps CPU preparation, and saves restartable
stage results. A researcher can resume completed work without repeating model
inference.

The native batch-one path was reproduced from an immutable release on real
GPUs. The bounded adaptive and batch experiments also completed their permitted
work. Their limits remain explicit: held-out adaptive evaluation was not
admitted, and production microbatching remains disabled.

## Delivered software

The CLI and implementation are in [`src/bir`](../src/bir). The engine provides
manifest validation, stable source/request identities, staged model lifetime,
a 256 MiB decoded-frame cache, bounded preparation overlap, atomic results,
explicit failures, and cache-only resume. Tests cover source clocks, placement,
batch routing, recovery, and resource accounting.

The compatible-request batching utility and collector are implemented. A real
two-request calibration established native agreement for its measured pair.
It did not establish the required memory reserve, so the engine and CLI retain
batch size one. The separate spatial-budget API and frozen evaluator are in
[`scripts`](../scripts) and [the adaptive module documentation](ADAPTIVE.md).

Weights, videos, environments, raw captions, private connection settings, and
operational logs remain outside public Git. Sanitized evidence is in
[`reports`](../reports); public input identities are in [`manifests`](../manifests).

## Configurations actually verified

| Component | Verified configuration |
| --- | --- |
| Runtime | Linux x86_64, Python 3.12.12, PyTorch 2.11.0+cu128, Transformers 5.8.1 |
| Native inference code | Immutable commit `d2701b08a08b88d18887abed81f71c8373478d7b` |
| Resume verification helper | Commit `aad9dcfa7ee0ec398e007ad3766a2441d6337602`; inference modules unchanged |
| Inputs | Original `homer_52`, `homer_50`, `homer_41`; human profile, one camera, empty instruction; 138.767089 unique source seconds |
| Placement | TempSeg initially verified on one RTX 3090; full staged runs use one allocation of three same-node RTX 3090 GPUs for Captioner |
| Job resources | Four CPUs, 128 GiB requested host RAM, 20 GiB temporary disk for full native runs |
| Native settings | Original model values, mixed BF16/FP32 rules, unmerged adapter, preprocessing, frame selection, prompts, greedy generation, parsing and stitching |
| Engine | Staged reference, persistent, and optimized conditions; batch size one |

The placement-only adapter keeps inference parameters and nonempty buffers on
CUDA. No quantization or CPU/disk inference-weight offload was used. Actual
model loading needs about 50.956 GiB of parameter storage across the GPUs.
The tested setup does not imply that one or two 24 GiB GPUs can hold Captioner.

## Fidelity, performance, and recovery

The controlled benchmark ran all three conditions on the same three GPUs and
the same three videos:

| Condition | Subprocess wall time | Model loads |
| --- | ---: | --- |
| Reference | 1,487.463 s | Three TempSeg and three Captioner loads |
| Persistent | 1,371.814 s | One load per model |
| Optimized | 1,328.397 s | One load per model |

Optimized used **10.694% less wall time**, or 1.120× reference throughput, in this
one ordered trial. All 45 captions and 65 model-input records matched. The
optimized cache retained at most 268,429,152 bytes against its 268,435,456-byte
limit. All three conditions passed cache-only resume.

This is a whole-workflow comparison. Preparation and metadata reuse differ
between conditions, so it is not a pure model-lifetime experiment. Filesystem
caches, fixed execution order, and the small corpus limit generalization.
See [the benchmark boundaries and per-example evidence](NATIVE_BENCHMARK.md).

The final fresh reproduction completed all six stages and exported 45 captions.
It again matched the optimized benchmark and Gate C, including all 65 input
records, native frame/clock/prompt audits, model checks, and placement. It loaded
each model once. The engine timer was 1,331.233 s; the CLI timer was 1,331.457 s;
the probe timer was 1,354.710 s. These nested boundaries are not additive and
are not another controlled speedup measurement.

The same allocation then verified an isolated CPU-only resume. It reused six
stages with zero fresh attempts, preparation, model loads, or generation. The
original evidence stayed unchanged. The entire allocation completed in
1,358 seconds with exit code zero and used 1.131667 GPU-hours. Independent
engine and native reviews passed. See [the reproduction evidence](../reports/final-reproduction.json).

## Annotation quality and adaptive result

Native development review used independent text references and two blinded AI
reviewers. Temporal micro F1 was 0.6118. Joint temporal/AI reference-event micro
F1 was 0.0941. These scores do not establish visual accuracy, human agreement,
or held-out performance. See [the native evaluation method and limits](NATIVE_EVALUATION.md).

Adaptive development completed 21 items: seven independent intervals under
native, compact, and format-gated adaptive policies. All policies produced the
same seven captions and scored 3/7 under two separate blinded AI reviews.
Compact used 7.942% less full interval-policy time than native, with source
frames and clocks retained. The frozen rule selected compact as the fixed
comparator. All adaptive outputs passed the format gate, including captions
with zero reference-agreement scores.

The unchanged final admission check estimated 117.693 minutes for all 81 final
items, above its frozen 90-minute limit. The estimate includes potential native
fallback and 1.5 headroom. No final job or held-out prediction was generated.
The targets and rules were not changed. **The adaptive quality conclusion is
inconclusive.** This fixed-interval experiment omits TempSeg and does not
establish full-pipeline adaptive speedup. See [the adaptive report](ADAPTIVE_RESULTS.md).

## Batching result

Two compatible 4,344-token requests produced identical captions in separate
native calls and one native batch. The rows completed after 14 and 16 generated
tokens. Peak reserved memory left only 0.37–0.46 GiB on each GPU, below the
predeclared 2 GiB reserve. There was no out-of-memory failure. The script skipped
the larger 512-token stress test and collector demonstration and wrote no
admission. No optional sweep followed. See [the calibration evidence](BATCH_CALIBRATION.md#gpu-calibration-result).

## Resource closure and reproduction

All 15 scheduler parent allocations are reconciled: 11 GPU and four CPU-only
allocations. Twelve completed, two failed, and one was cancelled during
diagnosis. Total use is **8.773056 of 24 GPU-hours** and **3.007778 of 32 CPU-only
core-hours**. The GPU total includes failed, cancelled, idle, and resume time.
CPU cores allocated with GPUs used another 12.387222 core-hours, reported
separately. Eleven of 16 GPU submission slots were used. No project allocation
or reservation remains. No paid API or paid external compute was added.

Combined measured project storage is about 130.13 GiB, below the 250 GiB cap.
The [resource report](../reports/resources.json) gives exact measurements,
observation times, allocation outcomes, and accounting limits.

Use the [verified reproduction commands](REPRODUCTION.md#5-verified-release-reproduction)
with the pinned environment and prepared inputs. The short user command is:

```sh
bir run --manifest jobs.jsonl --source "$FLORA_SOURCE" \
  --run-dir runs/native --condition optimized
```

Repeat the same command to resume. Use a new run directory for fresh inference.
The final exports are `annotations/*.caption.json`; signed segmentation and
stage history are retained in `run.sqlite`. Keep these raw artifacts private.

## Remaining limits

- Three short human single-view videos are verified. Multi-camera profiles,
  longer inputs, concurrent model residency, and larger contexts need separate
  measurements. The configured native output limit was retained; maximum-length
  input/output combinations were not stress validated.
- The default engine remains batch one. Full-length batch-two memory safety
  and production batching are unvalidated.
- Adaptive held-out quality and the one-percentage-point population margin
  are unvalidated. Native agreement alone does not establish useful labels.
- Preparation, generation, and audit timers have different boundaries. The
  measured speedup is not a guarantee for a new corpus or machine.
- FLORA source and weights retain NVIDIA's non-commercial research/evaluation
  license. This release does not train or control a robot.
