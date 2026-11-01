# Project status

Updated: 2026-11-01 UTC

## Active next phase

The user has requested a more substantial performance contribution and a
researcher-facing annotation/review workflow. Follow [the phase-two brief](PHASE2_PROMPT.md).
Use the delivered native release as the baseline, profile the actual bottleneck,
implement the selected acceleration, and compare on a broader workload.

Phase-two implementation is active. Source-only selection is frozen in
[`manifests/phase2-native.json`](../manifests/phase2-native.json): eleven new
complete episodes, 726.897933 source seconds, in two counterbalanced blocks.
All eleven original videos passed scheduled CPU preparation with unchanged
bytes, durations, and geometry. This used 132 seconds on four CPU cores.

The selected optimization removes lossless PNG serialization/readback from
human single-view caption preparation and prepares one next request during
generation. A real two-request CPU check measured 17.688 s for the original
path and 7.535 s for direct preparation, with exact processor tensors and
frame/clock/prompt agreement. This is not an end-to-end GPU speedup result.
The GPU compatibility pilot passed: all 45 captions and 65 ordered native model
inputs match the accepted optimized baseline. The accelerated three-episode
subprocess took 836.677 seconds with one load per model and batch size one.
This is not a matched speedup comparison. The profiled baseline recorded about
188 seconds in PNG saving on one development episode, supporting the selected
transport optimization. See [the pilot evidence](../reports/phase2-native-pilot.json).
The scheduled CPU check passed all 45 development caption requests using the exact
native processor, including concurrent tokenizer access. All tensor digests also
match the prior actual GPU Captioner inputs in order. The check loaded no weights
and performed no inference. It used 1,008 allocated seconds on four CPU cores.
See [the preparation evidence](../reports/phase2-preparation-check.json).
The reviewed candidate is `87fff61`; CI passed.

A bounded metadata spill cache repairs the eight-entry reuse limit. Runtime
dependency versions now enter the execution identity. The lightweight
annotation/review/export workflow passed local browser playback, correction,
HTTP, and export checks on real precomputed output. An independent review and
actual CLI tests cover isolated CPU resume verification; the fresh GPU workflow
reproduction remains pending. Existing phase-one
evidence remains unchanged. Follow [the finite protocol](PHASE2_PROTOCOL.md);
the budget does not reset. One pending profile submission was cancelled before
allocation after a manifest relocation error was found. It used zero GPU time
but consumed one submission. Profiling and the successful pilot shared one
corrected allocation.
The [frozen execution plan](../manifests/phase2-execution.json) admits both
unchanged blocks at 116 minutes each on three GPUs. Their requested 11.6 GPU-hours
plus 1.5 protected reproduction hours fit the 13.748611 hours available at
admission. The three submission slots then available were assigned. The selection and success target
remain unchanged.

The first broader block failed in the previous optimized baseline. Captioner
ran out of GPU memory on episodes 59 and 60. The other three episodes completed
with 65 serialized captions. All five segmentation stages passed. The accelerated
condition did not start. Identical input shapes succeeded elsewhere, so this
does not establish a universal input-size limit or prove allocator fragmentation.
See [the preserved failure evidence](../reports/phase2-block0-failure.json).

All eleven episodes remain counted. The complete eleven-episode paired success
criterion did not pass. The unchanged second block completed all six preselected
episodes in both versions. The new path took 2,246.460 subprocess seconds versus
3,534.320 for the previous optimized path: **1.573284× throughput and 36.438699%
less GPU capacity-time** on the same three RTX 3090 GPUs. All 121 captions and
170 ordered native model inputs match. Independent review confirmed source,
frame, prompt, parameter, placement, runtime, identity, and output agreement.

This is one candidate-first block. It does not replace the failed block or
establish the full-corpus target. The [combined comparison report](../reports/phase2-comparison.json)
retains all eleven episodes, both allocations, the incomplete status, and no
pooled speedup. The two comparison allocations used 7.612500 GPU-hours,
including the failed baseline work.

Current aggregate accounting: 17.863889 GPU-hours used and 1.5 reserved for the
protected final workflow reproduction. All sixteen GPU submission slots are
now used. CPU-only use is 4.274444 core-hours. The final reproduction is running
with the immutable accelerated code. It must verify fresh annotation, isolated
CPU resume, status, and export before delivery. No further GPU submissions are
permitted under the current cap.

## Delivered baseline

The native batch-one research release is implemented and independently verified.
It follows planning commit `459f49d`. See the [final report](FINAL_REPORT.md) for
the delivered scope, measured results, reproduction commands, and limits.
The installed WATcloud skill and account reference were read. Current SSH,
account association, scheduler, and storage access were checked. No old expiry
notice was treated as a current failure.

The user approved changing the project storage headroom check to available
space >=120% of remaining planned new writes. The 250 GiB project cap is
unchanged. This replaces the original whole-filesystem 20%-free rule; it is a
project policy choice, not a cluster quota change.

- Native stage adapter, placement guard, decoded-frame cache, resumable store,
  staged engine, resource ledger, and evaluation functions are implemented.
- Local tests exercise actual synthetic video decoding and native frame planning.
- CPU preparation completed. Pinned model files total 129,079,719,871 bytes.
  The preparation allocation used 2.752222 CPU core-hours. All required assets
  and the isolated Linux environment are present.
- Framework GPU test passed on one RTX 3090 using PyTorch 2.11.0+cu128.
  The numerical BF16 result was correct. The allocation completed in 41 seconds.
- Native TempSeg passed on three development episodes after the diagnostic
  serialization fix. The job completed in 395 seconds, used one RTX 3090,
  and retained one model across 45 generated segments. All checkpoint
  signatures passed verification. Peak GPU allocation was 18.353 GiB.
- The earlier failed allocation is retained and counted. See
  [the successful measurement](../reports/native-tempseg.json) and
  [the earlier failure](../reports/tempseg-initial-failure.json).
- Gate C passed. Captioner completed all three initial development episodes and
  serialized 45 nonempty captions on three same-node RTX 3090 GPUs. The 1,276
  second allocation completed with exit code zero. One model instance served
  all three episodes. Native parameter checks and GPU residency passed.
  See [the end-to-end evidence](../reports/native-end-to-end.json).
- The adaptive protocol and task-separated split are frozen. All three exact
  development/final videos passed preparation with original bytes and clocks.
- Native development AI reference review is complete. Temporal micro F1 is
  0.6118; joint temporal/AI reference-event micro F1 is 0.0941. This three-episode
  development result does not measure held-out or visual accuracy.
  See [the reference review](../reports/native-reference-review.json).
- The CPU-only resume check passed on an isolated state backup. It resumed all
  six stages with zero fresh attempts and no model-load or generation events.
  Request identities, stage artifact hashes, and original evidence were unchanged.
- CPU caption-pair preparation and ordered replay passed. This does not admit
  GPU batching. See [the CPU preparation report](../reports/caption-batch-cpu-preparation.json).
- Controlled native performance is measured. Adaptive development is complete;
  held-out adaptive quality remains unvalidated.
- The first native benchmark was stopped after the reference path failed a later
  Captioner placement check. Its successful and failed work is preserved.
  A bounded two-episode diagnosis reproduced the rejection and recorded a typed
  fatal-load failure. Local tests verify that later fresh stages are blocked.
  The model object was released, but unused allocator reserve remained on two
  GPUs. The capacity check omitted that owned cache.
  See [the diagnostic evidence](../reports/native-lifecycle-diagnosis.json).
- The corrected capacity check passed a fresh two-episode native reload test.
  Both TempSeg and Captioner loaded twice. All 30 captions, segment boundaries,
  and native input audits exactly matched the prior Gate C outputs. Actual model
  placement was unchanged, and the check retained 4 GiB headroom per GPU.
  The allocation completed in 916 seconds and used 0.763333 GPU-hours.
  This validates repeated loading for this configuration. It does not establish
  benchmark speedup or adaptive quality.
  See [the reload verification](../reports/native-reload-verification.json).
- The full native benchmark passed independent review on three same-node RTX 3090
  GPUs. Optimized execution took 1,328.397 subprocess seconds versus 1,487.463
  for reference: 10.694% less wall time in one ordered trial. All 45 captions and
  native inputs matched. All three cache-only resume checks passed. The optimized
  retained-frame cache stayed within 256 MiB. See
  [the measurements and limits](NATIVE_BENCHMARK.md).
- Adaptive development completed all 21 items. Compact used 7.942% less
  interval-policy time than native and produced the same seven captions.
  Two blinded AI reviews gave every policy 3/7 reference-event agreement.
  The frozen rule selected compact as the fixed comparator.
- Final adaptive evaluation was not admitted. The unchanged conservative
  projection is 117.693 minutes, above the frozen 90-minute limit. No final
  inference ran. Targets and rules remain unchanged; quality is inconclusive.
  See [the adaptive results and limits](ADAPTIVE_RESULTS.md).
- The bounded batch calibration completed three generation calls. Both batch
  captions matched native singles and Gate C. Peak reserved memory left only
  0.37–0.46 GiB per GPU, below the predeclared 2 GiB reserve. Stress and collector
  tests were skipped; no batch admission was written. Native batch size one
  remains the default. See [the calibration result](BATCH_CALIBRATION.md#gpu-calibration-result).
- The protected final reproduction passed. All three original episodes completed
  in fresh state, with 45 captions and all 65 model inputs matching the optimized
  benchmark and Gate C. The isolated CPU resume reused six stages with zero
  new attempts, preparation, model loads, or generation. Original evidence
  remained unchanged. Independent engine and native reviews passed.
  See [the reproduction evidence](../reports/final-reproduction.json).

## Delivered phase-one resource closure (historical)

- GPU-hours used/reserved: 8.773056 / 0 of 24, including failed allocations.
  The stopped benchmark used 0.639167 GPU-hours. The diagnosis used another
  0.554167 GPU-hours, and the successful reload used 0.763333 GPU-hours.
  The successful full benchmark used another 3.5 GPU-hours.
  Adaptive development used 0.715833 GPU-hours.
  Batch calibration used 0.138333 GPU-hours.
  Final reproduction, including its CPU-only resume, used 1.131667 GPU-hours.
  All completed allocation outcomes are reconciled.
- GPU submissions and reservations: 11 of 16.
- CPU-only core-hours used/reserved: 3.007778 / 0 of 32.
- All 15 parent allocations are reconciled: 12 completed, two failed, and one
  was cancelled during diagnosis. No project jobs or reservations remain.
- The final reproduction fulfilled its protected allowance. Unused reserved
  time is released and is not counted as allocated use.
- Combined measured remote and local project storage is about 130.13 GiB,
  below the 250 GiB cap. See [the resource closure report](../reports/resources.json)
  for exact totals, measurement times, and CPU accounting boundaries.
- A durable coordinator marker and private operational ledger are established.

## Next actions

Execute the phase-two brief through implementation, a usable example, broader
fresh-run comparison, and reproduction. Plan within remaining project resources;
no new GPU job was submitted by this documentation update. The original adaptive
experiment and its inconclusive result remain part of the baseline history.

Private records retain exact job IDs, connection settings, output paths, skill
hashes, and scheduler output. Public reports contain sanitized measurements.
