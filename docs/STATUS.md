# Project status

Updated: 2026-10-31

## Current state

Implementation is in progress from planning commit `459f49d`.
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
- Controlled native performance and adaptive quality results are still pending.
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
- The second full native benchmark is queued for shared cluster capacity.
  No complete performance comparison is available yet.

## Resource state

- GPU-hours used/reserved: 3.287222 / 6 of 24, including failed allocations.
  The stopped benchmark used 0.639167 GPU-hours. The diagnosis used another
  0.554167 GPU-hours, and the successful reload used 0.763333 GPU-hours.
  All completed allocation outcomes are reconciled.
- GPU submissions and reservations: 8 of 16.
- CPU-only core-hours used/reserved: 3.007778 / 0 of 32.
- Final reproduction allowance: 2.25 GPU-hours and one submission slot, based
  on the completed native pilot.
- Measured persistent footprint: 128.775 GiB of the 250 GiB cap.
- A durable coordinator marker and private operational ledger are established.

## Next actions

Verify the queued native reference, persistent, and optimized comparison on
matching inputs and placement when it completes. GPU batching requires its own
measured admission and native batch-one agreement. Then run the bounded adaptive
development/final experiment if its measured projection fits the remaining
budget and reproduction allowance.

Private records retain exact job IDs, connection settings, output paths, skill
hashes, and scheduler output. Public reports contain sanitized measurements.
