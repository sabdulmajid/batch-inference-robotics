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
- Captioner is submitted for three same-node RTX 3090 GPUs. Complete native
  annotation remains unvalidated until that job and its artifacts pass.
- The adaptive protocol and task-separated split are frozen. All three exact
  development/final videos passed preparation with original bytes and clocks.
- Native temporal and joint AI reference review is frozen for the three initial
  development episodes and their 40 independent references.
- A small CPU-only resume check is queued after successful Captioner completion.
  It will verify an isolated copy of the exact inference state.
- No performance or adaptive quality result exists.

## Resource state

- GPU-hours used/reserved: 0.267222 / 3 of 24, including the failed allocation.
- GPU submissions: 4 of 16.
- CPU-only core-hours used/reserved: 3.006111 / 0.166667 of 32.
- Measured persistent footprint: 128.775 GiB of the 250 GiB cap.
- A durable coordinator marker and private operational ledger are established.

## Next actions

Verify Captioner placement, residency, complete outputs, and final scheduler
status. Captioner will continue the signed TempSeg checkpoints from the same
code revision and input manifest. Checkpoint headers estimate 50.956 GiB of
native weights before activations and caches. Runtime fit remains unproved.
Complete at most three full episodes before any larger benchmark.

Private records retain exact job IDs, connection settings, output paths, skill
hashes, and scheduler output. Public results will be added after verification.
