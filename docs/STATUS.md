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
- The first TempSeg allocation completed model generation but failed when
  serializing set-valued loader diagnostics. All three work items remain failed;
  no complete annotations were exported. A deterministic serialization fix passed
  the actual Transformers loading-info and checkpoint resume tests. Peak GPU allocation was 18.353 GiB on one RTX 3090.
- Captioner and complete native annotation remain unvalidated.
- No performance or adaptive quality result exists.

## Resource state

- GPU-hours used/reserved: 0.157500 / 0 of 24, including the failed allocation.
- GPU submissions: 2 of 16.
- CPU-only core-hours used/reserved: 2.752222 / 0 of 32.
- Measured persistent footprint: 128.775 GiB of the 250 GiB cap.
- A durable coordinator marker and private operational ledger are established.

## Next actions

Fix and test diagnostic serialization, then rerun fresh native TempSeg.
Complete a faithful Captioner placement probe on at least three same-node
24 GiB GPUs. Checkpoint headers estimate 50.956 GiB of native weights before
activations and caches; runtime fit is not yet established. The framework smoke passed;
see [its sanitized record](../reports/framework-smoke.json).
Complete at most three full episodes before any larger benchmark.

Private records retain exact job IDs, connection settings, output paths, skill
hashes, and scheduler output. Public results will be added after verification.
