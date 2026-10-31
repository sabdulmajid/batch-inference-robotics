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
- CPU preparation is submitted. Pinned model files total 129,079,719,871 bytes.
- No framework GPU test, TempSeg inference, Captioner inference, or complete
  native annotation has passed yet.
- No performance or adaptive quality result exists.

## Resource state

- GPU-hours used/reserved: 0 / 0 of 24.
- GPU submissions: 0 of 16.
- CPU-only core-hours reserved: 8 of 32; actual use awaits completion.
- Planned new persistent footprint: 180 GiB of the 250 GiB cap.
- A durable coordinator marker and private operational ledger are established.

## Next actions

Verify CPU job completion and expected asset/environment artifacts. Then run the
framework smoke, native TempSeg, and Captioner placement probes in that order.
Complete at most three full episodes before any larger benchmark.

Private records retain exact job IDs, connection settings, output paths, skill
hashes, and scheduler output. Public results will be added after verification.
