# Project status

Updated: 2026-10-31

## Current state

Planning pack created after inspecting the public repository and FLORA reference source.

- Repository initially empty.
- No implementation has been validated.
- No SSH session or WATcloud job has been started by this planning workflow.
- No model weights or datasets have been downloaded.
- No speedup, GPU memory measurement, or quality-preservation claim exists.

## Next gates

1. Synchronize the local checkout without losing existing work.
2. Resolve private connection settings, live scheduler permissions, and authorized storage.
3. Freeze upstream revisions, contracts, evaluation splits, and the shared resource ledger.
4. Run framework smoke, native TempSeg, Captioner placement, and at most three full episodes.
5. Implement and benchmark the native engine, then run a bounded adaptive captioning experiment.

## Required ongoing records

The orchestrator should add concise, sanitized entries for:

- Current commit and phase.
- Decisions and reasons.
- Meaningful verification results.
- Native/adaptive configurations and public manifest IDs.
- Aggregate GPU-hour budget used/reserved/remaining.
- Exact remaining work and the next safe action.
- Links to committed reports and private artifact identifiers, without exposing private paths.

Keep raw job IDs, host details, absolute storage paths, commands containing private settings, and scheduler output in the ignored operational ledger.

