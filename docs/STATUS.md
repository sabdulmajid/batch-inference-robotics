# Project status

Updated: 2026-10-31

## Current state

Planning pack created after inspecting the public repository and FLORA reference source. Updated to use the locally installed `$watcloud` skill as the operational guide, with the repository retaining project-specific limits and verification gates.

The user reports that the skill is installed and validated on their Mac. Its Mac-local file was not accessible from the planning workspace, so this update integrates the supplied installation summary; it does not certify or modify the skill's actual contents. The executing local agent must read the installed skill before remote work.

- Repository initially empty.
- No implementation has been validated.
- No SSH session or WATcloud job has been started by this planning workflow.
- No model weights or datasets have been downloaded.
- No speedup, GPU memory measurement, or quality-preservation claim exists.

## Next gates

1. Synchronize the local checkout without losing existing work.
2. Read the installed `$watcloud` skill; resolve private connection settings, live scheduler permissions, and authorized storage.
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

