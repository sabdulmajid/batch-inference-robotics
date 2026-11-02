# Recorded rollouts to an existing offline-RL trainer

Deliver one complete path from recorded video and robot transitions to reviewable
annotations, optional terminal outcome evidence, and a robomimic IQL dataset.
Preserve the annotation CLI, native engine, checkpoints, and JSONL labels.

## Scope and contracts

- Start with the official robomimic Lift low-dimensional dataset. Preserve its
  observations, next observations, seven-dimensional actions, environment rewards,
  recorded simulator clocks, and source bootstrap masks. Its derived `dones` do
  not establish physical termination. Unknown termination/truncation stays unknown.
- Attach versioned trajectory data through episode and source identities. Keep
  trajectory configuration out of completed visual inference identities. Validate
  clocks and camera mappings explicitly; never pair streams merely by index.
- Use a portable JSON bundle with an exact immutable `payload_json` string and
  its SHA256. Corrections use a separate append-only hash chain. Preserving the
  original payload bytes avoids Python/browser floating-point serialization drift.
- Keep Rewarder independent of segmentation/captioning, with its own request,
  attempts, failures, source/model/runtime provenance, and raw terminal output.
  Training reward mappings are separate and versioned. Unknown or failed scores
  cannot become zero rewards. A recording cut cannot become an episode terminal.
- Export state observations/actions to pinned robomimic IQL. Select environment
  rewards or native FLORA rewards explicitly. Preserve source fields separately.
  Retrospective labels and scores never enter policy observations by default.

## Finite implementation and verification

1. Fix review identity changes after resume. Preserve inapplicable historical
   edits visibly and reject new corrections to pending/failed predictions.
2. Implement one trajectory adapter, bundle validation, and trainer export.
   Test identity, shape, missing-data, clock, boundary, reward, and duplicate-output
   failures. Round-trip real data exactly.
3. Implement Rewarder integration and recovery tests without native inference.
   Read and pin the actual upstream contract and isolate its dependencies.
4. Run bounded CPU IQL optimizer updates, save/reload, and a subsequent update.
   Execute fresh simulator episodes and separately verify rendered replay where
   supported. Report this as integration evidence, not learned competence.
5. Ship a static video-first browser app. Own-data files stay in the browser.
   Verify review/download/reopen/CLI/export/training-loader round trips. Publish
   licensed examples on a free anonymous host tied to this repository if possible.
6. Package resources for wheel installation, publish tested quickstarts and
   reviewed evidence, run CI, and deliver normally through GitHub.

## Resource and coordination constraints

GPU work is paused on all machines. Cumulative use is not reset: the prior
release recorded 18.563889/24 GPU-hours and 16/16 submissions. Native Rewarder
validation requires a new explicit pause/submission-limit decision. No GPU job
is part of this implementation phase.

One coordinator owns resource accounting and any remote CPU submission. Use
isolated environments, bounded CPU jobs, and existing storage limits. Local CPU
smoke work uses two threads and a ten-minute execution ceiling initially. No paid
APIs, new paid compute, domain purchase, or recurring service is authorized.

File ownership separates trajectory/bundle integration, Rewarder, trainer
integration, and the static interface. The coordinator owns shared CLI changes,
packaging, integration, commits, deployment, and independent final review.

## Evidence boundaries

The historical 1.57× result describes one six-video annotation comparison only.
It is not Rewarder or policy-learning evidence. Synthetic fixtures remain labeled
tests. A CPU optimizer/checkpoint smoke is not policy improvement. A later controlled
BC / environment-reward IQL / native-FLORA-reward IQL comparison will use identical
trajectories and settings, held-out simulator episodes, and multiple seeds.

The public example must distinguish native predictions, source task descriptions,
review edits, unavailable scores, and replay from new policy execution. Keep full
review history in the bundle/journal; label JSONL contains the latest applicable
correction. Preserve upstream and dataset licenses and all prior reports.
