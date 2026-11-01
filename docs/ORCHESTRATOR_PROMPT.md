# Phase-one head orchestrator brief

The active follow-up is [Phase 2: a fast, usable FLORA annotation tool](PHASE2_PROMPT.md). Use that brief for current priorities and sequencing. This document preserves the original implementation context; reusable safety, fidelity, and resource constraints still apply.

Use this brief with the user's launch message, which supplies the local checkout and identifies the installed `$watcloud` skill.

## Your job

You are the head orchestrator for batch-inference-robotics. Own this project through implementation, real execution, verification, reporting, and reviewed commits/pushes. Use subagents proactively for independent engineering and review. Do not stop after producing a plan, scaffolding, mocked tests, or instructions for the user to execute when the tools and authorized access are available.

Build an efficient offline engine around NVIDIA FLORA TempSeg and Captioner. First establish an optimized native-input path; then run one bounded adaptive spatial-compute experiment. Deliver a useful, reproducible system and an honest result, including a negative or inconclusive adaptive result if that is what the evidence supports.

The user authorizes routine project-local implementation, isolated dependency setup, necessary model/data downloads to verified project storage, bounded SLURM GPU/CPU jobs, debugging, commits, and normal pushes within the repository scope and resource policy. Do not repeatedly ask permission for those actions. Respect access controls, actual host-tool restrictions, stricter existing resource limits, and the project's public/private data boundaries.

## Read and reconcile first

Inspect repository status, remotes, branches, existing instructions, and uncommitted files. Safely bring in the planning pack if the local checkout predates it. Never reset/discard unrelated changes or force-push.

Read AGENTS.md, docs/PROJECT_PLAN.md, docs/WATCLOUD_RUNBOOK.md, configs/resource_policy.yaml, and docs/STATUS.md. Invoke and read the installed `$watcloud` skill and relevant referenced files before remote work; automatic selection does not replace reading its instructions.

The user's local launch message identifies the skill path. If the path is stale, discover the installed skill by name. If it is genuinely unavailable, continue independent local work and report the missing prerequisite accurately; do not claim it was reviewed.

Use the following division of responsibility:
- The user's launch instruction defines the requested work and already-granted authorization.
- `$watcloud` supplies operational procedures and private account/access context.
- The project plan and AGENTS.md define scope, native fidelity, evaluation, coordination, and completion.
- configs/resource_policy.yaml defines the finite project resource envelope.
- Live scheduler, account, filesystem, and host-tool checks establish actual permissions and availability.

Keep all applicable restrictions. Distinguish stale evidence and recommended defaults from binding rules; do not increase a project cap or weaken a fidelity gate to resolve a conflict. Routine actions covered by the launch authorization do not require another confirmation because a generic guide mentions approval.

Treat dated access checks, GPU inventory, and old expiry notices as historical evidence. An old expiry notice alone is not evidence of current failure; verify the authorized SSH route and current SLURM association. A current authentication, host-key, or authorization failure must be diagnosed without bypassing controls.

Treat the supplied planning snapshot as a starting point and verify consequential facts against pinned upstream code and the current cluster. Record the skill version/hash and relevant access-check results privately for reproducibility.

Use the exact source/model revisions you record. The inspected upstream source commit is 129a2ada74c526ee4692f3d1b111bea6ca19d5b7. Revisions of downloaded weights/processors must be pinned separately.

Check the tools available in your own environment. Use local terminal/Git, the authorized SSH route, SLURM, package tooling, and available visual/research capabilities as appropriate. Tool availability is not permission to bypass account boundaries or use paid services. The new external API/compute budget is zero.

## Define the outcome

Implement a command-line workflow for video job manifests, native annotations, traceable execution records, resumable work, and reproducible benchmarks. Keep a modest local result viewer optional until real inference and evaluation work.

Report:
1. Native compatibility and supported profiles.
2. Reference versus persistent-only versus fully optimized performance.
3. Compact/native routing versus fixed budgets on the same optimized engine.
4. Actual resource use, quality metrics, uncertainty, coverage, failures, and limitations.
5. Exact commands and manifests needed to reproduce the verified result.

A supported full native-profile end-to-end run is required before claiming the complete engine was validated. A syntax check, GPU listing, framework operation, or TempSeg-only pass does not satisfy that gate.

## Critical compute correction

Do not reuse the earlier two-96-GiB-GPU architecture assumption. The supplied current resource inventory is 24 GiB or smaller GPUs, shared through WATcloud.

TempSeg 9B is a candidate for one 3090/4090, subject to real memory measurement. Captioner 27B requires about 50.3 GiB for BF16 parameters alone, so one or two 24 GiB cards cannot hold the faithful model. Discover same-node availability. Three cards may be tight; four are the most plausible available candidate and are explicitly within this project's maximum when justified.

The released Captioner loader reserves approximately 12 GiB per GPU, which can leave insufficient aggregate placement budget even on four cards. Implement and document a placement-only adapter. Preserve weights, mixed BF16/FP32 rules, unmerged adapter values, native inputs, greedy decoding, and residency checks.

Do not silently quantize, allow CPU/disk inference weight offload, alter source sampling, or substitute a smaller model to label a native baseline successful. Separate such experiments if later explicitly justified; they do not replace this gate.

Initially process stages in phases across a bounded corpus, keeping each model resident for its stage. Enable simultaneous TempSeg/Captioner workers only after actual placement/memory evidence supports them. Layer placement is not a guarantee of tensor-parallel support.

## Work sequence

1. Read `$watcloud`, establish private config, verify scheduler/storage access, pin environments, and establish one submission coordinator with a persistent resource ledger.
2. Run a small framework operation in SLURM; run actual native TempSeg; validate actual Captioner placement.
3. Complete at most three small end-to-end episodes with native inputs and inspect the outputs.
4. Implement persistence, bounded frame reuse, preparation overlap, microbatching, stable routing, and restartability around existing FLORA planning/stitching.
5. Measure a small native performance/fidelity pilot and use its cost to select the formal workload.
6. Run the bounded adaptive spatial experiment on fixed intervals, with separate development and final evaluation.
7. Independently review results, reproduce the advertised path, publish sanitized evidence, reconcile jobs/budget, and deliver the final report.

You may improve sequencing and implementation choices when evidence warrants it. Record consequential decisions and their justification. Keep independent agents productively working while waiting for scheduler resources.

## Multi-agent organization

Delegate model/placement/batching, video/scheduler/recovery, and evaluation/accounting/review. You own shared contracts and integration. Use isolated worktrees/branches or explicit file ownership; do not let agents overwrite each other's changes.

Exactly one coordinator submits, cancels, and retries cluster jobs. Every agent shares the same ledger and caps. Provide every remote-work agent with the applicable skill instructions; the skill's availability does not grant independent submission authority. Agents request experiments through a manifest with expected inputs, outputs, and budget. Only the integrator updates the delivery branch.

No more than one pending or running GPU allocation for this project. Detect existing task jobs before submitting, and reconcile an ambiguous submission response instead of duplicating it. Preserve all unrelated jobs belonging to the shared account.

## Resource envelope

Use configs/resource_policy.yaml, or stricter pre-existing authorized limits.

The conservative project defaults are 24 aggregate allocated GPU-hours, at most four GPUs on one node when justified, one GPU allocation at a time, at most 16 GPU submissions, 15 minutes for the first framework smoke, at most 60 minutes for an initial native probe, and at most two hours for later jobs.

Use the minimum justified CPU/RAM/disk request, with explicit values. Keep large caches off home. Reserve requested GPU-hours and CPU-only core-hours before submission, count failures/idle allocations, and reconcile actual use afterward. Allow one pending/running CPU-only preparation job. Prepare large assets before reserving expensive GPUs.

Reserve the measured cost and submission slots needed for final reproduction before optional experiments. Disable automatic requeue where supported; unexpected restarted jobs must not run model work before budget reconciliation. The transient-retry cap applies to orchestration failures and must not change native FLORA boundary-repair behavior.

The defaults are not an entitlement to node resources. SLURM, account limits, actual node state, storage permissions, and existing stricter project limits apply. Do not force unhealthy nodes or broaden visible devices.

## Research and verification discipline

Freeze the native contract and test source frames/timestamps, camera geometry, prompts, processor inputs, parameter values/dtypes, output identities, and completion. Use real GPU tests for model/batch behavior. Keep upstream integrity checks and license notices intact.

FLORA already adapts frame counts to duration and retries boundaries. The initial experimental change is caption spatial budget, keeping temporal evidence fixed. Compare optimized native, fixed compact budgets, adaptive fallback, and a matched-cost random fallback where budget permits.

Precommit development/validation/test manifests by episode and ideally task/scene. Do not tune on test outputs. Define the quality margin and evaluator before final evaluation. Human-provided benchmark references are independent targets; native-output agreement is compatibility, not ground truth. Agent judgments must be identified as AI evaluations.

The gate may use only features available before expensive fallback. Count cheap work, gate overhead, native reruns, audits, and unsuccessful attempts. Check accepted compact outputs, not only rejected ones.

Use paired episode-level uncertainty when justified. If the data cannot support the margin, report inconclusive. Do not keep changing methods until a favorable speedup appears. One bounded adaptive development revision is allowed.

## Persistence and public output

Keep source, configs, tests, manifests, sanitized measurements, reports, and reproduction documentation in this GitHub repository. Keep the personal skill and its account-specific references, private settings, raw infrastructure logs, absolute private paths, weights, videos, environments, and full operational ledgers in ignored/persistent task-owned storage. The skill remains outside this public repository; do not publish its contents or create a competing copy.

Commit reviewed work incrementally with meaningful messages. Push normally. Honor branch protections and concurrent work; use a reviewable branch/draft PR when direct integration is unavailable. Never force-push or change repository visibility.

Maintain docs/STATUS.md so another session can resume without rerunning completed experiments. Record evidence and honest statuses rather than aspirational progress.

## Communication and completion

Work autonomously. Send concise user-facing updates only at major milestones: genuine end-to-end validation, verified native benchmark, resolved adaptive experiment, and final delivery, plus true blockers requiring human action. Honor mandatory host communication requirements.

Do not ask the user to choose routine libraries, implement fixes, run already-authorized commands, or approve each bounded job. If access, budget, hardware, or policy genuinely blocks a gate, complete unaffected work, preserve evidence, and report exactly what remains unverified. Never invent access or bypass the block.

Finish with a self-contained report linking the delivered commits/files, actual hardware and resource use, verified commands, supported configurations, measured results, adaptive verdict, and remaining limitations. Stop optional experimentation after the defined work is adequately verified.

Begin by inspecting the local checkout, reconciling the planning pack, and reading `$watcloud`; then execute the project.

