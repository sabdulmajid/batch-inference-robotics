# Agent instructions

## Mission and authority

Implement and evaluate the project described in docs/PROJECT_PLAN.md. Work autonomously on routine implementation, dependency setup, bounded experiments, debugging, review, commits, and pushes within the user's launch authorization and configs/resource_policy.yaml. Respect actual account, filesystem, scheduler, and host-tool restrictions.

The launch instruction supplies authorization for project-specific model downloads and scheduled multi-GPU probes within the stated envelope; do not repeatedly ask for permission already granted. Never invent access, disable a safeguard, exceed a cap, or infer ownership of unrelated jobs from a shared Unix username.

Read existing local instructions and preserve user work. The plan is guidance: improve architecture and sequencing when evidence supports it, record important decisions, and retain scope, fidelity, and budget boundaries. Do not stop at a scaffold, mock-only test, plan, or unexecuted script when the required access is available.

## Coordination

- The head orchestrator owns integration, shared interfaces, the resource ledger, and final reporting.
- Delegate independent implementation and review tasks. Use isolated branches/worktrees or non-overlapping files; avoid concurrent edits to shared interfaces.
- Exactly one resource coordinator submits, cancels, or retries SLURM jobs. All agents share the same project-wide budget and one pending/running GPU allocation.
- Record an experiment request before submission. Reconcile ambiguous submissions against scheduler state before retrying.
- Reserve GPU/CPU costs and a final reproduction allowance. Disable automatic requeue where supported and reconcile unexpected restarts before model work.
- Persist checkpoints in docs/STATUS.md and the private run ledger so interruption or context compaction does not restart experiments.
- User-facing updates should mark substantial milestones or genuine blockers. Honor any mandatory host communication requirements; avoid narrating routine tool calls.

## Fidelity and scientific integrity

Keep native preprocessing, source clocks, camera layouts, prompts, pinned revisions, dtype/value checks, parsing, and stitching intact in native mode. Explicit placement-only changes are permitted and documented.

Do not silently quantize, offload inference weights to CPU/disk, alter frame budgets, merge/recast adapters, or substitute a different model to claim native success. Keep experimental changes separately named and measured.

A completed TempSeg run is not a completed TempSeg-plus-Captioner run. A successful submission, visible GPU, mocked test, or syntactically valid caption is not evidence of successful annotation quality.

Independent references establish annotation quality; native-output agreement establishes compatibility. Precommit splits, metrics, and decision thresholds. Preserve failures and denominator accounting. Negative and inconclusive adaptive results are valid outcomes.

## Compute and storage

Read the locally installed `$watcloud` skill and its relevant referenced material before remote work. Resolve the exact skill path from the user's launch message or installed-skill discovery. Use it for the existing SSH route, account, scheduler, storage, environment, and monitoring procedures. Keep private keys local and preserve host-key verification.

Read docs/WATCLOUD_RUNBOOK.md as the project-specific overlay and configs/resource_policy.yaml as the project spending envelope. The skill does not expand these caps. Apply actual site/account restrictions and any stricter existing authorized limits. Distinguish a recommendation, a dated fact, and a binding restriction when reconciling guidance; the user's existing authorization covers routine in-scope jobs and must not trigger repeated confirmation.

Verify current access through the skill's authorized route and live scheduler/account checks. Dated inventory, prior access tests, and old expiry notices are historical evidence, not a substitute for current verification. If the skill cannot be found, continue local work and report the missing prerequisite accurately before remote execution.

Give every subagent assigned remote work the applicable skill instructions and project constraints. The single coordinator retains exclusive submission/cancellation/retry ownership.

Keep login-server work light. Use SLURM for inference, checkpoint loading, large preprocessing, package compilation, and substantial data operations. Large caches and artifacts go to a verified, authorized persistent project directory; home and job-local temporary storage are not substitutes.

Never cancel by username/account, reset GPUs, kill unrelated processes, change drivers, use sudo, weaken permissions, or delete shared caches. Cancel only exact task-created IDs recorded in the ledger.

Validate a batch request, then run a small actual application test, then size any larger workload from measurements. Preserve scheduler GPU visibility. Worker isolation may narrow the granted device set, never broaden it.

## Repository and release

The repository is public. Keep the personal skill and its account-specific references, real connection settings, private absolute paths, raw cluster/account logs, tokens, private keys, datasets, model weights, and environments out of commits. Use .local/ and other ignored task directories for these.

Keep upstream source pinned and intact, with integrity and license checks. Prefer adapters around native code. Do not relabel covered upstream code as permissively licensed.

Stage explicit reviewed files. The integrator alone updates the delivery branch. Use normal fast-forward pushes; never force-push, discard unrelated changes, or overwrite concurrent work. Preserve branch protections. If protected, push a reviewable branch/draft PR without bypassing policy.

Use meaningful tests for timestamp fidelity, model placement, batching identities, retry/resume, and metric accounting. Verify real GPU execution where available. Stop optional sweeps after the predefined gates and bounded adaptive experiment.

At completion, report separately: implemented software, real end-to-end validation, fidelity, performance, adaptive-quality verdict, resource consumption, reproduction commands, and remaining limitations.

