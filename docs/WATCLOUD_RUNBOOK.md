# WATcloud project compute overlay

Use the locally installed `$watcloud` skill as the operational guide for this project. It owns the account/access context, SSH route, scheduler procedures, storage conventions, environment setup, and monitoring instructions. Read the actual skill and relevant referenced material before remote work.

This document adds project-specific coordination, spending, model-placement, and verification requirements. It intentionally omits private connection values and a duplicate cluster inventory.

## 1. Resolve guidance and verify access

The user's launch message identifies the installed skill. If that path is stale, discover the installed `watcloud` skill by name. Keep the skill and its account-specific references outside this public repository. Resolve required settings into ignored, task-owned local configuration and record the skill version/hash privately.

Use these sources for their respective purposes:

| Source | Responsibility |
| --- | --- |
| User's launch instruction | Scope and already-granted authorization |
| Installed `$watcloud` skill | Operational procedure and private account/access context |
| docs/PROJECT_PLAN.md and AGENTS.md | Engineering, fidelity, evaluation, coordination, and completion |
| configs/resource_policy.yaml | Project compute, storage, concurrency, and experiment caps |
| Live account, scheduler, filesystem, and host tools | Current permissions, availability, and enforceable restrictions |

The skill does not expand the project resource envelope. Honor actual site/account restrictions and stricter existing authorized limits. Distinguish a recommended default, a dated fact, and a binding restriction before treating guidance as a conflict. Routine in-scope jobs are covered by the launch authorization; do not repeatedly request permission already granted.

Historical access tests, inventory, and expiry notices are diagnostic context. An old expiry notice alone does not establish a current access failure. Verify the existing authorized SSH route and current account/SLURM association as directed by the skill. Preserve host-key verification, local private keys, scheduler device restrictions, and unrelated account activity.

A current authentication/authorization failure, unexplained host-key change, or genuinely missing skill is a prerequisite problem to diagnose accurately. Complete unaffected local work and report what remains blocked; never invent access or bypass controls.

## 2. Sole coordinator and budget ledger

One coordinator owns all project sbatch/scancel actions, including heavy CPU preparation. Maintain one pending/running project GPU allocation and at most one pending/running CPU-only preparation job across every agent and worktree. Give remote-work agents the applicable skill instructions, but keep submission authority with the coordinator.

Before submission, record:

- Unique project run ID and project-specific job name/comment.
- Code commit, source/model revisions, environment fingerprint, configuration hash.
- Inputs, expected outputs, native/experimental profile, and benchmark condition.
- Requested nodes, GPUs/shards, CPUs, host RAM, temporary disk, and wall time.
- Budget reservation and persistent output/log locations.

Use configs/resource_policy.yaml for every limit. The current project envelope includes 24 aggregate allocated GPU-hours, at most four same-node GPUs per job when justified, 16 GPU submissions, and a two-hour maximum for later jobs. These are project caps, not cluster entitlements.

Reserve GPU count multiplied by requested wall hours before submitting. Conservatively charge a shard allocation as one physical device for this budget. Pending jobs retain their reservation until resolved. Reconcile actual allocated elapsed time after completion, including failed and idle allocations.

Reserve CPU-only requested cores multiplied by requested wall hours against the separate core-hour cap. Avoid double-counting job-step rows or generic and typed GPU TRES when reconciling actual allocations.

Before optional experiments, reserve the measured cost and submission slots needed for final reproduction and evidence collection. Shorten optional evaluation if needed to preserve that reserve.

On a lost SSH response after submission, reconcile the unique project run name/comment and ledger with scheduler state before retrying. Use a durable task-owned submission lease/lock if another orchestrator could operate remotely; a local worktree lock alone is insufficient.

A shared Unix username does not establish job ownership. Cancel only exact task-created job IDs recorded in the ledger.

## 3. Resource selection for FLORA

Follow the skill for live node, partition, GRES, account, and QOS discovery. Use current allocation/state information; inventory totals are not free GPUs. Select the smallest justified request supported by live rules and measured application needs.

The planning hardware snapshot consists of GPUs with 24 GiB or less. Verify it before use. Native TempSeg is a candidate for one 3090/4090, subject to runtime measurement. Native Captioner 27B needs approximately 50.3 GiB for BF16 parameters alone, plus runtime overhead. One or two 24 GiB devices cannot hold that faithful model.

For Captioner, inspect actual checkpoint sizes and same-node topology. Three GPUs are conditional; up to four are allowed when justified. This ceiling is not a default allocation for every task. Multiple GPU memories do not combine without explicit application-supported placement.

The released Captioner loader's per-device reserve may prevent placement even on four cards. Use the documented placement-only adapter described in the project plan. Preserve parameter identity/dtypes, native input construction, generation settings, and all-GPU inference residency. Quantization, CPU/disk weight offload, fewer temporal frames, or model substitution does not satisfy the native gate.

Begin with staged TempSeg and Captioner execution. Do not assume both models can remain resident together. Layer placement does not establish tensor-parallel support.

Use shared GPU memory requests when the complete workload safely fits and the skill's rules permit them. The native models will generally need whole devices; performance isolation may also justify a whole-device request. Never mix shard and whole-GPU requests for the same allocation. Preserve the scheduler's granted device set; worker/container isolation may narrow it, never broaden it.

## 4. Storage and environment

Use the skill to discover an authorized persistent project directory and check live permissions, quota, and free space. Keep small configuration files in home; redirect large model, package, and build caches to the verified project storage.

Enforce the 250 GiB new-project-storage ceiling and at least 20% free headroom. Estimate model snapshots, environments, datasets, outputs, and temporary copies before downloading. Reuse valid pinned task-owned assets; coordinate downloads with locks.

Use an isolated project environment and pin actual compatible Python/framework/CUDA-related versions. Keep HOME unchanged. Preserve scheduler device restrictions when using an environment or rootless container. System drivers, global environments, shared caches, and other agents' files remain outside this task.

Use SLURM for checkpoint loading, inference, substantial decoding/preprocessing, package compilation, and substantial asset preparation. Prepare large assets before allocating expensive GPUs. Keep login-server work light.

Treat job-local temporary storage as disposable. Request enough temporary disk for the chosen environment and workload, then persist required results, logs, and completion metadata before job termination.

## 5. Validate requests and prove the application

Generate project-specific batch scripts using the skill's current instructions. Use literal resolved #SBATCH values, explicit resource limits, persistent log paths, a reproducible environment, and the exact application command. Create log directories before submission.

Validate each changed resource request with sbatch --test-only. This is a scheduler validation, not an application test or reservation. Submit a short real test only after validating the request and recording its budget reservation.

Disable automatic requeue with --no-requeue where supported. Inspect SLURM_RESTART_COUNT in the wrapper. An unexpected restarted execution exits with an explicit incomplete status before model work; the coordinator reconciles prior attempt time and reserves any permitted retry. Site-driven restarts still count toward resource use.

Propagate application failures to the job exit status. Save partial records without marking unfinished work complete.

Progress through these gates:

1. At most 15 minutes for an actual small operation in the project's GPU framework; check numerical output and GPU execution.
2. Bounded native TempSeg and Captioner-placement probes; initial native probe limits are at most 60 minutes.
3. At most three short complete native episodes through decoding, TempSeg, stitching, Captioner, and serialized annotations.
4. Verify resume/retry behavior and output identity.
5. Use measured memory, elapsed time, and per-episode cost to select any larger workload.

A visible GPU, successful submission, framework operation, or TempSeg-only output does not satisfy the complete annotation gate. Record the precise gate passed.

Measure peak GPU memory per device in the application and suitable device telemetry. MaxRSS is host RAM. Verify actual model residency and expected output artifacts, not only pre-run GPU listings.

## 6. Monitoring, retries, and completion

Use the skill's scheduler inspection and completion commands with project-specific job IDs. Poll at the interval in configs/resource_policy.yaml rather than allowing every agent to poll independently.

Verify final state, exit code, allocation records, application outputs, and completion markers. A timeout, OOM, unsupported kernel, or malformed result needs diagnosis before another submission.

The transient-retry cap applies to transport, worker, or I/O failures. It does not change the pinned FLORA segmentation boundary-repair algorithm or its native repair-window limits; all such work still counts toward cost. A deterministic failure requires a concrete changed hypothesis, not an unchanged repeated run.

At completion, release unnecessary task allocations, reconcile budgets and ownership records, preserve evidence, and publish sanitized summaries. Record hardware type/count, elapsed time, peak memory, quality/performance measurements, artifact identifiers, reproduction commands, and remaining limitations.

Keep raw account/node details, job IDs, private absolute paths, SSH settings, and complete operational logs in ignored task-owned storage. Public reports should contain the measurements and provenance needed to assess the result without copying the personal skill or private infrastructure records.

## 7. Stop conditions

Stop new expensive work if it would exceed the project envelope, required access/storage cannot be validated, faithful native inference cannot fit safely, or the software stack cannot execute it within the constraints.

Continue useful independent implementation, tests, documentation, and feasible already-authorized work. Report the exact blocked gate and the smallest remaining requirement. Preserve native fidelity and truthful result labels.

For routine implementation and bounded jobs already authorized by the launch instruction, proceed without another permission request. A genuine account, security, budget, or policy block must remain visible.

## References

The installed `$watcloud` skill is the operational starting point. Consult its referenced official documentation when current rules or procedures require verification. The project plan records the separate FLORA source/model evidence and scientific contract.

