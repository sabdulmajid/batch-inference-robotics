# WATcloud operating runbook

This document is public and intentionally uses configuration variables for account-specific connection values. The user's launch message supplies the private settings. Store them only in an ignored local configuration with owner-only permissions.

## 1. Authorization and connection

Use the verified project account and existing Mac SSH configuration. The launch message supplies the jump alias, key path, login hosts, username, SLURM account, and QOS.

Inspect the effective SSH configuration before diagnosing a timeout. Preserve host-key verification. Do not overwrite SSH configuration, remove known-host entries, copy private keys, enable credential forwarding, or try unrelated credentials.

The known working pattern is:
~~~bash
ssh -J "$BIR_SSH_JUMP" -i "$BIR_SSH_KEY" "$BIR_SSH_USER@$BIR_LOGIN_HOST"
~~~

The fallback login host may be used through the same authorized route. Direct-connection failure is not evidence that the account is invalid. An unexplained host-key change or unavailable required credential is a genuine blocker; do not bypass it.

The user reported access verified on 2026-10-31. Recheck live state before allocating resources.

## 2. Read current scheduler state

Run lightweight checks on the login server, using the actual resolved username:
~~~bash
id -un
sacctmgr -n -P show assoc where users="$BIR_SSH_USER" format=Cluster,Account,User,Partition,QOS
sinfo -N -o '%N|%P|%t|%G'
scontrol show partitions -o
squeue -u "$BIR_SSH_USER"
~~~

Inspect relevant candidate nodes with scontrol show node NODE_NAME. Existing jobs can belong to other agents sharing the account. Inventory and partition maxima are not an allocation or a guarantee of availability.

The user-supplied snapshot is:
- RTX 4090: one 24 GiB GPU.
- RTX 3090: six 24 GiB GPUs across the cluster.
- RTX 2080 Ti: three 11 GiB GPUs.
- GTX 1080: one 8 GiB GPU.
- compute partition maximum: 24 hours.
- compute_dense partition maximum: seven days.

These are historical input facts, not values to force into a submission. Read current rules. Never force DOWN/DRAIN nodes or infer rules from partition names.

## 3. Sole coordinator and budget ledger

One coordinator owns all sbatch/scancel actions. Maintain one pending/running project GPU allocation across every agent and worktree. Subagents propose runs in a shared experiment manifest instead of submitting independently.

Before submission, record:
- Unique project run ID and project-specific job name/comment.
- Code commit, source/model revisions, environment fingerprint, configuration hash.
- Inputs, output expectations, profile, and benchmark condition.
- Requested nodes, GPUs/shards, CPUs, host RAM, temporary disk, and wall time.
- Budget reservation and persistent output/log locations.

Reserve GPU count multiplied by requested wall hours against the remaining 24-hour project GPU budget before submitting. Conservatively charge a shard allocation as one physical device for this budget. Reconcile actual allocated elapsed time after completion. Pending jobs do not consume GPU time, but keep their reservation until resolved.

Failed and idle allocations consume budget. Do not double-count SLURM job-step rows or generic and typed GPU TRES. Before every CPU-only submission, reserve requested CPUs multiplied by requested wall hours against the separate core-hour cap, including failures and idle time. Permit at most one pending or running heavy CPU-only job across all agents.

Before optional experiments, reserve the measured cost and required submission slots for final reproduction and evidence collection. New experiments must fit after outstanding reservations and this release reserve; shorten optional evaluation when necessary.

On a lost SSH response after submission, search using the unique project run name/comment and ledger before retrying. Never assume the submission failed. A local lock alone is insufficient if another orchestrator may operate remotely: use a durable task-owned submission lease/lock and reconciled scheduler state.

## 4. Resource selection

Use SLURM for GPU work and heavy CPU work. Login nodes are for inspection, small edits, and submission.

A shared request example is:
~~~text
--gres=shard:rtx_3090:4096,tmpdisk:20480
~~~
Counts are MiB. A shard must fit on one GPU, and its VRAM request is not enforced. Measure peak use and request adequate headroom.

A whole-GPU example is:
~~~text
--gres=gpu:rtx_3090:1,tmpdisk:20480
~~~
Do not request shard and gpu for the same allocation. Whole GPUs are appropriate when the memory-intensive native workload needs them or when isolated performance measurements justify them.

Explicitly set CPU count, system RAM, temporary disk, runtime, account, QOS, partition, and job name. The --mem option is system RAM, not VRAM. Request --nodes=1 for the initial model-parallel Captioner experiments.

Prefer one 3090/4090 for TempSeg feasibility. For faithful Captioner, estimate actual checkpoint bytes and request a justified three/four same-node allocation. The project ceiling explicitly allows up to four GPUs for this purpose, not as a default for every task.

Do not use the smaller GPUs for the native BF16 baseline without independently proving support and recording any numerical change. Do not silently quantize/offload to fit.

Preserve SLURM's device restrictions. Inside containers, verify both visibility and actual allocation; never replace the scheduler mask with all host GPU indices.

## 5. Storage and environments

Keep small configuration files in home. The user supplied a default home quota of 20 GiB; inspect actual usage:
~~~bash
getfattr -n ceph.quota "$HOME"
getfattr -n ceph.dir.rbytes "$HOME"
~~~

Discover an authorized writable project directory on an appropriate /mnt/wato-drive* mount. Check ownership/permission, free space, quota, and expected contents. Do not recursively scan or change permissions on unrelated directories.

Before model downloads, estimate snapshots, environments, datasets, outputs, and at least 20% headroom. Enforce the 250 GiB new-project-storage ceiling. The annotation workflow may need roughly 150 GiB for model/cache assets; inspect exact revisions and reuse valid task-owned assets rather than downloading duplicates.

Set project-specific model, package, and build caches there, including HF_HOME and FLORA_MODEL_CACHE. Keep HOME unchanged. Use locks around shared downloads.

Use an isolated environment; pin the actual compatible Python/framework/CUDA-related package versions. Do not install drivers, use sudo, alter global Python/Conda environments, or edit shell startup files.

Rootless Docker is an option inside an allocation:
~~~bash
slurm-start-dockerd.sh
~~~
Its image/storage costs must fit the requested temporary disk. Do not start privileged containers or expose unallocated GPUs.

Treat /tmp as disposable. Persist required outputs, run/configuration records, application memory telemetry, and logs before job end. Stage model files to temporary disk only when requested capacity and measured I/O benefits justify it.

## 6. Validate and execute

Generate a project-specific batch script with literal resolved #SBATCH values. Shell variables inside #SBATCH directives are not expanded. Validate the generated script and create persistent log directories first.

Include explicit account, QOS, partition, node/GPU request, CPU, host RAM, temporary disk, wall limit, job name, output/error paths, reproducible environment, and exact application command.

Disable automatic requeue with --no-requeue where supported. Inspect SLURM_RESTART_COUNT in the job wrapper. An unexpected restarted execution exits with an explicit incomplete status before model work; it must not consume another unreserved execution. The coordinator reconciles all prior attempt time and reserves any permitted retry. Site-driven restarts still count toward aggregate resource use.

Use robust shell error handling, ensure application failures propagate to the job exit status, and save partial records without falsely marking completion.

~~~bash
sbatch --test-only job.sbatch
sbatch --parsable job.sbatch
~~~

The first command checks scheduling syntax/resource admissibility. It neither reserves resources nor validates application correctness.

Inside the granted allocation:
1. Record nvidia-smi/device/driver/framework information.
2. Run an actual framework operation and check its result.
3. For a FLORA gate, load the actual pinned model and run the intended native request.
4. Measure GPU memory in the application and per-device telemetry, not only with a pre-run snapshot.
5. Save expected output artifacts and completion metadata.

A segmentation-only pass must be labeled as such. A full end-to-end gate requires actual captions and verified serialization.

Start with the 15-minute framework smoke, then bounded native probes. Use measurements to project cost before any larger run. The resource policy limits each job to two hours and the complete project to its aggregate budget. Partition maxima do not override these limits.

## 7. Monitoring, failures, and cleanup

Use one coordinator and reasonable polling, approximately every two minutes, with nonblocking host mechanisms where available:
~~~bash
squeue -j JOB_ID
scontrol show job JOB_ID
~~~

After termination:
~~~bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES
~~~

Inspect allocation records and steps carefully. MaxRSS measures host memory, not GPU memory. Verify the application outputs and completion markers as well as final state/exit code.

A timeout, OOM, unsupported kernel, or malformed result needs diagnosis before another submission. Permit at most one diagnosed transient retry per work item for transport, worker, or I/O failures. This cap does not change FLORA's pinned native segmentation boundary-repair policy or repair-window limits; that work still counts toward cost. A deterministic failure requires a concrete changed hypothesis; an unchanged repeat is not an experiment.

Cancel only exact job IDs created for this task and recorded in the ledger. Never cancel by username/account, run broad pkill, reset GPUs, delete shared caches, change scheduling priorities unfairly, or touch another agent's checkpoints.

At completion, release unnecessary task allocations, reconcile the ledger, preserve evidence, and publish only sanitized summaries. Record what ran, hardware type/count, elapsed time, peak memory, quality/performance results, artifact identifiers, and remaining limitations.

## 8. Stop conditions

Stop new expensive work when the resource cap would be exceeded, access cannot be validated, required storage is unavailable, native inference cannot fit safely, or the relevant software stack cannot execute it.

Continue useful independent implementation, tests, documentation, and already-feasible experiments. Report the exact blocked gate and smallest remaining requirement. Do not silently weaken the native baseline, buy new compute, create new credentials, or claim missing measurements.

## Official references

- https://cloud.watonomous.ca/docs/compute-cluster/slurm
- https://cloud.watonomous.ca/docs/compute-cluster/user-manual
- https://cloud.watonomous.ca/docs/compute-cluster/quotas
- https://cloud.watonomous.ca/machines
- https://slurm.schedmd.com/sbatch.html
- https://slurm.schedmd.com/sacct.html

