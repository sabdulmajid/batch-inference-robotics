# FLORA inference engine: execution plan

Date: 2026-10-31. This is an implementation and evaluation brief, not a report of completed experiments.

## 1. Objective and finite scope

Build a reliable batch annotation engine around the released FLORA TempSeg and Captioner. Demonstrate the effects of persistent model lifetime, bounded frame reuse, preprocessing overlap, and real request batching. Then test one constrained research hypothesis: whether selective compact/native spatial budgets reduce captioning compute at a predeclared quality margin compared with the best fixed budget on the same optimized engine.

Deliver working software, actual measurements, explicit failure states, reproducible commands, and a positive, negative, or inconclusive research conclusion. The engine remains valuable if adaptive routing fails.

The initial scope is offline video annotation. Rewarder optimization, robot policy training, new model training, multi-node inference, an Internet service, and extensive UI work are optional future projects. A local result viewer may be added after the core gates.

The agent may improve routine design decisions and task assignments. It must preserve the native contract, truthful comparisons, non-disruptive cluster operation, and the agreed resource envelope.

## 2. Verified source starting point

Inspected upstream: NVIDIA/Flora, commit 129a2ada74c526ee4692f3d1b111bea6ca19d5b7. Pin source and model revisions separately; do not assume the source commit identifies downloaded weights.

The annotation entry point is captioner/infer.py. Under captioner/batch_utils/, cli.py loops over episodes; api.py exposes DataEnginePipeline and uses a global request lock; pipeline.py creates a new Session per episode; models.py replaces the resident model when switching stages and closes it afterward. The generation helper handles a single output. media.py decodes selected frames by scanning from the beginning. Single-view captioning already gathers frame requests, while other paths repeat decoding.

The native code already provides frame planning, profile-specific context, stitching, adaptive boundary retries, source identifiers, and audit metadata. Reuse those capabilities. A new scheduler is not a reason to rewrite temporal semantics.

Inspect these files first:
- captioner/batch_utils/api.py, cli.py, pipeline.py, media.py, models.py, preprocessing.py
- captioner/batch_utils/vendor/tempseg_fp32_loader.py
- native temporal scheduler/stitcher and caption planning modules under vendor/
- upstream AGENTS.md, model cards, actual LICENSE, and model manifests

Keep downloaded upstream files, manifests, and integrity checks intact. Implement resource-aware loading and scheduling in explicit adapters. Preserve attribution when any upstream code is redistributed.

## 3. Hardware feasibility before architecture commitments

The supplied WATcloud snapshot has six RTX 3090s and one RTX 4090, each around 24 GiB, plus smaller GPUs. These are cluster totals. Live SLURM state and actual same-node topology govern feasibility.

Approximate BF16 weight storage:
- TempSeg 9B: 16.8 GiB before FP32 adapter parameters, special parameters, activations, vision processing, and cache.
- Captioner 27B: 50.3 GiB before runtime overhead.

One 24 GiB GPU is the first TempSeg candidate. One or two such GPUs cannot hold the full Captioner weights. Three same-node GPUs are a conditional, tight possibility; four are the strongest candidate in the supplied inventory. Neither is a proven working configuration.

The current Captioner loader sets each device's max_memory to approximately total GiB minus 12. Across four 24 GiB cards this allows at most 48 GiB for placement, below the approximate weight floor. Implement a documented placement-only adapter, deriving per-device budgets from actual parameter sizes and realistic activation headroom. Keep the native all-GPU residency verification.

A placement-adapted reference preserves model identity, BF16/FP32 treatment, unmerged FP32 LoRA, input construction, prompt, and generation settings. It changes where layers reside. Call it a placement-adapted reference; do not call it an untouched official execution.

Verify the device map, parameter values/dtypes, computational buffers, and absence of CPU/disk inference offload. CPU initialization, decoding, and audit hashing are normal CPU work. Changing quantization, numerical precision, frame count, image size, or model identity is a different experimental configuration.

Layer placement is not tensor parallelism. Do not assume a GPU count is a supported tensor-parallel degree. Do not combine independent nodes' memory.

## 4. Execution gates

### Gate A — local and remote preflight

Inspect the local checkout, existing instructions, status, branches, remotes, and uncommitted work. Integrate the planning commit safely. Discover available host tools and use the existing Git/SSH credentials.

Verify account, QOS, partitions, live GRES, existing jobs, candidate node resources, and project storage. Create an isolated project environment and private operational configuration. Establish one submission coordinator and a durable run/budget ledger before any GPU job.

Prepare large assets and dependencies through appropriate scheduled CPU work so a multi-GPU allocation is not consumed waiting for downloads or package compilation. Download one pinned copy per needed model and use locks to avoid duplicate agent downloads.

### Gate B — genuine GPU feasibility

Run the project's actual GPU framework in a small allocation; verify a small numerical operation, device execution, and memory. This is a framework test, not a FLORA test.

Then run actual native TempSeg on one short supported single-view video, batch one. Add a clearly named segmentation-only entry point if needed because the public annotation API continues into Captioner.

Estimate Captioner memory from checkpoint metadata. Request the smallest justified same-node allocation; four 3090s are a reasonable initial faithful candidate when three cannot be justified. Execute one native caption request with the placement adapter, then a representative native-size request. Diagnose before retrying. Keep all jobs within the resource policy.

Use a standard single-view profile first. A pass does not validate three-camera, longer-window, variable-rate, or maximum-context configurations.

After measuring the pilot, reserve enough GPU-hours and submission slots for the final release reproduction and evidence collection. Optional experiments must fit after this reserve and outstanding job reservations.

### Gate C — small complete end-to-end test

On at most three short valid episodes, run actual decoding, TempSeg, native reconstruction, caption-context construction, Captioner, and JSON export. Inspect the source and generated labels using available visual tools; preserve the provenance and distinguish inspection from independent benchmark scoring.

Validate a resume/retry path and verify no duplicate or misrouted outputs. Record final job exit status, actual outputs, every visible/used GPU, peak memory, duration, and configuration.

Do not start a large performance or quality run until this gate passes.

### Gate D — optimized native engine

Start with staged corpus execution:
1. Retain TempSeg across a bounded group of episodes.
2. Persist segmentation and audit records.
3. Release TempSeg and retain Captioner across the resulting caption requests.

This fits limited hardware more naturally than assuming both models can remain resident. Enable simultaneous stage workers only after the full relevant three-GPU Captioner envelope is proven and the resource accounting supports a one-plus-three split. If staged execution is faster or more efficient, keep it.

Implement bounded frame caching, asynchronous preparation, compatible microbatches, result routing, atomic output completion, and restartable work. Start with the reference Transformers backend; a vLLM or other backend port is optional after native compatibility is established.

### Gate E — bounded adaptive experiment

Keep TempSeg, intervals, source timing, context selection, camera layout, prompts, and decoding fixed. First vary caption spatial budget only. Define budgets in pixel area, preserve aspect ratio and processor compatibility, and log realized image sizes/visual tokens. Reduced temporal sampling is a separate future experiment.

Use a compact first pass plus a small gate and optional native fallback. Count compact work, routing, fallback, runtime audits, failures, and retries. A gate can use cheap-input features and generation statistics available at decision time. Token probability and low motion are not calibrated correctness guarantees.

Offline paired compact/native results may develop the gate. Runtime routing must not require the full native result on every example. Audit accepted compact outputs, including subtle small-object, occlusion, and short-action cases.

Allow one bounded development revision, then freeze the gate before the held-out test. A loss or inconclusive result still completes the experiment.

### Gate F — reproducible release

Run the advertised smoke/reproduction command from a clean project environment or equivalently isolated checked-out commit. Verify relevant automated tests and one real complete inference run. Publish sanitized configurations, data manifests, per-example metrics, aggregate reports, and exact commands. Update status and README to match actual support.

Do not spend the remaining budget on optional sweeps once these gates are sufficiently verified.

## 5. Suggested software boundaries

Choose simpler equivalents if they meet the contracts.

- Job/episode manifest: stable IDs, source checksums, profiles, instructions, cameras, requested stages.
- Native adapter: pinned loader, planning, parsing, stitching, and tensor-input audit.
- Frame store: source hash, camera, frame index, decode/color policy; bounded RAM and eviction.
- Worker: persistent model, explicit placement, batch invocation, valid-output checks.
- Scheduler: stage dependencies, batch compatibility, memory/token limits, backpressure, retries.
- Run store: atomic state transitions, immutable attempts, completion records, resume.
- Adaptive policy: named spatial presets, decision features, fallback, audit.
- Evaluation/reporting: metrics, cost accounting, uncertainty, and evidence export.

A single-host Python implementation with processes, bounded queues, and a small persistent manifest is sufficient. A distributed platform is not a prerequisite.

Group requests by model, profile, visual shape/token estimate, input length, and generation budget. Verify actual processed input lengths and measured memory. Correct padding, attention masks, video boundaries, positional inputs, per-request state, stop conditions, and output extraction matter.

Asynchronous completion must preserve episode, request, window, and segment identities. Existing temporal reconstruction must receive the required source bindings and ordering.

Decoded-frame reuse is the first caching boundary. Do not assume KV or vision embeddings are interchangeable across models, geometry, timestamps, or context. Preserve native color conversion and any transport transformations. Native masks and scheduler visibility must remain intact.

Only use GPU allocation IDs provided by SLURM. Child workers may narrow those devices; they must not expose physical GPUs outside the grant.

## 6. Dataset and evaluation protocol

Use lawful public data with existing independent references. WGO-Bench is one concrete candidate: 100 videos and 743 reference intervals. It has a single published split named train, which does not establish a project development/test split.

Before inspecting final outputs, commit manifests grouping related clips by episode and preferably task/scene. Use external development episodes with WGO held out when practical. If data/time constraints require a WGO pilot split, publish the split and do not present it as full benchmark replication. Respect profile, duration, and input limits; document exclusions before runs and report coverage.

Native correctness checks:
- Source frame IDs, timestamps, camera arrangement, frame preparation, prompt, and processor equivalence.
- Mixed dtype/value and model-residency checks.
- Batch-one versus batched output differences and parsed validity.
- Cross-window, overlapping-action, short-span, missing-context, and supported variable-rate cases.
- Duplicate submission, interrupted worker, retry ordering, and atomic completion.
- GPU-hour denominator calculations including failed and idle allocations.

Meaningful tests exercise actual risks. Mocks help scheduler development but do not replace GPU or semantic validation.

Independent annotation metrics:
- Segmentation F1 with declared one-to-one temporal matching.
- Caption correctness on fixed reference intervals, isolating the captioner.
- Joint temporal/semantic F1.
- Completion/coverage and error categories.

If reproducing FLORA's WGO scoring, follow its exact protocol, pinned rubric/judge, subset aggregation, and any scoring-only boundary normalization. Keep raw predictions unchanged.

Use available authorized evaluators within the zero-new-paid-API default. Existing reference captions are independent targets; exact string match alone is not factual equivalence. If a full reproducible judge is unavailable, provide a blinded, documented bounded review or deterministic subset and explicitly limit the claims. Agent judgments are AI evaluations, not human-created ground truth. Do not request new paid services or invent a successful quality gate.

Select a practical loss margin before testing. A suggested pilot target is at most one absolute percentage point on the chosen caption-quality metric; the orchestrator may choose a justified alternative before observing held-out results. Use paired episode-level uncertainty where sample size supports it. Distinguish a small observed change from evidence establishing the margin. Insufficient evidence means inconclusive.

## 7. Finite run matrix and fair accounting

Initial matrix:
A. Placement-adapted reference, weights local.
B. Persistent execution only.
C. Full optimized native engine.
D. Optimized fixed compact spatial budget.
E. Optional second fixed compact budget, selected during development.
F. Compact plus calibrated fallback.
G. Random fallback with approximately matched compute.

Do not run every matrix entry on the full corpus automatically. Use the measured pilot to choose the smallest comparisons that answer the questions. Start with at most 12 performance episodes. A 100-episode final evaluation is a ceiling contingent on projected cost fitting the remaining budget.

Report systems improvement A/B/C separately from adaptive improvement C/D/E/F/G. Use identical hardware allocation, placements, model revisions, corpus, and requested outputs for a direct comparison. Profile hardware differences separately if allocations differ.

Main metrics:
- Unique completed episode-hours per wall-clock hour.
- The same source hours per total allocated GPU-hours.
- Per-stage latency and end-to-end median/p95 under a stated arrival/queue condition.
- Peak per-device VRAM, host RAM, completion, retries, failures, and cache behavior.

Three synchronized cameras represent one episode duration; state the counting convention. GPU hours count every allocated device and idle time. Keep cold process start, first download/compile, warm inference, and cached final-result retrieval separate. Never count a replayed finished annotation as fresh model inference.

Use the job allocation record once; avoid counting both allocation and step records or both generic and typed GPU TRES. Retain application measurements because MaxRSS is host memory.

For adaptive results, report actual costs for rejected cases rather than multiplying fallback rate by an unrelated average. Include unchanged segmentation/decoding in end-to-end results. Charge any runtime audit and gate overhead.

## 8. Team and autonomy

The head agent owns interfaces and evidence. Suggested workers:
- Native model/placement and batch execution.
- Video preparation, scheduling, and restartability.
- Evaluation, accounting, and independent review.

Reassign based on the actual bottleneck. Freeze shared contracts before parallel implementation. A reviewer who did not implement a metric should inspect its denominator, split, and comparison.

Only the resource coordinator submits jobs. Agents write experiment requests with expected cost and outputs. Log every submission and completion. Diagnose failures using bounded hypotheses; do not repeat OOMs or queue duplicate runs. The transient-retry cap governs orchestration failures, not FLORA's existing native boundary-repair algorithm. Disable automatic requeue where supported and reconcile any restart before new model execution.

Make routine decisions autonomously. Where an actual access, policy, hardware, or budget blocker exists, finish independent work and state exactly what is unverified. Do not weaken the reference or claim full completion to avoid reporting a limitation.

## 9. Release acceptance and outcomes

Required evidence:
- A runnable CLI and documented environment.
- At least one real complete supported native-profile annotation, if hardware/access permits.
- A recorded native fidelity comparison.
- A measured native performance comparison on the declared subset.
- A bounded adaptive pilot with supported, unsupported, or inconclusive quality–cost conclusion.
- Reproducible manifests, sanitized per-example results, public report, limitations, and private full ledger.
- Reviewed commits pushed normally to the intended repository.
- Resource ledger reconciled and no unwanted task jobs left running or queued.

Use separate statuses for software implemented, real end-to-end validated, native fidelity validated, benchmark completed, and adaptive hypothesis resolved. If a required hardware gate cannot run, mark the release limited/hardware-blocked rather than complete at that gate.

Do not invent target speedups, generalize a single-view pass to all profiles, or label placement-adapted results as the original NVIDIA production system.

## 10. Primary references

- [Pinned upstream source](https://github.com/NVIDIA/Flora/tree/129a2ada74c526ee4692f3d1b111bea6ca19d5b7)
- [Annotation implementation](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/captioner/batch_utils/pipeline.py)
- [Model loader](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/captioner/batch_utils/models.py)
- [Media path](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/captioner/batch_utils/media.py)
- [Captioner model card](https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/captioner/README.md)
- [FLORA paper and evaluation protocol](https://arxiv.org/html/2610.31779v1)
- [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench)

