# Phase 2: a fast, usable FLORA annotation tool

## Objective and authority

Turn the existing baseline into a tool a researcher can use on a real video dataset: one clear command produces timestamped labels, an easy way to inspect and correct them, and substantially better processing efficiency.

The user considers the current release a useful but basic baseline. The requested contribution is meaningful acceleration and a practical research workflow. Accept the previous GPU runs as the development baseline. Further private-archive auditing is not the main task.

Own implementation, experiments, integration, and delivery. Use subagents proactively and exercise engineering judgment. Refactor internal scheduling or processing when justified; the existing architecture is a starting point. Keep changes reproducible and preserve the working v1 path.

Read AGENTS.md, the installed `$watcloud` skill, configs/resource_policy.yaml, docs/FINAL_REPORT.md, and the relevant source before acting. This brief defines the active phase and replaces phase-one sequencing and adaptive-experiment priorities. Earlier contracts, actual access restrictions, and project-wide resource limits remain applicable.

## What already exists

Release e51aaa723a5de9e5e532e178e4041a78e335f207 provides native TempSeg/Captioner execution, staged persistence, bounded decoded-frame caching, preparation overlap, checkpoints, resume, and benchmark tooling. Its modest speedup was measured on three short videos. Normal inference is batch one. The adaptive gate only detects empty or incomplete output.

Keep this release as the versioned baseline. Do not spend the phase re-establishing its entire audit trail. Reuse prepared models, data, environments, valid results, and the private resource ledger.

Two known weaknesses should be addressed as ordinary engineering work:
- Metadata reuse falls to zero above eight jobs under the current same-order stage passes and eight-entry LRU.
- The ordinary CLI's execution identity omits output-affecting runtime dependencies, although the benchmark/probe wrappers validate versions.

These repairs alone do not satisfy the acceleration objective.

## 1. Establish the bottleneck, then change the execution path

Begin with a small real profile. Attribute fresh-run time to model loading, decoding, image transformations/serialization, processor work, transfers, generation, synchronization, and output handling. Distinguish necessary computation from repeated preparation and optional instrumentation.

Choose the highest-leverage acceleration path from evidence. Candidate approaches include:

- Reusing validated source metadata and decoded/prepared content without the small-manifest cache cliff.
- Eliminating repeated decoding or lossless image serialization/readback while preserving the actual native pixels, timing, and required transformations.
- Overlapping useful per-request input preparation with GPU execution; the existing overlap is mostly episode validation/probing.
- Scheduling compatible caption requests across episodes and integrating true memory-admitted microbatching.
- A supported faster inference path, kernel, or backend if the exact model, precision, processor, and generation contract can be demonstrated.
- Moving redundant invariant checks out of repeated hot paths while retaining appropriate validation and explicit audit capability.

These are options, not a checklist. Select a coherent approach rather than implementing several speculative backends. Use available primary documentation and profiling tools when they resolve an actual implementation question. Run a small end-to-end compatibility check after a consequential change, then iterate.

Aim for a substantial gain over the current optimized v1 runner; use roughly 2x end-to-end throughput as an ambitious development target, not a promised result. Establish a realistic success criterion from the first profile before final measurements. A minor gain with extra documentation does not fulfill the intended acceleration goal.

Measure GPU-seconds per source-video minute as well as elapsed time. Additional GPUs must not masquerade as an algorithmic or efficiency improvement.

Preserve native temporal evidence, model values/precision, prompts, and stopping behavior for native comparisons. Changes to precision, model, spatial/frame budgets, or output behavior require a separately named and evaluated mode. Do not hide them in the native speedup.

A new batching strategy needs a fresh, justified admission test on representative input/output sizes and current hardware. Preserve the old failed calibration as history. Do not obtain admission merely by deleting a memory check or by testing only tiny generated outputs.

## 2. Make the usefulness visible

Deliver a short, coherent researcher workflow:

1. Install and check the required environment/hardware.
2. Point the tool at a folder or manifest of supported videos.
3. Run annotation with clear progress, bounded resource use, and actionable failures.
4. Open a lightweight local review view: play an action interval, see its caption, inspect failures, and make corrections if needed.
5. Export useful timestamped records and resume interrupted work with the same command.

Choose good command names and sensible defaults. Existing low-level commands may remain available. Add a thin user workflow around the functioning engine rather than requiring users to reconstruct research scripts.

Provide a quick demonstration that makes the value clear without first downloading every model. Use a permitted example or reproducible public-data download, with attribution. Label precomputed demonstration output explicitly. Keep private data private and respect FLORA's research license.

Export a documented machine-readable format such as JSONL, with source identity, intervals, generated labels, and failures. If review edits are supported, preserve the generated label and store the reviewed label/provenance separately. Add other export formats only when they serve a concrete use case.

Keep the result viewer small. Its purpose is to inspect actual outputs and show practical value. Performance engineering remains the primary work.

Rewrite the README around who this helps, a quick example, the actual acceleration mechanism, measured comparisons, hardware requirements, and limits. Describe implemented capabilities and planned work distinctly.

## 3. Validate on a broader workload

Treat the original three clips as development inputs. Freeze a broader evaluation manifest before inspecting final predictions: target approximately 12-20 complete episodes across multiple supported manipulation tasks, lengths, and visual complexity, with enough items to expose behavior beyond the eight-job cache capacity.

Choose total duration from measured pilot cost. The existing policy caps the initial performance pilot at 12 episodes; a larger frozen final workload must remain within its separate ceiling and the remaining aggregate budget. If the target corpus cannot fit, explain the reduced scope and preserve an honest generalization limit.

Compare at least:
- The previous optimized v1 release.
- The new optimized path.
- The upstream/placement-adapted reference when affordable and needed to establish compatibility.

Use identical source inputs, model settings, hardware/placement, and timing boundaries. Separate startup from sustained throughput. Use the same audit/instrumentation level in matched measurements; report optional audit overhead separately. Fresh inference must not reuse completed annotation results.

Repeat paired trials or counterbalance execution order within the bounded plan. Report the actual replication level, uncertainty where supported, coverage, failures, peak memory, full elapsed time, and allocated GPU cost. A microbenchmark may explain the mechanism but cannot replace the full-workflow result.

Validate the changes most likely to break correctness: timestamps, frames, prompts, request routing, memory pressure, and resume. Existing tests are useful; add focused tests for genuine new risks.

Bind traces to episode/request identities. Check compatibility on representative inputs. If outputs change, investigate and review the changed cases against the actual video before making quality claims. Do not equate agreement with v1 to visual correctness. The interface should make the remaining need for label review clear.

Do not tune on final outputs or discard slow/failing clips from the denominator. A new semantic-adaptive experiment is optional only after the useful native acceleration path is established.

## 4. Use compute efficiently

Use the installed `$watcloud` skill for remote operations. One coordinator owns submission, cancellation, retries, and accounting for all agents.

Continue the existing project ledger. The delivered report recorded about 8.773 of 24 GPU-hours and 11 of 16 GPU submissions used. About 15.23 GPU-hours and five submissions therefore remained at that observation; reconcile current state before planning. These are remaining allowances, not a new budget or a guarantee of availability.

Create a finite phase-two run plan from measured costs. Reserve fresh final reproduction first, then fit profiling, the selected optimization pilot, and broader comparison within the remaining GPU-hours, submission slots, job limits, CPU/storage limits, and actual scheduler availability. Do CPU implementation and preparation before occupying GPUs.

The old adaptive experiment's frozen 90-minute rule belongs to that experiment. Preserve its inconclusive result unchanged. Declare a new phase-two protocol with feasible run limits within the existing project-wide caps; do not inherit an obsolete experimental phase limit or silently revise the old result.

Routine implementation, project-local setup, bounded scheduled work, debugging, and normal commits/pushes are already authorized. Do not add recurring confirmation steps. Existing safeguards and genuine access/resource limits remain binding. If the budget becomes insufficient, finish useful affordable work and report the specific remaining requirement rather than resetting the ledger.

## 5. Delegate and deliver

Organize subagents around performance profiling/optimization, researcher workflow, and benchmark/correctness review. Adjust ownership to the work. You own interfaces, integration, resource decisions, and the final result. Use isolated worktrees or clear file ownership.

Maintain a short status record and commit reviewed changes normally. Preserve unrelated work, branch protections, and private operational data.

Give concise milestone updates: bottleneck identified, material optimization working, broader comparison complete, usable release delivered. Avoid routine tool narration and another long audit report; honor any mandatory host communication requirements.

Completion means:
- A researcher can use the documented workflow and understand the benefit quickly.
- A substantive execution-path improvement has been implemented and tested.
- Broader fresh-run evidence supports the claimed performance and compatibility.
- A clean reproduction and inspectable example work.
- The repository accurately explains the mechanism, results, supported configurations, and remaining limits.

If the desired acceleration cannot be achieved within hardware/budget constraints, say so plainly and identify the measured limiting factor. A functional workflow can still be delivered, but do not label the acceleration objective achieved merely because the documentation and tests are complete.

Final report: what a researcher can now do, what made it faster, how large and repeatable the measured benefit is, where it works, and the exact quickstart. Keep operational detail in the repository.

Start with the existing checkout and a bounded profiling/implementation plan, then execute it through delivery.
