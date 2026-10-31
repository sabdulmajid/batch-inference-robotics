# Batch Inference for Robotics

A research project to execute NVIDIA FLORA annotation efficiently, then test whether selective visual-compute allocation improves the quality–cost trade-off.

**Status: planning pack only. No project GPU runs, speedups, or quality results have been established.**

## Start here

1. Read [AGENTS.md](AGENTS.md).
2. Give the head agent [the orchestrator brief](docs/ORCHESTRATOR_PROMPT.md), together with the private connection details from the launch message.
3. Follow [the project plan](docs/PROJECT_PLAN.md), [WATcloud runbook](docs/WATCLOUD_RUNBOOK.md), and [resource policy](configs/resource_policy.yaml).
4. Keep [STATUS.md](docs/STATUS.md) current with evidence and remaining work.

The local launch message supplies machine-specific SSH settings. Store those in ignored local configuration, not this public repository.

## Intended outcome

- A CLI accepting video jobs and producing timestamped FLORA annotations.
- A faithful, hardware-compatible reference path and an optimized native path.
- Persistent execution, bounded video decoding/caching, and compatible request batching where measured memory permits.
- An experimental compact/native caption router evaluated against fixed budgets.
- Reproducible run manifests, per-example measurements, correctness checks, and an honest final report.
- A small local annotation/result viewer only after the inference and evaluation gates work.

No model training, physical robot, public hosted service, or multi-node inference is required for the first release.

## Hardware reality

The supplied cluster inventory consists of 24 GiB or smaller GPUs. A 27B model's BF16 parameter storage alone is approximately 50.3 GiB; one or two 24 GiB cards cannot hold it faithfully. Three or four cards on one node are candidates, subject to actual topology, scheduler availability, placement, and measured runtime memory.

The reference Captioner loader also reserves 12 GiB per visible GPU. A documented placement-only adaptation is therefore part of the feasibility work. It must preserve checkpoint identity, numerical precision, preprocessing, and generation behavior.

Start with one-GPU TempSeg feasibility, then prove Captioner placement and a small complete annotation run. Initially process the stages in phases rather than assuming simultaneous model residency.

## Scope and honesty

The primary deliverable is an optimized engine using native inference inputs. The adaptive experiment may produce a positive, negative, or inconclusive result. A hardware limitation is a reported limitation, never a fabricated passing result.

The conservative defaults authorize one project GPU allocation at a time, up to four same-node GPUs only when justified for faithful model placement, and 24 aggregate allocated GPU-hours. These are project defaults, not WATcloud policy. See the resource policy for all limits.

FLORA code and weights carry NVIDIA's noncommercial research/evaluation terms. Preserve upstream notices and keep dependencies and redistribution rights explicit. Model weights, datasets, private configuration, and raw infrastructure logs do not belong in Git.

## Sources

The planning snapshot inspected FLORA commit **129a2ada74c526ee4692f3d1b111bea6ca19d5b7** on 2026-10-31. Revalidate source behavior when implementing; record the exact revision used.

- [FLORA repository](https://github.com/NVIDIA/Flora)
- [FLORA paper](https://arxiv.org/html/2610.31779v1)
- [WATcloud SLURM](https://cloud.watonomous.ca/docs/compute-cluster/slurm)
- [WATcloud user manual](https://cloud.watonomous.ca/docs/compute-cluster/user-manual)
- [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench)

