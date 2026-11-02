# Rollout integration evidence

Version 0.3.0 connects recorded robot data to review and an existing offline-RL
trainer. It preserves the annotation engine and adds independent trajectory,
review-bundle, Rewarder, and training-export artifacts.

## What was executed

The official robomimic Lift proficient-human dataset supplies real state
observations, next observations, seven controller actions, environment rewards,
and simulator timestamps. Two demonstrations contain 117 transitions. Import and
export preserve all 46 source arrays exactly, including dtype. Original clocks,
source bootstrap masks, and per-episode identities also pass round-trip checks.
Unknown physical termination and truncation remain unknown. Source `dones`
combine success and the final stored sample; they are not fabricated terminals.
See [the data receipt](../reports/rl-data-roundtrip.json).

Pinned robomimic IQL completed 20 CPU optimizer updates. Actor, critic and value
parameters changed. Its checkpoint restored the same policy action and optimizer
state; one subsequent update succeeded. The policy then executed in two fresh
Lift simulator episodes, seeds 101 and 202, with 400 steps each. Success was
**0/2**. This is an integration smoke test, not learned competence or evidence of
policy improvement. The final warm-cache run used 7.98 seconds wall time on
macOS arm64 with two CPU threads. This is not a speed benchmark. All three local
trainer checks, including earlier checks, used 49.90 seconds wall and 24.22
seconds process CPU. See [the trainer receipt](../reports/rl-cpu-smoke.json).

A clean installed wheel can run the lightweight workflow without PyTorch. Its
export is readable by the actual robomimic `SequenceDataset`, including a batch
with action shape 16×1×7. See [the installed-package check](../reports/installed-package.json).

The real human-video example contains 15 native FLORA captions from the existing
release. Browser corrections preserve the original payload and full journal.
An actual browser download was reopened through the normal file picker; the
correction remained. CLI validation accepted that downloaded journal. Explicitly
selected local media passed SHA-256 verification and played through a Blob URL.
Source inspection and the static app's restricted fetch path show that own-file
media, names, labels, and corrections have no upload path.

The robot replay passed an independent CPU render check: 59 frames at 20 Hz,
256×256, with recorded source times preserved. OSMesa reported the llvmpipe
software renderer. This video shows saved simulator states, not fresh policy
execution. The public two-demo bundle and HDF5 retain all 117 original
transitions. Public copies omit machine-specific simulator XML; its original
hash is retained. Re-render from the pinned source dataset. See [replay evidence](../reports/rollout-replay.json)
and [public data checks](../reports/rollout-public-data.json).

Both scheduled CPU attempts are counted: failed environment setup used 723 s on
two CPUs; the corrected setup and replay used 645 s on two CPUs. Aggregate
CPU-only allocation use is 5.034444 core-hours. GPU totals remain 18.563889 hours
and 16 submissions, with no new GPU work. All jobs are reconciled. See
[resource closure](../reports/rollout-resources.json).

## Executed versus pending

| Capability | Evidence and limit |
| --- | --- |
| CPU trajectory and HDF5 workflow | Real data, exact round trip, actual trainer loader |
| IQL update/checkpoint/policy path | Real CPU execution; 0/2 fresh successes; smoke only |
| Native annotation | Historical native result retained; no new GPU inference in this phase |
| Rewarder companion | Real pinned CLI/input boundary and CPU parser/recovery tests; no native model execution |
| FLORA-derived training reward | Strict source-bound mapping and failure tests; genuine score execution remains pending |
| Robot FLORA captions | Unavailable for the new real robot example during the GPU pause |
| Learning comparison | BC versus environment-IQL versus genuine FLORA-IQL was not run |

## Preserved historical result

Phase two measured **1.573284×** throughput on one preselected six-video block,
with 121 captions and 170 native input records unchanged. The full eleven-video
criterion did not pass after the baseline exhausted memory in the other block.
This says nothing about Rewarder speed, robot-policy improvement, or production
reliability. [Historical scope and measurements](PHASE2_RESULTS.md) remain intact.

## Next native validation gate

GPU use remains paused, and all 16 submission slots are consumed. The
[one bounded Rewarder proposal](REWARDER.md#one-bounded-later-native-validation-proposal)
requires explicit changes to both restrictions and access to its stated hardware.
No model download, GPU job, training run, or paid compute is implied by the
browser or package installer. The current WATcloud inventory does not establish
access to an H100. No substitute placement is claimed to have passed.
