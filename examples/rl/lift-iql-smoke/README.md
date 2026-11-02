# CPU IQL smoke artifacts

This checkpoint comes from 20 real robomimic IQL updates on 117 official Lift transitions. It is an integration test, not a competent policy. Fresh simulator success was 0/2 over 400 steps each. The resumed checkpoint contains one additional update after exact optimizer recovery.

The source dataset declares MIT licensing. Attribution: robomimic contributors, Stanford Vision and Learning Lab; Lift proficient-human demonstrations, robosuite 1.5.1. See [the trainer guide](../../../docs/TRAINER_INTEGRATION.md) for source hashes, limits, and commands. The accompanying metadata is [the measured receipt](../../../reports/rl-cpu-smoke.json).

Only load trusted PyTorch checkpoints. They use pickle serialization. The supplied small checkpoints were produced by this project's CPU smoke script.
