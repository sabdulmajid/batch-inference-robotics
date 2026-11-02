# Public research examples

## Human action video and labels

`human.mp4` is the unchanged `homer_41` video from MacroData's WGO-Bench,
revision `9cfb7bb98d60a58319f4b3d7b41374eedfb7980c`. Its dataset card declares
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Source: https://huggingface.co/datasets/macrodata/WGO-Bench/tree/9cfb7bb98d60a58319f4b3d7b41374eedfb7980c
Credit: MacroData, WGO-Bench, original HomER dataset contributors.
The source SHA and complete model provenance are in `human.bir.json`.
The captions are genuine precomputed NVIDIA FLORA research outputs, not human
truth. FLORA's model use is subject to its non-commercial research license:
https://github.com/NVIDIA/Flora/blob/129a2ada74c526ee4692f3d1b111bea6ca19d5b7/LICENSE
This demo and its human example are offered for non-commercial research only.
Corrections remain separate from original predictions. Retain attribution and
share-alike conditions when redistributing covered media or adaptations.

## Robot trajectory replay

`robot.mp4` renders recorded simulator states from the official robomimic
Lift proficient-human dataset. It is replay, not policy execution. Native
FLORA captions and Rewarder results are unavailable for this example.
Source: https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/74fa018461f479cd9fd15b924a16103012096203
The dataset card declares MIT licensing. Credit: robomimic and robosuite
contributors. This repository retains source revision, SHA, controller metadata,
and recorded timestamps. No robot actions were invented for the human video.
Upstream code/license: https://github.com/ARISE-Initiative/robomimic/blob/d309eaecc18acf4152a830a895a6984b8ac71b05/LICENSE
Simulator/code/license: https://github.com/ARISE-Initiative/robosuite/blob/a071383d53568ab798eb315c0e95357911be922d/LICENSE
Robot assets retain their upstream notices. This release makes no claim of
universal or commercial clearance.
