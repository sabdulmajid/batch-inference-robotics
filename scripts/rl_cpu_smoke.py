#!/usr/bin/env python3
"""Run real robomimic IQL updates, checkpoint recovery, and fresh CPU rollouts.

This is a small integration test. It does not establish learned competence.
Only load checkpoints produced by this command: PyTorch checkpoints use pickle.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import time


OBSERVATION_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
REVISIONS = {
    "robomimic": "d309eaecc18acf4152a830a895a6984b8ac71b05",
    "robosuite": "a071383d53568ab798eb315c0e95357911be922d",
}
SOURCE_HASHES = {
    "algo/iql.py": "c0d131eb328fd88ae7d86f68bf005c8788020244e6a0b2568eee424b2a680045",
    "algo/algo.py": "a94ac3f6cc3c4242fcc0d15979f2a9fa38a7cf90be9a69b1a085a0117a2b013c",
    "utils/dataset.py": "603bde0f2b15b7bf7eac0bed128df984fae38682572e7c907dbafc935b7a62f5",
    "envs/env_robosuite.py": "efa1b2e47861de966784192c2c7216943c74874588ea720d8e1e89e6597cf8ff",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_int(value: str, *, minimum: int, maximum: int) -> int:
    result = int(value)
    if not minimum <= result <= maximum:
        raise argparse.ArgumentTypeError(f"must be in [{minimum}, {maximum}]")
    return result


def state_equal(left, right) -> bool:
    """Compare nested optimizer state without casting or discarding tensors."""
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            state_equal(a, b) for a, b in zip(left, right)
        )
    if hasattr(left, "detach"):
        import torch
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    return left == right


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict:
    start = time.monotonic()
    cpu_start = time.process_time()
    args.output.mkdir(parents=True, exist_ok=False)
    # Set these before importing native libraries. No CUDA or MPS selection exists.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache = args.output.resolve().parent / ".cpu-trainer-cache"
    cache.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["HF_HOME"] = str(cache / "huggingface")

    def timeout(_signum, _frame):
        raise TimeoutError(f"CPU smoke exceeded {args.max_wall_seconds} seconds")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.max_wall_seconds)
    report = {"schema_version": "bir_iql_smoke_result_v1", "status": "running", "device": "cpu"}
    try:
        import h5py
        import numpy as np
        import robomimic
        import torch
        from robomimic.algo import RolloutPolicy, algo_factory
        from robomimic.config import config_factory
        from robomimic.utils import env_utils as EnvUtils
        from robomimic.utils import file_utils as FileUtils
        from robomimic.utils import obs_utils as ObsUtils
        from robomimic.utils import train_utils as TrainUtils
        from robomimic.utils.dataset import SequenceDataset

        source_hashes = {name: sha256_file(Path(robomimic.__file__).parent / name) for name in SOURCE_HASHES}
        if source_hashes != SOURCE_HASHES:
            raise ValueError("installed IQL, loader, checkpoint, or environment adapter differs from the pinned source")
        for package, expected in {"robomimic": "0.5.0", "robosuite": "1.5.1", "mujoco": "3.2.7"}.items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"unsupported {package} version; install the pinned CPU requirements")

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cpu")
        config = config_factory("iql")
        config.train.cuda = False
        config.train.seed = args.seed
        config.train.batch_size = args.batch_size
        config.train.num_data_workers = 0
        config.train.num_epochs = 1
        config.train.hdf5_use_swmr = False
        config.train.hdf5_normalize_obs = False
        config.experiment.name = "bir-iql-cpu-smoke"
        config.experiment.epoch_every_n_steps = args.steps
        config.experiment.render_video = False
        config.experiment.logging.log_tb = False
        config.experiment.rollout.n = args.rollouts
        config.experiment.rollout.horizon = args.horizon
        config.experiment.rollout.terminate_on_success = False
        config.observation.modalities.obs.low_dim = list(OBSERVATION_KEYS)
        config.observation.modalities.obs.rgb = []
        config.algo.actor.layer_dims = (64, 64)
        config.algo.critic.layer_dims = (64, 64)
        config.lock()
        ObsUtils.initialize_obs_utils_with_config(config)
        shape = FileUtils.get_shape_metadata_from_dataset(
            {"path": str(args.dataset)}, action_keys=["actions"], all_obs_keys=list(OBSERVATION_KEYS)
        )
        if shape["ac_dim"] != 7 or dict(shape["all_shapes"]) != {
            "object": [10], "robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2]
        }:
            raise ValueError("unsupported observation or action shape for the Lift CPU profile")
        metadata = FileUtils.get_env_metadata_from_dataset(str(args.dataset))
        if metadata.get("env_name") != "Lift" or metadata.get("env_version") != "1.5.1":
            raise ValueError("this tested CPU profile requires robosuite 1.5.1 Lift metadata")
        with h5py.File(args.dataset, "r") as handle:
            demos = sorted(handle["data"], key=lambda key: int(key.split("_")[-1]))
            if not demos or len(demos) > 20:
                raise ValueError("CPU smoke requires 1 to 20 explicitly exported demonstrations")
            transitions = sum(len(handle[f"data/{demo}/actions"]) for demo in demos)
            if transitions > 10000:
                raise ValueError("CPU smoke is limited to 10000 transitions")
            first_obs = {key: handle[f"data/{demos[0]}/obs/{key}"][0] for key in OBSERVATION_KEYS}
        dataset = SequenceDataset(
            hdf5_path=str(args.dataset), obs_keys=OBSERVATION_KEYS,
            action_keys=["actions"], dataset_keys=["actions", "rewards", "dones"],
            action_config=config.train.action_config, seq_length=1, frame_stack=1,
            hdf5_cache_mode="all", hdf5_use_swmr=False, load_next_obs=True,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        iterator = iter(loader)
        model = algo_factory("iql", config, obs_key_shapes=shape["all_shapes"], ac_dim=shape["ac_dim"], device=device)
        initial = {name: {key: value.detach().clone() for key, value in model.nets[name].state_dict().items()}
                   for name in ("actor", "critic", "vf")}
        losses = []

        def update(current_model):
            nonlocal iterator
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = current_model.process_batch_for_training(raw)
            batch = current_model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
            current_model.set_train()
            info = current_model.train_on_batch(batch, epoch=1, validate=False)
            current_model.on_gradient_step()
            metrics = current_model.log_info(info)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("non-finite IQL training metric")
            return {key: float(value) for key, value in metrics.items() if "loss" in key.lower()}

        for _ in range(args.steps):
            losses.append(update(model))
        changed = {name: any(not torch.equal(before, model.nets[name].state_dict()[key])
                             for key, before in initial[name].items()) for name in initial}
        if not all(changed.values()):
            raise ValueError(f"IQL network did not update: {changed}")
        model.set_eval()
        original_policy = RolloutPolicy(model, action_normalization_stats=dataset.get_action_normalization_stats())
        original_policy.start_episode()
        with torch.no_grad():
            action_before = original_policy(first_obs)
        checkpoint = args.output / "iql-checkpoint.pth"
        TrainUtils.save_model(model, config, metadata, shape, str(checkpoint),
                              variable_state={"gradient_steps": args.steps},
                              action_normalization_stats=dataset.get_action_normalization_stats())
        # This file was produced by this process. Do not use this on untrusted pickle input.
        reloaded, checkpoint_data = FileUtils.policy_from_checkpoint(device=device, ckpt_path=str(checkpoint))
        reloaded.policy.deserialize(checkpoint_data["model"], load_optimizers=True)
        restored_optimizers = state_equal(model.serialize()["optimizers"], reloaded.policy.serialize()["optimizers"])
        reloaded.start_episode()
        with torch.no_grad():
            action_after = reloaded(first_obs)
        if not np.array_equal(action_before, action_after) or not restored_optimizers:
            raise ValueError("checkpoint changed the policy action or optimizer state")
        reloaded_state = copy.deepcopy(reloaded.policy.serialize()["nets"])
        resumed_loss = update(reloaded.policy)
        resumed_changed = any(not torch.equal(value, reloaded.policy.nets.state_dict()[key])
                              for key, value in reloaded_state.items())
        if not resumed_changed:
            raise ValueError("reloaded IQL did not perform the subsequent optimizer update")
        TrainUtils.save_model(reloaded.policy, config, metadata, shape, str(args.output / "iql-resumed.pth"),
                              variable_state={"gradient_steps": args.steps + 1},
                              action_normalization_stats=dataset.get_action_normalization_stats())
        # Evaluate the original saved checkpoint. Do not use saved demonstration states.
        policy, _ = FileUtils.policy_from_checkpoint(device=device, ckpt_path=str(checkpoint))
        evaluation = []
        for seed in (101, 202)[:args.rollouts]:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            env = EnvUtils.create_env_from_metadata(copy.deepcopy(metadata), render=False,
                                                   render_offscreen=False, use_image_obs=False)
            try:
                observation = env.reset()
                policy.start_episode()
                success = False
                total_reward = 0.0
                for step in range(args.horizon):
                    with torch.no_grad():
                        action = policy({key: observation[key] for key in OBSERVATION_KEYS})
                    if not np.isfinite(action).all():
                        raise ValueError("policy produced a non-finite action")
                    observation, reward, done, _ = env.step(action)
                    total_reward += float(reward)
                    success = success or bool(env.is_success()["task"])
                    if done:
                        break
                evaluation.append({"seed": seed, "steps": step + 1, "task_success": success,
                                   "environment_reward_sum": total_reward,
                                   "boundary": "environment_done" if done else "evaluation_time_limit",
                                   "reset": "fresh_simulator_reset", "uses_recorded_states": False})
            finally:
                env.env.close()
        report.update({
            "status": "passed", "algorithm": "robomimic.IQL", "dataset_sha256": sha256_file(args.dataset),
            "dataset_membership": demos, "transitions": transitions, "seed": args.seed,
            "optimizer_updates": args.steps, "resumed_optimizer_updates": 1,
            "network_parameters_changed": changed, "checkpoint_policy_action_exact": True,
            "checkpoint_optimizer_state_exact": restored_optimizers, "resumed_network_changed": resumed_changed,
            "checkpoint_sha256": sha256_file(checkpoint), "losses": losses, "resumed_losses": resumed_loss,
            "evaluation": evaluation, "success_count": sum(item["task_success"] for item in evaluation),
            "evaluation_count": len(evaluation), "evaluation_horizon": args.horizon,
            "observation_keys": list(OBSERVATION_KEYS), "uses_retrospective_policy_observations": False,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("robomimic", "robosuite", "torch", "torchvision", "mujoco", "numpy", "h5py")},
            "intended_source_revisions": REVISIONS,
            "verified_robomimic_source_sha256": source_hashes,
            "evidence_scope": "Optimizer, loader, checkpoint and fresh policy execution smoke; not policy improvement.",
        })
        (args.output / "config.json").write_text(config.dump() + "\n")
        dataset.close_and_delete_hdf5_handle()
    except BaseException as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        report["wall_seconds"] = time.monotonic() - start
        report["process_cpu_seconds"] = time.process_time() - cpu_start
        _write_json(args.output / "result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing results are never replaced")
    parser.add_argument("--steps", type=lambda value: bounded_int(value, minimum=1, maximum=100), default=20)
    parser.add_argument("--rollouts", type=lambda value: bounded_int(value, minimum=0, maximum=2), default=2)
    parser.add_argument("--horizon", type=lambda value: bounded_int(value, minimum=1, maximum=400), default=400)
    parser.add_argument("--threads", type=lambda value: bounded_int(value, minimum=1, maximum=2), default=2)
    parser.add_argument("--batch-size", type=lambda value: bounded_int(value, minimum=1, maximum=64), default=16)
    parser.add_argument("--max-wall-seconds", type=lambda value: bounded_int(value, minimum=1, maximum=600), default=600)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("status", "optimizer_updates", "evaluation_count", "success_count", "wall_seconds")}))


if __name__ == "__main__":
    main()
