#!/usr/bin/env python3
"""Fetch a pinned real robot demonstration and render saved states on CPU.

Rendering requires Linux and OSMesa. It is trajectory replay, not policy execution.
Use --download-only for the lightweight data step on other systems.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import urllib.request


DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
DATASET_SHA256 = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
DATASET_BYTES = 21084088
DATASET_PATH = "v1.5/lift/ph/low_dim_v15.hdf5"
DATASET_URL = f"https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/{DATASET_REVISION}/{DATASET_PATH}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_dataset(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        total = 0
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=60) as response, temporary.open("xb") as output:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > DATASET_BYTES:
                        raise ValueError("dataset download exceeded pinned byte count")
                    output.write(block)
            if total != DATASET_BYTES or sha256_file(temporary) != DATASET_SHA256:
                raise ValueError("dataset download does not match the pinned digest and size")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != DATASET_BYTES or sha256_file(path) != DATASET_SHA256:
        raise ValueError("existing dataset does not match the supported source revision")


def validate_clock(times, control_freq: float) -> None:
    import numpy as np
    values = np.asarray(times, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) < 2 or control_freq <= 0:
        raise ValueError("invalid recorded clock or controller rate")
    if not np.allclose(np.diff(values), 1.0 / control_freq, atol=1e-9, rtol=0):
        raise ValueError("recorded state times do not match the source controller rate")


def render_example(dataset: Path, output: Path, demo_id: str = "demo_0") -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("CPU replay requires Linux OSMesa; macOS accelerated graphics are not selected")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"
    start = time.monotonic()
    cpu_start = time.process_time()
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from OpenGL import GL
    from robomimic.utils import env_utils as EnvUtils
    from robomimic.utils import obs_utils as ObsUtils

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": [
        "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"
    ]})
    with h5py.File(dataset, "r") as source:
        metadata = json.loads(source["data"].attrs["env_args"])
        demo = source[f"data/{demo_id}"]
        states = demo["states"][:]
        if len(states) > 100:
            raise ValueError("example replay is limited to 100 recorded states")
        times = states[:, 0]
        frequency = float(metadata["env_kwargs"]["control_freq"])
        validate_clock(times, frequency)
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        ep_meta = demo.attrs.get("ep_meta", None)
    env = EnvUtils.create_env_from_metadata(copy.deepcopy(metadata), render=False,
                                           render_offscreen=True, use_image_obs=False)
    video = output / "replay.mp4"
    try:
        env.reset_to({"states": states[0], "model": model_xml, "ep_meta": ep_meta})
        renderer = GL.glGetString(GL.GL_RENDERER).decode("utf-8")
        # OSMesa supplies the software OpenGL context; preserve its actual renderer.
        if not any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast", "software")):
            raise RuntimeError(f"expected a software renderer, received {renderer!r}")
        with imageio.get_writer(video, fps=frequency, codec="libx264", quality=8,
                                output_params=["-threads", "2", "-pix_fmt", "yuv420p"]) as writer:
            for state in states:
                env.reset_to({"states": state})
                image = env.render(mode="rgb_array", height=256, width=256, camera_name="agentview")
                if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                    raise ValueError("unexpected simulator render shape or dtype")
                writer.append_data(image)
    finally:
        env.env.close()
    result = {
        "schema_version": "bir_rollout_replay_v1",
        "dataset": "robomimic/robomimic_datasets", "dataset_revision": DATASET_REVISION,
        "dataset_path": DATASET_PATH, "dataset_sha256": DATASET_SHA256, "demo_id": demo_id,
        "video": video.name, "video_sha256": sha256_file(video), "camera": "agentview",
        "rendered_frame_times_s": times.tolist(), "video_fps": frequency,
        "frame_count": len(states), "duration_s": len(states) / frequency,
        "clock_mapping": {"scale": 1.0, "offset_s": -float(times[0]),
                          "source": "recorded MuJoCo times and source controller rate"},
        "replay_not_policy_execution": True, "renderer": renderer, "render_backend": "osmesa",
        "environment_version": metadata["env_version"], "render_dimensions": [256, 256],
        "wall_seconds": time.monotonic() - start, "process_cpu_seconds": time.process_time() - cpu_start,
        "license": "MIT (upstream dataset card)",
        "task": "Lift the cube with the Panda robot.",
        "annotation_status": "unavailable: this replay has not been processed by FLORA",
        "rewarder_status": "unavailable: native Rewarder execution is paused",
    }
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo", choices=("demo_0", "demo_1"), default="demo_0")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not args.download_only and args.output is None:
        parser.error("--output is required for replay")
    fetch_dataset(args.dataset)
    if args.download_only:
        print(json.dumps({"dataset_sha256": DATASET_SHA256, "bytes": DATASET_BYTES, "status": "verified"}))
    else:
        result = render_example(args.dataset, args.output, args.demo)
        print(json.dumps({key: result[key] for key in ("demo_id", "frame_count", "video_sha256", "renderer")}))


if __name__ == "__main__":
    main()
