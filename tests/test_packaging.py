"""Regressions found during the independent installed-wheel workflow check."""
import copy

import pytest

from bir.bundle import make_bundle
from bir.trajectory import SOURCE_MASK, export_training, validate_trajectory
from bir.workflow import digest


def source_attribute_fixture():
    """One synthetic transition with the real adapter's attribute names."""
    value = dict(
        schema_version="bir_trajectory_v1", episode_id="demo_0",
        source=dict(dataset="synthetic-test", revision="fixture", sha256="a" * 64, demo_id="demo_0"),
        observations={"state": [[0.0]]}, next_observations={"state": [[1.0]]}, actions=[[0.5]],
        observation_times_s=[0.0], action_times_s=[0.0], next_observation_times_s=[0.05],
        terminated=[None], truncated=[None], environment_rewards=[1.0], source_bootstrap_stop=[1],
        boundary=dict(kind="recording_end", source_mask_semantics=SOURCE_MASK),
        action_spec=dict(representation="normalized fixture", units=["normalized"], bounds=[-1, 1], controller={}),
        environment_metadata=dict(env_name="Lift", type=1, env_kwargs={"test_only": True}),
        clock_mappings=[], source_attributes=dict(model_file="<mujoco/>", camera_info="{}", ep_meta="{}"),
    )
    value["transition_ids"] = [digest([value["source"]["sha256"], "demo_0", 0, 0.0, 0.05])]
    value["trajectory_id"] = digest(value)
    return value


def test_real_camera_info_attribute_is_accepted_and_preserved_in_export(tmp_path):
    h5py = pytest.importorskip("h5py")
    trajectory = source_attribute_fixture()
    assert validate_trajectory(trajectory, episode_id="demo_0") == 1
    bundle = make_bundle([dict(episode_id="demo_0", task="Synthetic fixture only.", media=[], records=[],
                               trajectory=trajectory, outcome=None)])
    output = tmp_path / "export.hdf5"
    export_training(bundle, output, reward_source="environment", boundary_mode="source-mask")
    with h5py.File(output, "r") as handle:
        assert handle["data/demo_0"].attrs["camera_info"] == "{}"
        assert handle["data/demo_0"].attrs["ep_meta"] == "{}"
        assert handle["data/demo_0"].attrs["model_file"] == "<mujoco/>"


def test_source_metadata_cannot_override_export_provenance():
    trajectory = source_attribute_fixture()
    for reserved in ("bir_metadata", "bir_trajectory_id", "num_samples"):
        altered = copy.deepcopy(trajectory)
        altered["source_attributes"][reserved] = "not permitted"
        altered["trajectory_id"] = digest({key: value for key, value in altered.items() if key != "trajectory_id"})
        with pytest.raises(ValueError, match="source attribute"):
            validate_trajectory(altered)
