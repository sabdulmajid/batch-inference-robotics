"""Adversarial CPU fixtures for rollout/export contracts, not model evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = np = None

from bir.bundle import (attach, bundle_summary, load_bundle, make_bundle,
                        save_bundle, validate_bundle)
from bir.manifest import file_sha256
from bir.rewarder import prepare_request, run_terminal
from bir.trajectory import SOURCE_MASK, export_training, validate_trajectory
from bir.workflow import digest


class RolloutContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "fixture.mp4"
        self.video.write_bytes(b"Synthetic contract fixture, not native model input evidence")
        self.media = [dict(camera="front", filename="fixture.mp4", source_sha256=file_sha256(self.video), duration_s=.15)]
        self.task = "Lift the cube with the Panda robot."
        self.t = dict(schema_version="bir_trajectory_v1", episode_id="demo_0",
            source=dict(dataset="synthetic-contract-test", revision="test", sha256="a" * 64, demo_id="demo_0"),
            actions=[[.5, -.5], [.25, -.25], [1., -1.]],
            observations={"state": [[0., 1.], [1., 2.], [2., 3.]]},
            next_observations={"state": [[1., 2.], [2., 3.], [3., 4.]]},
            observation_times_s=[0., .05, .1], action_times_s=[0., .05, .1], next_observation_times_s=[.05, .1, .15],
            terminated=[None, None, None], truncated=[None, None, None],
            environment_rewards=[0., 0., 1.], source_bootstrap_stop=[0, 0, 1],
            simulator_states=[[0., 0.], [.05, 1.], [.1, 2.]],
            array_dtypes={"actions": "<f8", "rewards": "<f8", "dones": "<i8", "states": "<f8", "obs/state": "<f8", "next_obs/state": "<f8"},
            boundary=dict(kind="recording_end", source_mask_semantics=SOURCE_MASK, termination_evidence="Unknown in fixture"),
            action_spec=dict(representation="normalized test command", units=["normalized", "normalized"], bounds=[-1, 1], controller={"type": "fixture"}),
            environment_metadata={"env_name": "Lift", "type": 1, "env_kwargs": {"test_only": True}},
            clock_mappings=[dict(camera="front", video_sha256=self.media[0]["source_sha256"], scale=1., offset_s=0., source="explicit fixture shared clock")],
            source_attributes={"model_file": "<mujoco/>"})
        self.reseal()

    def reseal(self):
        t = self.t
        t["transition_ids"] = [digest([t["source"]["sha256"], t["source"]["demo_id"], i, a, b])
            for i, (a, b) in enumerate(zip(t["observation_times_s"], t["next_observation_times_s"]))]
        t["trajectory_id"] = digest({k: v for k, v in t.items() if k != "trajectory_id"})

    def episode(self, **changes):
        e = dict(episode_id="demo_0", task=self.task, media=copy.deepcopy(self.media), records=[], trajectory=copy.deepcopy(self.t), outcome=None)
        e.update(changes)
        return e

    def bundle(self, **changes):
        return make_bundle([self.episode(**changes)])

    def outcome(self, episode_id="demo_0", task=None):
        request = prepare_request(episode_id, task or self.task, [self.video], ["front"], whole_episode=True, clock_offsets_s=[0])
        def fixture(request, attempt):
            raw = dict(id=episode_id, task=task or self.task, score=4, expected_score=4., progress=.75,
                probabilities={str(i): float(i == 4) for i in range(1, 6)}, rationale=None,
                duration_s=.15, sampled_frames=2, output="Score: 4")
            (attempt / "native.jsonl").write_text(json.dumps(raw) + "\n")
            return {"fixture": True}
        return run_terminal(request, self.root / "scores", _test_executor=fixture)

    def test_valid_explicit_transition_contract_and_unknown_flags(self):
        self.assertEqual(validate_trajectory(self.t, episode_id="demo_0", media=self.media), 3)
        self.assertEqual(self.t["terminated"], [None] * 3)
        self.assertEqual(self.t["truncated"], [None] * 3)

    def test_disconnected_clocks_wrong_source_and_episode_links_are_rejected(self):
        for mutation in ["clock", "source", "episode", "flags"]:
            with self.subTest(mutation=mutation):
                t = copy.deepcopy(self.t)
                if mutation == "clock": t["next_observation_times_s"][0] = .04
                if mutation == "source": t["clock_mappings"][0]["video_sha256"] = "0" * 64
                if mutation == "episode": t["episode_id"] = "other"
                if mutation == "flags": t["terminated"][0] = True
                t["trajectory_id"] = digest({k: v for k, v in t.items() if k != "trajectory_id"})
                with self.assertRaises(ValueError):
                    validate_trajectory(t, episode_id="demo_0", media=self.media)

    def test_missing_next_observation_rows_are_not_fabricated(self):
        self.t["next_observations"]["state"].pop()
        self.reseal()
        with self.assertRaisesRegex(ValueError, "transition count"):
            self.bundle()

    def test_outcome_binding_rejects_other_episode_task_and_media(self):
        for outcome in [self.outcome(episode_id="demo_99"), self.outcome(task="Different task.")]:
            with self.assertRaisesRegex(ValueError, "do not match|binding|identity"):
                self.bundle(outcome=outcome)
        outcome = self.outcome()
        altered_media = copy.deepcopy(self.media)
        altered_media[0]["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "do not match|binding|identity"):
            self.bundle(trajectory=None, media=altered_media, outcome=outcome)

    def test_attachment_and_bundle_round_trip_preserve_original_and_history(self):
        record_id = "c" * 64
        edit = dict(schema_version="bir_review_v1", revision=1, record_id=record_id,
            generated_record_sha256=record_id, reviewed_caption="Historical correction.",
            action="caption_correction", reviewer="fixture-reviewer", reviewed_utc="2026-11-01T00:00:00Z", previous_entry_sha256=None)
        edit["entry_sha256"] = digest(edit)
        original = make_bundle([self.episode(trajectory=None)], reviews=[edit])
        original_copy = copy.deepcopy(original)
        attached = attach(original, episode_id="demo_0", trajectory=self.t, outcome=self.outcome())
        self.assertEqual(original, original_copy)
        self.assertEqual(attached["reviews"], [edit])
        path = self.root / "portable.json"
        save_bundle(attached, path)
        loaded = load_bundle(path)
        self.assertEqual(loaded, attached)
        self.assertEqual(bundle_summary(loaded)["historical_unapplied_reviews"], 1)
        self.assertEqual(bundle_summary(loaded)["native_reward_artifacts"], 0)
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_bundle(attached, path)
        self.assertEqual(load_bundle(path), attached)

    def test_simulator_state_length_is_rejected(self):
        self.t["simulator_states"].pop()
        self.reseal()
        with self.assertRaises(ValueError):
            self.bundle()

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_exact_arrays_clocks_and_source_mask_round_trip_without_fake_terminals(self):
        # This checks format semantics; real-source checks are a separate integration run.
        bundle = self.bundle()
        path = self.root / "train.hdf5"
        receipt = export_training(bundle, path, reward_source="environment", boundary_mode="source-mask")
        self.assertEqual(receipt["transitions"], 3)
        with h5py.File(path, "r") as f:
            g = f["data/demo_0"]
            for path, values in [("actions", self.t["actions"]), ("rewards", self.t["environment_rewards"]),
                                 ("obs/state", self.t["observations"]["state"]), ("next_obs/state", self.t["next_observations"]["state"]),
                                 ("bir/observation_times_s", self.t["observation_times_s"]),
                                 ("bir/action_times_s", self.t["action_times_s"]), ("bir/next_observation_times_s", self.t["next_observation_times_s"])]:
                np.testing.assert_array_equal(g[path][()], values)
            np.testing.assert_array_equal(g["dones"][()], [0, 0, 1])
            np.testing.assert_array_equal(g["bir/terminated"][()], [-1, -1, -1])
            np.testing.assert_array_equal(g["bir/truncated"][()], [-1, -1, -1])
            self.assertEqual(g["actions"].dtype.str, "<f8")
            self.assertEqual(g["obs/state"].dtype.str, "<f8")
            self.assertEqual(json.loads(f.attrs["bir_provenance"])["boundary_mode"], "source-mask")

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_declared_source_float32_is_preserved_exactly(self):
        for key in ["observations", "next_observations"]:
            self.t[key]["state"] = np.asarray(self.t[key]["state"], dtype="float32").tolist()
        self.t["array_dtypes"].update({"obs/state": "<f4", "next_obs/state": "<f4"})
        self.reseal()
        path = self.root / "float32.hdf5"
        export_training(self.bundle(), path, reward_source="environment", boundary_mode="source-mask")
        with h5py.File(path, "r") as f:
            self.assertEqual(f["data/demo_0/obs/state"].dtype.str, "<f4")
            np.testing.assert_array_equal(f["data/demo_0/obs/state"][()], self.t["observations"]["state"])

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_lossy_dtype_override_cannot_change_actions_or_observations(self):
        for dtype_key in ["actions", "obs/state"]:
            t = copy.deepcopy(self.t)
            t["array_dtypes"][dtype_key] = "|i1"
            if dtype_key == "obs/state": t["observations"]["state"][0][0] = .5
            t["trajectory_id"] = digest({k: v for k, v in t.items() if k != "trajectory_id"})
            with self.subTest(dtype_key=dtype_key), self.assertRaises(ValueError):
                export_training(self.bundle(trajectory=t), self.root / (dtype_key.replace("/", "-") + ".hdf5"), reward_source="environment", boundary_mode="source-mask")

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_default_dtype_cannot_round_a_recorded_large_integer(self):
        self.t["array_dtypes"].pop("obs/state")
        self.t["observations"]["state"][0][0] = 9007199254740993
        self.reseal()
        with self.assertRaisesRegex(ValueError, "change a recorded value|lossy|dtype"):
            export_training(self.bundle(), self.root / "default-loss.hdf5", reward_source="environment", boundary_mode="source-mask")

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_source_attributes_cannot_replace_export_identity_or_transition_count(self):
        for key, value in [("num_samples", "999"), ("bir_episode_id", "other"), ("bir_trajectory_id", "0" * 64)]:
            t = copy.deepcopy(self.t)
            t["source_attributes"][key] = value
            t["trajectory_id"] = digest({k: v for k, v in t.items() if k != "trajectory_id"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                export_training(self.bundle(trajectory=t), self.root / (key + ".hdf5"), reward_source="environment", boundary_mode="source-mask")

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_unknown_boundary_cannot_be_reinterpreted_as_true_termination(self):
        with self.assertRaisesRegex(ValueError, "Unknown termination"):
            export_training(self.bundle(), self.root / "bad.hdf5", reward_source="environment", boundary_mode="terminated")
        with self.assertRaisesRegex(ValueError, "evidenced true episode ending"):
            export_training(self.bundle(outcome=self.outcome()), self.root / "bad-flora.hdf5", reward_source="flora-terminal", boundary_mode="source-mask")

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_true_truncation_keeps_bootstrap_and_recorded_next_observation(self):
        self.t["terminated"] = [False, False, False]
        self.t["truncated"] = [False, False, True]
        self.t["boundary"]["kind"] = "time_limit"
        self.reseal()
        path = self.root / "truncated.hdf5"
        export_training(self.bundle(), path, reward_source="environment", boundary_mode="terminated")
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["data/demo_0/dones"][()], [0, 0, 0])
            np.testing.assert_array_equal(f["data/demo_0/next_obs/state"][()], self.t["next_observations"]["state"])

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_explicit_reward_source_missing_rewards_and_duplicate_export_are_enforced(self):
        for source in [None, "default", "mixed"]:
            with self.assertRaisesRegex(ValueError, "Explicit supported"):
                export_training(self.bundle(), self.root / "bad.hdf5", reward_source=source, boundary_mode="source-mask")
        self.t["environment_rewards"] = None
        self.reseal()
        with self.assertRaisesRegex(ValueError, "rewards are absent"):
            export_training(self.bundle(), self.root / "missing.hdf5", reward_source="environment", boundary_mode="source-mask")
        self.t["environment_rewards"] = [0., 0., 1.]
        self.reseal()
        path = self.root / "once.hdf5"
        export_training(self.bundle(), path, reward_source="environment", boundary_mode="source-mask")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_training(self.bundle(), path, reward_source="environment", boundary_mode="source-mask")
        self.assertEqual(path.read_bytes(), before)

    @unittest.skipUnless(h5py is not None, "Training export tests require the data extra")
    def test_video_only_episode_exclusion_is_recorded_in_training_membership(self):
        extra = self.episode(episode_id="video-only", trajectory=None, media=[], records=[])
        bundle = make_bundle([self.episode(), extra])
        receipt = export_training(bundle, self.root / "membership.hdf5", reward_source="environment", boundary_mode="source-mask")
        self.assertEqual([x["episode_id"] for x in receipt["included"]], ["demo_0"])
        self.assertEqual(receipt["exclusions"], [dict(episode_id="video-only", reason="No recorded robot transitions")])


if __name__ == "__main__":
    unittest.main()
