import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from bir.manifest import file_sha256, load_manifest
from scripts.prepare_wgo import DatasetPin, assign_groups, embedded_video, inspect_parquet, prepare_parquet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/Flora"
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("av", "numpy", "PIL", "pyarrow"))


class WgoLogicTests(unittest.TestCase):
    def test_embedded_bytes_do_not_follow_dataset_paths(self):
        self.assertEqual(embedded_video({"bytes": b"video", "path": "/never/read/me"}), b"video")
        with self.assertRaises(ValueError):
            embedded_video({"bytes": None, "path": "/never/read/me"})

    def test_task_scene_and_duplicate_video_links_keep_connected_groups_together(self):
        records = [
            dict(id="homer_a", task_hash="task1", source_family="homer", metadata={"scene_id": "one"}),
            dict(id="homer_b", task_hash="task1", source_family="homer", metadata={"scene_id": "two"}),
            dict(id="homer_c", task_hash="task2", source_family="homer", metadata={"scene_id": "two"}, source_sha256="video"),
            dict(id="homer_d", task_hash="task3", source_family="homer", metadata={}, source_sha256="video"),
        ]
        assign_groups(records)
        self.assertEqual(len({r["group_id"] for r in records}), 1)
        original = {r["id"]: r["proposed_split"] for r in records}
        assign_groups(list(reversed(records)))
        self.assertEqual(original, {r["id"]: r["proposed_split"] for r in records})


@unittest.skipUnless(HAS_DEPS and SOURCE.is_dir(), "requires PyArrow, CPU video dependencies, and pinned source")
class WgoNativePreparationTests(unittest.TestCase):
    def test_real_parquet_selects_shortest_supported_dev_and_preserves_video_bytes(self):
        import av
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, original = [], {}
            for index in range(10):
                video = root / f"source-{index}.mp4"
                frame_count = 8 + index * 2
                with av.open(str(video), "w") as container:
                    stream = container.add_stream("mpeg4", rate=8)
                    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                    for frame_index in range(frame_count):
                        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 16 + frame_index,
                                                                  dtype=np.uint8), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                identity = f"homer_{index:04d}"
                original[identity] = video.read_bytes()
                rows.append(dict(id=identity, video=original[identity], instruction=f"Move object number {index}.",
                                 segments=[dict(start_sec=0.0, end_sec=frame_count / 8, subtask="Move the object.")],
                                 metadata=json.dumps(dict(scene_id=f"scene-{index}"))))
            rows.append(dict(id="homer_bad", video=b"invalid native video", instruction="Invalid example.",
                             segments=[dict(start_sec=0.0, end_sec=1.0, subtask="Move the object.")], metadata="{}"))
            rows.append(dict(id="droid_0001", video=b"not decoded for this profile", instruction="Robot example.",
                             segments=[dict(start_sec=0.0, end_sec=1.0, subtask="Move the object.")], metadata="{}"))
            parquet = root / "fixture.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            pin = DatasetPin("synthetic/unit-test", "fixture-v1", file_sha256(parquet))
            index = inspect_parquet(parquet, pin=pin)
            self.assertEqual(index["row_count"], 12)
            self.assertEqual(index["split_status"], "proposal_not_frozen")
            public = root / "public.json"
            output = root / "prepared"
            summary = prepare_parquet(parquet, SOURCE, output, pin=pin, public_manifest=public)
            self.assertEqual(summary["selected_count"], 3)
            self.assertEqual(summary["new_model_calls"], 0)
            self.assertEqual(summary["reencoded_videos"], 0)
            eligible = [c for c in summary["candidates"] if c["native_supported"] and c["references_valid"] and
                        c["proposed_split"] == "development"]
            expected = [c["id"] for c in sorted(eligible, key=lambda c: (c["duration_s"], c["id"]))[:3]]
            self.assertEqual(summary["selected_ids"], expected)
            jobs = load_manifest(output / "jobs.jsonl", contract={"test": "native-input-validation"})
            for job in jobs:
                self.assertEqual(job.video.read_bytes(), original[job.id])
            serialized = public.read_text()
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private_validation_error", serialized)
            bad = next(c for c in summary["candidates"] if c["id"] == "homer_bad")
            self.assertFalse(bad["native_supported"])
            self.assertIn("Native validation rejected", bad["exclusion_reason"])
            with self.assertRaisesRegex(ValueError, "fresh output"):
                prepare_parquet(parquet, SOURCE, output, pin=pin)

    def test_checksum_failure_precedes_parquet_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.parquet"
            path.write_bytes(b"not parquet")
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_parquet(path, pin=DatasetPin("fixture", "one", "0" * 64))


if __name__ == "__main__":
    unittest.main()
