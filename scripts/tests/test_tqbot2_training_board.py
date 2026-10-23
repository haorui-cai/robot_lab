"""Exercise bilingual log APIs and CSV export against real synthetic event files."""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

APP = Path(__file__).resolve().parents[1] / "tools/TQBot2_tensorboard_zh"
sys.path.insert(0, str(APP))
from server import Repository, create_handler, descriptor
from runs import continuation, describe


class TrainingBoardTests(unittest.TestCase):
    def test_completed_resume_keeps_absolute_range_and_segment_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "logs/rsl_rl/tqbot2_standard_walk/2026-10-16_15-09-42_rewardv4_fixture"
            (run / "params").mkdir(parents=True)
            (run / "params/agent.yaml").write_text("resume: true\nmax_iterations: 4499\n")
            events = run / "events.out.tfevents.fixture"
            events.touch()
            checkpoint = run / "model_11999.pt"
            checkpoint.touch()
            source = root / "logs/rsl_rl/tqbot2_standard_walk/previous/model_7500.pt"
            (source.parent / "params").mkdir(parents=True)
            (source.parent / "params/env.yaml").write_text("rewards:\n  joint_pos_penalty:\n    weight: -0.01\n")
            output = root / "outputs/fixture"
            output.mkdir(parents=True)
            (output / "training_completed.json").write_text(json.dumps({"status": "completed",
                "iterations_requested": 4499, "iterations_completed": 4499,
                "last_iteration_index": 11999, "checkpoint": str(checkpoint)}))
            meta = {"experiment": "scratch", "source_checkpoint": str(source), "first_iteration": 7501,
                    "final_iteration": 11999, "iterations": 4499, "additional_iterations": 4499,
                    "target_phase_iteration": 12000, "reward_changes": {"joint_pos_penalty_weight": -0.2}}
            row = describe(root, run, [events], meta, output / "launch_metadata.json")
            self.assertEqual((row["status"], row["iteration"], row["target"]), ("completed", 11999, 12000))
            self.assertEqual(row["title"], "行走 v4 · 从零组")
            self.assertEqual(row["continuation"]["first_iteration"], 7501)
            self.assertEqual(row["continuation"]["final_iteration"], 11999)
            self.assertEqual(row["continuation"]["segment_completed"], 4499)
            self.assertEqual(row["continuation"]["segment_iterations"], 4499)
            self.assertEqual(row["continuation"]["source_checkpoint"], "model_7500.pt")
            self.assertIn("-0.01 → -0.2", row["continuation"]["focus"][0]["zh"])

    def test_stopped_resume_and_missing_origin_do_not_imply_zero_start(self):
        phase = continuation({"first_iteration": 5301, "final_iteration": 11999,
                              "additional_iterations": 6699, "target_phase_iteration": 12000},
                             {}, {}, 7515, "resume")
        self.assertEqual(phase["segment_completed"], 2215)
        self.assertEqual(phase["total_iterations"], 12000)
        self.assertEqual(phase["segment_iterations"], 6699)
        unknown = continuation({}, {"max_iterations": "3000"}, {}, None, "resume")
        self.assertIsNone(unknown["first_iteration"])
        self.assertIsNone(unknown["final_iteration"])
        self.assertIsNone(unknown["segment_completed"])
        cursor = continuation({"resume_cursor": 10201, "expected_iteration_range": [10201, 14999],
                               "additional_iterations": 4799}, {}, {}, 14999, "resume")
        self.assertEqual(cursor["segment_completed"], 4799)
        self.assertEqual(cursor["total_iterations"], 15000)
        cumulative_event = continuation({"resume_cursor": 10201, "target_phase_iteration": 15000,
                                         "additional_iterations": 4799}, {},
                                        {"status": "completed", "iterations_completed": 15000,
                                         "iterations_requested": 15000}, 14999, "resume")
        self.assertEqual(cumulative_event["segment_completed"], 4799)

    def test_generic_discovery_saved_configuration_and_completion_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "logs/rsl_rl/future_experiment/2026-10-14_01-00-00_fixture"
            run.mkdir(parents=True)
            writer = EventFileWriter(str(run))
            writer.add_event(Event(step=2999, wall_time=1000,
                summary=Summary(value=[Summary.Value(tag="Curriculum/terrain_breakdown/boxes/mean", simple_value=4.)])))
            writer.close()
            (run / "params").mkdir()
            (run / "params/agent.yaml").write_text("resume: false\nmax_iterations: 3000\nalgorithm:\n  learning_rate: 0.001\n  schedule: adaptive\n  entropy_coef: 0.01\nactor:\n  obs_normalization: false\ncritic:\n  obs_normalization: false\n")
            (run / "params/env.yaml").write_text("foot_link_name: '.*_foot'\nscene:\n  num_envs: 4096\n  robot:\n    actuators:\n      calf:\n        effort_limit: 30.2175\n  terrain:\n    terrain_generator: null\nevents:\n  randomize_foot_material: null\n")
            output = root / "outputs/fixture"
            output.mkdir(parents=True)
            (output / "launch_metadata.json").write_text(json.dumps({"run_directory": str(run), "iterations": 3000}))
            checkpoint = run / "model_2999.pt"
            checkpoint.write_bytes(b"fixture")
            (output / "training_completed.json").write_text(json.dumps({"status": "completed", "iterations_completed": 3000,
                "iterations_requested": 3000, "last_iteration_index": 2999, "checkpoint": str(checkpoint)}))
            repository = Repository(root)
            row = repository.scan()[0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["completed_iterations"], 3000)
            self.assertEqual(row["iteration"], 2999)
            self.assertEqual(row["profile"]["torque_limits"]["calf"], 30.2175)
            self.assertIs(row["profile"]["actor_normalization"], False)
            self.assertEqual(row["profile"]["terrain_types"], [])
            data = repository.scalars([row["id"]], "terrain", "")
            self.assertEqual(data["metrics"][0]["name"], "随机方块 · 平均课程等级")

    def test_terrain_translation_and_statistic_boundaries(self):
        for stat in ("mean", "max", "env_count", "move_up_rate", "move_down_rate", "top_completed_rate", "level_9_fraction"):
            metric = descriptor(f"Curriculum/terrain_breakdown/hf_pyramid_slope/{stat}")
            self.assertIn("凸形坡面", metric["name"])
            self.assertIn("Pyramid slope", metric["name_en"])
            self.assertEqual(metric["direction"], "neutral")
        self.assertNotIn("未记录", descriptor("Curriculum/terrain_levels")["hint"])

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.run_dir = cls.root / "logs/rsl_rl/tqbot2_rough/2026-10-13_10-15-13_resume3000"
        cls.run_dir.mkdir(parents=True)
        writer = EventFileWriter(str(cls.run_dir))
        cls.tags = ["Train/mean_reward", "Curriculum/terrain_levels", "Episode_Reward/recovery_effort",
                    "Episode_Reward/recovery_torque_change", "Episode_Reward/joint_torques_l2"]
        for step in range(10):
            writer.add_event(Event(step=step, wall_time=1000+step,
                                   summary=Summary(value=[Summary.Value(tag=tag, simple_value=step*.1)
                                                          for tag in cls.tags])))
        writer.close()
        output = cls.root / "outputs/fixture"
        output.mkdir(parents=True)
        (output / "launch_metadata.json").write_text(json.dumps({"run_directory": str(cls.run_dir), "ordinary_foot_static_range": [.15, 1.]}))
        (output / "train.log").write_text("Learning iteration 9/10\nTraining time: 1 seconds\n")
        handler = create_handler(Repository(cls.root))
        handler.log_message = lambda *args: None
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close(); cls.thread.join(); cls.temporary.cleanup()

    def fetch(self, path):
        with urlopen(self.base+path, timeout=10) as response:
            return response.read(), response.headers

    def run_id(self):
        runs = json.loads(self.fetch("/api/runs")[0])["runs"]
        self.assertEqual(runs[0]["status"], "completed")
        self.assertEqual(runs[0]["iteration"], 9)
        return runs[0]["id"]

    def test_raw_event_values_and_bilingual_names(self):
        data = json.loads(self.fetch("/api/scalars?"+urlencode({"runs": self.run_id(), "group": "all"}))[0])
        source = EventAccumulator(str(self.run_dir), size_guidance={"scalars": 0}); source.Reload()
        for tag in self.tags:
            expected = [[e.step,e.value,e.wall_time] for e in source.Scalars(tag)]
            self.assertEqual(data["runs"][0]["values"][tag], expected)
        for metric in data["metrics"]:
            self.assertTrue(metric["name"] and metric["name_en"])
        self.assertIn("所有并行环境", next(m for m in data["metrics"] if m["tag"]=="Curriculum/terrain_levels")["hint"])

    def test_search_both_languages(self):
        for query in ("力矩", "torque"):
            data = json.loads(self.fetch("/api/scalars?"+urlencode({"runs":self.run_id(), "q":query}))[0])
            self.assertEqual(len(data["metrics"]), 3)

    def test_csv_preserves_raw_values_and_zoom_window(self):
        data, headers = self.fetch("/api/export.csv?"+urlencode({"runs":self.run_id(), "group":"overview", "low":3, "high":5}))
        self.assertIn("attachment", headers["Content-Disposition"])
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual([int(r[1]) for r in rows[1:]], [3,4,5])
        self.assertIn("Mean episode return", rows[0][2])
        source = EventAccumulator(str(self.run_dir), size_guidance={"scalars": 0}); source.Reload()
        self.assertEqual(float(rows[1][2]), source.Scalars("Train/mean_reward")[3].value)

    def test_boundaries_and_simultaneous_language_page(self):
        for path, status in (("/server.py",404), ("/../catalog.py",404), ("/api/scalars?runs=unknown",400)):
            with self.assertRaises(HTTPError) as error: self.fetch(path)
            self.assertEqual(error.exception.code, status)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base+"/api/runs", headers={"Host":"foreign.example"}))
        self.assertEqual(error.exception.code, 403)
        html=self.fetch("/")[0].decode()
        self.assertIn("训练看板 / Training Board",html)
        self.assertNotIn('id="langZh"',html)


if __name__ == "__main__":
    unittest.main()
