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
from server import Repository, create_handler


class TrainingBoardTests(unittest.TestCase):
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
