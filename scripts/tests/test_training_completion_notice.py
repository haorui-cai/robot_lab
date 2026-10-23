"""Do not announce success for early, failed, or marker-less simulator exits."""

import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "tools/run_training_with_completion_notice.py"
spec = importlib.util.spec_from_file_location("completion_notice", MODULE)
notice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notice)


class CompletionNoticeTests(unittest.TestCase):
    def test_notification_supports_older_notify_send(self):
        with patch.object(notice.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as call:
            self.assertEqual(notice.notify("完成", "3000轮")["return_code"], 0)
            self.assertNotIn("--print-id", call.call_args.args[0])

    def test_zero_exit_without_marker_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            valid, _ = notice.classify_exit(0, Path(directory) / "missing.json", 3000)
            self.assertFalse(valid)

    def test_only_full_success_with_checkpoint_is_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_2999.pt"
            checkpoint.write_bytes(b"fixture")
            event = root / "event.json"
            data = {"status": "completed", "iterations_requested": 3000,
                    "iterations_completed": 3000, "checkpoint": str(checkpoint)}
            event.write_text(json.dumps(data))
            self.assertTrue(notice.classify_exit(0, event, 3000)[0])
            self.assertFalse(notice.classify_exit(1, event, 3000)[0])
            data["iterations_completed"] = 2900
            event.write_text(json.dumps(data))
            self.assertFalse(notice.classify_exit(0, event, 3000)[0])
            data["iterations_completed"] = 3000
            data["checkpoint"] = str(root / "missing.pt")
            event.write_text(json.dumps(data))
            self.assertFalse(notice.classify_exit(0, event, 3000)[0])


if __name__ == "__main__":
    unittest.main()
