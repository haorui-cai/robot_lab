"""Run a child job and notify once on exit, without polling its progress."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def classify_exit(exit_code, event_path, expected_iterations):
    """A zero exit alone cannot establish that Isaac completed learning."""
    try:
        event = json.loads(event_path.read_text())
    except (OSError, ValueError):
        return False, {"reason": "Training exited without a valid completion event."}
    valid = (
        exit_code == 0
        and event.get("status") == "completed"
        and event.get("iterations_requested") == expected_iterations
        and event.get("iterations_completed") == expected_iterations
        and Path(event.get("checkpoint", "")).is_file()
    )
    return valid, event


def notify(title, body):
    result = subprocess.run(
        ["notify-send", "--app-name=TQBot2训练", "--urgency=critical",
         "--expire-time=0", title, body],
        capture_output=True, text=True, timeout=15, check=False,
    )
    return {"return_code": result.returncode, "notification_id": result.stdout.strip(),
            "error": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A training command is required after --")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_path = args.event_file.resolve()
    if event_path.exists() or (output / "started.json").exists():
        parser.error("Use a fresh output directory and completion event path")
    status = {"status": "starting", "started_at": now(), "supervisor_pid": os.getpid(),
              "iterations": args.iterations, "command": command,
              "trigger": "blocking child wait; desktop notification once after exit; no polling"}
    try:
        with (output / "train.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status.update(status="running", training_pid=process.pid)
            write_json(output / "started.json", status)
            # One blocking waitpid: no sleeps, log readers, or periodic agent work.
            exit_code = process.wait()
        completed, event = classify_exit(exit_code, event_path, args.iterations)
        status.update(status="completed" if completed else "failed", finished_at=now(),
                      exit_code=exit_code, event=event)
    except Exception as error:
        completed = False
        status.update(status="failed", finished_at=now(), error=str(error))
    title = "TQBot2 训练完成" if completed else "TQBot2 训练异常退出"
    body = (f"{args.iterations} 轮训练已完成。\n模型：{status['event']['checkpoint']}" if completed
            else f"训练未确认完成 {args.iterations} 轮。\n日志：{output / 'train.log'}")
    # Persist the result even when the desktop notification service is unavailable.
    write_json(output / "result.json", status)
    try:
        status["notification"] = notify(title, body)
    except Exception as error:
        status["notification"] = {"error": str(error)}
    write_json(output / "result.json", status)


if __name__ == "__main__":
    main()
