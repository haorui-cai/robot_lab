"""Read-only local Chinese dashboard for existing TQBot2 TensorBoard event files."""

import argparse
import csv
import hashlib
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from catalog import LABELS, ENGLISH

ASSETS = Path(__file__).resolve().parent
EXTRA = {
    "Train/mean_episode_length": "平均回合长度",
    "Loss/value": "价值函数损失", "Loss/surrogate": "策略损失", "Loss/entropy": "熵项",
    "Loss/learning_rate": "学习率", "Policy/mean_std": "动作探索标准差",
    "Perf/total_fps": "采样速度", "Perf/collection_time": "采样耗时", "Perf/learning_time": "学习耗时",
    "Episode_Termination/time_out": "到达回合时限的比例",
    "Episode_Termination/terrain_out_of_bounds": "地形越界终止比例",
    "Episode_Termination/high_friction_flat_exit": "高摩擦平地区域退出比例",
}
GROUPS = {
    "overview": ["Train/mean_reward", "Curriculum/terrain_levels", "Metrics/base_velocity/error_vel_xy",
                 "Metrics/base_velocity/error_vel_yaw", "Episode_Reward/walking_energy_efficiency",
                 "Episode_Reward/feet_swing_clearance", "Episode_Reward/recovery_effort", "Train/mean_episode_length"],
    "tracking": ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
                 "Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/track_ang_vel_z_exp",
                 "Episode_Reward/lin_vel_z_l2", "Episode_Reward/ang_vel_xy_l2",
                 "Episode_Reward/stand_still", "Episode_Reward/upward"],
    "feet": ["Episode_Reward/" + name for name in ("feet_slide", "feet_swing_clearance", "feet_height_body",
             "feet_air_time", "feet_air_time_variance", "feet_gait", "feet_contact_without_cmd", "contact_forces")],
    "energy": ["Episode_Reward/" + name for name in ("walking_energy_efficiency", "walking_joule_proxy", "joint_power",
               "joint_torques_l2", "joint_acc_l2", "joint_pos_limits", "joint_pos_penalty", "joint_mirror", "action_rate_l2", "undesired_contacts")],
    "recovery": ["Episode_Reward/" + name for name in ("recovery_effort", "recovery_torque_change", "recovery_joint_velocity", "upward", "undesired_contacts")],
}


def descriptor(tag):
    name = EXTRA.get(tag, LABELS.get(tag, LABELS.get(tag.split("/")[-1], tag)))
    name_en = ENGLISH.get(tag, ENGLISH.get(tag.split("/")[-1], tag))
    unit = ""
    if tag == "Metrics/base_velocity/error_vel_xy": unit = "m/s"
    if tag == "Metrics/base_velocity/error_vel_yaw": unit = "rad/s"
    if tag == "Curriculum/terrain_levels": unit = "等级 0–9"
    if tag == "Train/mean_episode_length": unit = "控制步"
    if tag.endswith("_time") and tag.startswith("Perf/"): unit = "秒"
    if tag == "Perf/total_fps": unit = "步/秒"
    hint = ""
    if tag.startswith("Episode_Reward/"):
        hint = "已加权奖励；负值趋近 0 表示惩罚减小。训练统计受命令、课程和回合长度影响。"
    if tag.startswith("Metrics/"):
        hint = "Isaac 命令时间归一化的误差指标，不是固定工况测试的 RMSE；越低越好。"
    if tag == "Curriculum/terrain_levels":
        hint = "所有并行环境的平均等级，包含平地；不是坡面单独等级或最高通过等级。日志未记录分地形等级。"
    hint_en = ""
    if tag.startswith("Episode_Reward/"):
        hint_en = "Weighted reward; negative penalties improve toward zero. Commands, curriculum and episode length affect training statistics."
    if tag.startswith("Metrics/"):
        hint_en = "Isaac command-time-normalized error metric, not fixed-condition RMSE. Lower is better."
    if tag == "Curriculum/terrain_levels":
        hint_en = "Mean across all environments, including flat terrain; not a slope-only level or highest passed level. Per-terrain levels were not logged."
    unit_en = {"等级 0–9": "level 0–9", "控制步": "control steps", "秒": "seconds", "步/秒": "steps/s"}.get(unit, unit)
    return {"tag": tag, "name": name, "name_en": name_en, "unit": unit, "unit_en": unit_en, "hint": hint, "hint_en": hint_en,
            "direction": "lower" if tag.startswith("Metrics/") else "higher" if tag.startswith("Episode_Reward/") or tag == "Train/mean_reward" else "neutral"}


class Repository:
    def __init__(self, root):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self.runs = {}
        self.scan_at = 0
        self.cache = {}

    def scan(self):
        with self.lock:
            if time.monotonic() - self.scan_at < 10:
                return sorted(self.runs.values(), key=lambda row: row["mtime"], reverse=True)
            metadata = {}
            for path in self.root.glob("outputs/*/launch_metadata.json"):
                try:
                    item = json.loads(path.read_text())
                    if item.get("run_directory"):
                        metadata[str(Path(item["run_directory"]).resolve())] = (item, path)
                except (OSError, ValueError):
                    continue
            runs = {}
            for directory in (self.root / "logs/rsl_rl/tqbot2_rough").glob("*"):
                if not directory.is_dir() or not any(directory.glob("events.out.tfevents.*")):
                    continue
                meta, meta_path = metadata.get(str(directory.resolve()), ({}, None))
                run_id = hashlib.sha256(directory.name.encode()).hexdigest()[:16]
                tail = ""
                if meta_path:
                    log_path = meta_path.parent / "train.log"
                    try:
                        with log_path.open("rb") as stream:
                            stream.seek(max(0, log_path.stat().st_size - 16000))
                            tail = stream.read().decode(errors="replace")
                    except OSError:
                        pass
                alive = False
                if meta.get("pid"):
                    try:
                        command = Path(f"/proc/{int(meta['pid'])}/cmdline").read_bytes().decode(errors="replace")
                        expected = str(Path(meta["snapshot"]) / "scripts/reinforcement_learning/rsl_rl/train.py")
                        alive = expected in command
                    except (OSError, ValueError, KeyError):
                        pass
                status = "running" if alive else "completed" if "Training time:" in tail else "stopped"
                matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", tail)
                progress = [int(v) for v in matches[-1]] if matches else [None, meta.get("max_iterations", meta.get("target_phase_iteration"))]
                low = meta.get("ordinary_foot_static_range", [meta.get("foot_friction_min")])[0]
                load = meta.get("load", "")
                if "random" in load:
                    title, kind = f"从零训练 · 摩擦下限 {low}", "scratch"
                elif "resume3000" in directory.name:
                    title, kind = "续训完成 · 高摩擦平地", "resume"
                    if status == "running": title = "继续训练 · 高摩擦平地"
                elif "flat_mu2p5" in directory.name:
                    title, kind = "续训 · 磁盘中断前", "finetune"
                elif "foot_ground" in directory.name:
                    title, kind = "续训 · 足端间隙修正", "finetune"
                elif "gentle_energy" in directory.name:
                    title, kind = "续训 · 温和恢复与能耗", "finetune"
                elif "scratch20000" in directory.name:
                    title, kind = "原行走训练 · 新奖励前", "original"
                else:
                    title, kind = directory.name, "other"
                title_en = {"scratch": f"Scratch · friction floor {low}", "resume": "Finetune · high-friction flat",
                            "original": "Original walking training · before reward changes"}.get(kind, directory.name)
                if kind == "finetune":
                    title_en = "Finetune · before disk interruption" if "flat_mu2p5" in directory.name else "Finetune · foot clearance" if "foot_ground" in directory.name else "Finetune · gentle recovery & energy"
                runs[run_id] = {"id": run_id, "title": title, "title_en": title_en, "kind": kind, "name": directory.name,
                                "path": str(directory.resolve()), "status": status, "iteration": progress[0],
                                "target": progress[1], "friction_min": low,
                                "flat_friction": meta.get("flat_static_friction"),
                                "date": directory.name[:10] + " " + directory.name[11:19].replace("-", ":"),
                                "mtime": max(p.stat().st_mtime for p in directory.glob("events.out.tfevents.*"))}
            self.runs = runs
            self.scan_at = time.monotonic()
            return sorted(runs.values(), key=lambda row: row["mtime"], reverse=True)

    def scalars(self, run_ids, group, query):
        with self.lock:
            self.scan()
            selected = []
            all_tags = set()
            for run_id in run_ids:
                if run_id not in self.runs:
                    raise ValueError("训练记录不存在")
                run = self.runs[run_id]
                cached = self.cache.get(run_id)
                if cached is None or time.monotonic() - cached["loaded_at"] >= 15:
                    accumulator = cached["accumulator"] if cached else EventAccumulator(run["path"], size_guidance={"scalars": 0})
                    accumulator.Reload()
                    tags = [t for t in accumulator.Tags()["scalars"] if not t.endswith("/time")]
                    values = {}
                    for tag in tags:
                        # Keep every point; no lossy backend sampling or interpolation.
                        values[tag] = [[e.step, e.value, e.wall_time] for e in accumulator.Scalars(tag)]
                    cached = {"accumulator": accumulator, "values": values, "loaded_at": time.monotonic()}
                    self.cache[run_id] = cached
                all_tags.update(cached["values"])
                selected.append((run, cached["values"]))
            if group == "all":
                tags = sorted(all_tags)
            elif group == "diagnostics":
                tags = sorted(t for t in all_tags if t.startswith(("Loss/", "Policy/", "Perf/", "Episode_Termination/")))
            else:
                tags = [tag for tag in GROUPS.get(group, GROUPS["overview"]) if tag in all_tags]
            if query:
                tags = [t for t in sorted(all_tags) if query.lower() in (t + descriptor(t)["name"] + descriptor(t)["name_en"]).lower()]
            return {"metrics": [descriptor(tag) for tag in tags],
                    "runs": [{"id": run["id"], "values": {tag: values.get(tag, []) for tag in tags}} for run, values in selected],
                    "updated_at": time.time(), "point_policy": "all original event points"}


def create_handler(repository):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if self.headers.get("Host") not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                    return self.json_response({"error": "仅接受本机访问"}, 403)
                if url.path == "/api/runs":
                    return self.json_response({"runs": repository.scan()})
                if url.path == "/api/scalars":
                    params = parse_qs(url.query)
                    ids = params.get("runs", [""])[0].split(",")
                    if len(ids) > 8 or not ids[0]:
                        raise ValueError("请选择 1–8 个训练记录")
                    return self.json_response(repository.scalars(ids, params.get("group", ["overview"])[0], params.get("q", [""])[0]))
                if url.path == "/api/export.csv":
                    params = parse_qs(url.query)
                    ids = params.get("runs", [""])[0].split(",")
                    if len(ids) > 8 or not ids[0]:
                        raise ValueError("请选择 1–8 个训练记录")
                    data = repository.scalars(ids, params.get("group", ["overview"])[0], params.get("q", [""])[0])
                    axis = params.get("axis", ["iteration"])[0]
                    low = float(params["low"][0]) if "low" in params else None
                    high = float(params["high"][0]) if "high" in params else None
                    stream = io.StringIO(newline="")
                    writer = csv.writer(stream)
                    metrics = data["metrics"]
                    writer.writerow(["训练 / Run", "迭代 / Iteration", *[f"{m['name']} / {m['name_en']} [{m['tag']}]" for m in metrics]])
                    for run in data["runs"]:
                        rows = {}
                        for metric in metrics:
                            points = run["values"][metric["tag"]]
                            if not points:
                                continue
                            for step, value, wall in points:
                                x = (wall - points[0][2]) / 3600 if axis == "time" else step - points[0][0] if axis == "relative" else step
                                if (low is not None and x < low) or (high is not None and x > high):
                                    continue
                                rows.setdefault(step, {})[metric["tag"]] = value
                        for step in sorted(rows):
                            writer.writerow([repository.runs[run["id"]]["name"], step, *[rows[step].get(m["tag"], "") for m in metrics]])
                    payload = ("\ufeff" + stream.getvalue()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="TQBot2_training_metrics.csv"')
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers(); self.wfile.write(payload)
                    return
                assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
                if url.path not in assets:
                    return self.json_response({"error": "页面不存在"}, 404)
                filename, content_type = assets[url.path]
                payload = (ASSETS / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers(); self.wfile.write(payload)
            except ValueError as error:
                self.json_response({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                self.json_response({"error": "日志读取暂时失败，请稍后刷新"}, 500)

        def json_response(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(payload)

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), create_handler(Repository(args.root)))
    print(f"TQBot2 中文训练看板：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
