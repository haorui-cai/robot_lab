"""Discover event-bearing runs and describe their saved configuration and lifecycle."""

import hashlib
import json
import math
import re
from pathlib import Path

import yaml


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_yaml(path):
    try:
        value = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flag(value):
    return {"true": True, "false": False}.get(str(value).lower())


def integer(*values):
    for value in values:
        parsed = number(value)
        if parsed is not None and math.isfinite(parsed) and parsed >= 0 and parsed.is_integer():
            return int(parsed)
    return None


def continuation(meta, agent, event, iteration, kind):
    """Keep absolute iteration indices separate from this invocation's budget."""
    expected = meta.get("expected_iteration_range")
    expected = expected if isinstance(expected, list) and len(expected) == 2 else [None, None]
    start = integer(meta.get("first_iteration"), meta.get("resume_cursor"), expected[0])
    if start is None and (kind == "scratch" or kind == "finetune" and flag(agent.get("resume")) is False):
        start = 0
    budget = integer(meta.get("additional_iterations"), event.get("iterations_requested"),
                     meta.get("iterations"), meta.get("max_iterations"), agent.get("max_iterations"))
    total = integer(meta.get("target_cumulative_iterations"), meta.get("target_phase_iteration"))
    end = integer(meta.get("final_iteration"), expected[1])
    if end is None and total is not None and total > 0:
        end = total - 1
    if end is None and start is not None and budget is not None and budget > 0:
        end = start + budget - 1
    if total is None and end is not None:
        total = end + 1
    if budget is None and start is not None and end is not None and end >= start:
        budget = end - start + 1
    done = None
    if start is not None and iteration is not None:
        done = max(0, iteration - start + 1)
    elif event.get("status") == "completed":
        done = integer(event.get("iterations_completed"))
    if done is not None and budget is not None:
        done = min(done, budget)
    return {"first_iteration": start, "final_iteration": end, "segment_iterations": budget,
            "segment_completed": done, "total_iterations": total}


def reward_focus(meta, previous):
    """Describe the saved launch revision, never the mutable working-tree defaults."""
    changes = meta.get("reward_changes")
    if not isinstance(changes, dict):
        return []
    focus = []

    def weight(key, reward, zh, en):
        new = number(changes.get(key))
        old_cfg = previous.get(reward, {})
        old = number(old_cfg.get("weight")) if isinstance(old_cfg, dict) else None
        if new is not None and new != old:
            value = f"{old:g} → {new:g}" if old is not None else f"{new:g}"
            focus.append({"zh": f"{zh} {value}", "en": f"{en} {value}"})

    weight("feet_air_time_weight", "feet_air_time", "腾空奖励", "Air-time reward")
    if changes.get("feet_gait") == "PeriodNormalizedGaitReward":
        focus.append({"zh": "步态按完整周期归一化", "en": "Gait normalized by completed cycle"})
    weight("joint_pos_penalty_weight", "joint_pos_penalty", "站姿偏离权重", "Posture penalty weight")
    if "cadence_scoring" in changes:
        focus.append({"zh": "2 Hz 步频逐脚计分，固定除以 4", "en": "2 Hz cadence scored per foot, fixed divisor 4"})
    elif number(changes.get("target_frequency_hz")) is not None:
        hz = number(changes["target_frequency_hz"])
        focus.append({"zh": f"新增 {hz:g} Hz 步频奖励，关闭腾空奖励",
                      "en": f"Add {hz:g} Hz cadence reward; disable air-time reward"})
    if changes.get("clearance_shortfall_only") is True:
        cm = number(changes.get("target_clearance_m"))
        scale = number(changes.get("clearance_error_scale_m"))
        weight_value = number(changes.get("feet_ground_clearance_weight"))
        if cm is not None and scale is not None and weight_value is not None:
            focus.append({"zh": f"只罚不足 {cm * 100:g} cm；归一化尺度 {scale:g} m，权重 {weight_value:g}",
                          "en": f"Penalize clearance below {cm * 100:g} cm only; scale {scale:g} m, weight {weight_value:g}"})
    elif number(changes.get("target_clearance_m")) is not None:
        cm = number(changes["target_clearance_m"]) * 100
        focus.append({"zh": f"足高改为对地 {cm:g} cm 净空误差", "en": f"Replace body-relative foot height with {cm:g} cm ground clearance"})
    previous_posture = previous.get("joint_pos_penalty", {})
    previous_params = previous_posture.get("params", {}) if isinstance(previous_posture, dict) else {}
    if number(changes.get("commanded_swing_joint_scale")) == 0 and number(previous_params.get("swing_scale")) != 0:
        focus.append({"zh": "运动时摆动腿全部关节免站姿惩罚", "en": "Exempt all swing-leg joints from moving posture penalty"})
    return focus


def discover(root):
    metadata = {}
    for file in root.glob("outputs/*/launch_metadata.json"):
        item = read_json(file)
        if item.get("run_directory"):
            directory = Path(item["run_directory"]).resolve()
            if directory.is_relative_to(root):
                metadata[directory] = (item, file)
    # Discover event files, rather than maintaining a list of experiment names.
    directories = {p.parent.resolve() for p in (root / "logs/rsl_rl").glob("*/*/events.out.tfevents.*")}
    directories.update(metadata)
    for directory in sorted(directories):
        events = list(directory.glob("events.out.tfevents.*"))
        if directory.is_relative_to(root) and events:
            yield directory, events, *metadata.get(directory, ({}, None))


def describe(root, directory, events, meta, meta_file):
    agent = read_yaml(directory / "params/agent.yaml")
    env = read_yaml(directory / "params/env.yaml")
    output = meta_file.parent if meta_file else None
    result = read_json(output / "result.json") if output else {}
    event = read_json(output / "training_completed.json") if output else {}
    if isinstance(result.get("event"), dict):
        event = result["event"] or event
    started = read_json(output / "started.json") if output else {}
    pid = meta.get("pid") or started.get("training_pid")
    alive = False
    if pid:
        try:
            command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="replace")
            expected = (str(Path(meta["snapshot"]) / "scripts/reinforcement_learning/rsl_rl/train.py")
                        if meta.get("snapshot") else "scripts/reinforcement_learning/rsl_rl/train.py")
            run_name = directory.name[20:]
            alive = expected in command and (not run_name or run_name in command)
        except (OSError, ValueError):
            pass
    tail = ""
    if output:
        path = output / "train.log"
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 32000))
                tail = stream.read().decode(errors="replace")
        except OSError:
            pass
    matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", tail)
    iteration = int(matches[-1][0]) if matches else None
    target = (event.get("iterations_requested") or meta.get("iterations") or meta.get("max_iterations")
              or meta.get("target_phase_iteration") or number(agent.get("max_iterations")))
    if matches and event.get("status") != "completed":
        target = int(matches[-1][1])
    completed = event.get("iterations_completed")
    if event.get("status") == "completed" and completed == target and Path(event.get("checkpoint", "")).is_file():
        status = "completed"
        iteration = event.get("last_iteration_index", iteration)
    elif result.get("status") == "failed":
        status = "failed"
    elif alive:
        status = "running"
    elif "Training time:" in tail:
        status = "completed"
    elif output:
        status = "stopped"
    else:
        status = "unknown"
    if status == "completed" and completed is None and iteration is not None and iteration + 1 == target:
        completed = target
    initialization = meta.get("initialization", meta.get("load", ""))
    if flag(agent.get("resume")) is True:
        kind = "resume"
    elif (directory / "warm_start.txt").exists() or (initialization and not any(w in initialization.lower() for w in ("random", "fresh"))):
        kind = "finetune"
    elif flag(agent.get("resume")) is False or any(w in initialization.lower() for w in ("random", "fresh")):
        kind = "scratch"
    else:
        kind = "other"
    zh, en = {"scratch": ("从零训练", "Training from scratch"), "resume": ("恢复训练", "Resumed training"),
              "finetune": ("续训", "Fine-tuning"), "other": ("训练记录", "Training run")}[kind]
    phase = continuation(meta, agent, event, iteration, kind)
    if phase["total_iterations"] is None and matches:
        phase["total_iterations"] = int(matches[-1][1])
        phase["final_iteration"] = int(matches[-1][1]) - 1
    if phase["total_iterations"] is not None:
        target = phase["total_iterations"]
    title, title_en = zh, en
    version = re.search(r"(?:^|_)rewardv(\d+)(?:_|$)", directory.name)
    branch = meta.get("experiment")
    if directory.parent.name == "tqbot2_standard_walk" and version and branch in ("scratch", "warmstart100"):
        branch_zh, branch_en = {"scratch": ("从零组", "Scratch lineage"),
                                "warmstart100": ("继承组", "Warm-start lineage")}[branch]
        title = f"行走 v{version[1]} · {branch_zh}"
        title_en = f"Walk v{version[1]} · {branch_en}"
    source = meta.get("source_checkpoint") or meta.get("resume_checkpoint")
    source = str(source) if source else None
    previous = {}
    if source and Path(source).resolve().is_relative_to(root.resolve()):
        previous = read_yaml(Path(source).parent / "params/env.yaml").get("rewards", {})
    previous = previous if isinstance(previous, dict) else {}
    phase.update(source_checkpoint=Path(source).name if source else None,
                 source_run=Path(source).parent.name if source else None,
                 source_path=source, focus=reward_focus(meta, previous))
    robot = env.get("scene", {}).get("robot", {})
    events_cfg = env.get("events", {})
    material_event = events_cfg.get("randomize_foot_material")
    if not isinstance(material_event, dict):
        material_event = events_cfg.get("randomize_rigid_body_material")
    material = material_event.get("params", {}) if isinstance(material_event, dict) else {}
    friction = material.get("static_friction_range") or meta.get("static_foot_friction") or meta.get("ordinary_foot_static_range") or []
    dynamic = material.get("dynamic_friction_range") or meta.get("dynamic_foot_friction") or []
    algorithm = agent.get("algorithm", {})
    policy = agent.get("policy", {})
    slide = env.get("rewards", {}).get("feet_slide")
    slide = slide if isinstance(slide, dict) else {}
    selection = env.get("foot_link_name") or slide.get("params", {}).get("asset_cfg", {}).get("body_names")
    if isinstance(selection, list):
        selection = ", ".join(selection)
    torques = {name: number(cfg.get("effort_limit") if number(cfg.get("effort_limit")) is not None else cfg.get("effort_limit_sim"))
               for name, cfg in robot.get("actuators", {}).items()}
    notification = result.get("notification_retry", result.get("notification", {}))
    generator = env.get("scene", {}).get("terrain", {}).get("terrain_generator", {})
    generator = generator if isinstance(generator, dict) else {}
    profile = {
        "saved_params": bool(agent or env), "num_envs": number(env.get("scene", {}).get("num_envs")),
        "seed": number(agent.get("seed")), "initialization": kind,
        "learning_rate": number(algorithm.get("learning_rate")), "schedule": algorithm.get("schedule"),
        "entropy_coef": number(algorithm.get("entropy_coef")),
        "actor_normalization": flag(agent.get("actor", {}).get("obs_normalization", policy.get("actor_obs_normalization"))),
        "critic_normalization": flag(agent.get("critic", {}).get("obs_normalization", policy.get("critic_obs_normalization"))),
        "foot_selection": selection, "torque_limits": torques,
        "static_friction": [number(v) for v in friction], "dynamic_friction": [number(v) for v in dynamic],
        "terrain_types": list(generator.get("sub_terrains", {})),
        "checkpoint": Path(event["checkpoint"]).name if event.get("checkpoint") else None,
        "finished_at": event.get("finished_at") or result.get("finished_at"),
        "notification_sent": notification.get("return_code") == 0 if notification else None,
        "slip_function": slide.get("func"),
    }
    return {
        "id": hashlib.sha256(str(directory.relative_to(root)).encode()).hexdigest()[:16],
        "legacy_id": hashlib.sha256(directory.name.encode()).hexdigest()[:16],
        "title": title, "title_en": title_en, "kind": kind, "continuation": phase,
        "experiment": directory.parent.name, "name": directory.name, "path": str(directory),
        "status": status, "iteration": iteration, "target": target,
        "completed_iterations": completed if status == "completed" else None,
        "friction_min": number(friction[0]) if friction else None,
        "flat_friction": meta.get("flat_static_friction"), "profile": profile,
        "date": directory.name[:10] + " " + directory.name[11:19].replace("-", ":"),
        "mtime": max(p.stat().st_mtime for p in events),
    }
