"""CPU-only, source-backed audit of two TQBot2 reward-design claims.

This constructs synthetic reward inputs; it does not load a policy or a simulator.
Production reward functions and ContactSensor timer code are AST-extracted verbatim.
Run with the isaaclab environment's Python; results include source SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "outputs/TQBot2_soft033_dual_20261015/training_source_snapshot"
REWARDS = SNAPSHOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
URDF = SNAPSHOT / "source/robot_lab/data/Robots/TQBot/TQBot2_description/urdf/TQBot2_walk.urdf"
ENV = REPO / "logs/rsl_rl/tqbot2_standard_walk/2026-10-15_15-28-09_soft033_warmstart100_12000_20261015_152617/params/env.yaml"
ISAACLAB = Path("/home/eureka/IsaacLab/source/isaaclab/isaaclab")
SENSOR = ISAACLAB / "sensors/contact_sensor/contact_sensor.py"
MATH = ISAACLAB / "utils/math.py"
MANAGER = ISAACLAB / "managers/reward_manager.py"
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class PlainLoader(yaml.SafeLoader):
    """Read serialized config data without constructing arbitrary Python objects."""


def unknown_yaml(loader, tag, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


PlainLoader.add_multi_constructor("", unknown_yaml)


class Scene(dict):
    @property
    def sensors(self):
        return self


def extract_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names), (path, names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {n.name: [n.lineno, n.end_lineno] for n in selected}


def extract_contact_timer(namespace):
    tree = ast.parse(SENSOR.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ContactSensor")
    first_contact = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_first_contact"))
    update = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_buffers_impl")
    block = copy.deepcopy(next(n for n in update.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.cfg.track_air_time"))
    fn = ast.FunctionDef(name="update_contact_timer", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="env_ids")], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[block], decorator_list=[])
    module = ast.Module(body=[first_contact, fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SENSOR), "exec"), namespace)
    return {"compute_first_contact": [first_contact.lineno, first_contact.end_lineno], "timer_update_block": [block.lineno, block.end_lineno]}


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def make_env(n=1):
    data = NS(projected_gravity_b=tensor([[0, 0, -1]]).repeat(n, 1), root_com_lin_vel_b=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3))
    robot = NS(data=data)
    sensor_data = NS(**{name: torch.zeros(n, 4, dtype=torch.float64) for name in ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time")}, net_forces_w=torch.zeros(n, 4, 3, dtype=torch.float64))
    sensor = NS(data=sensor_data, _data=sensor_data, cfg=NS(track_air_time=True, force_threshold=1.0))
    env = NS(scene=Scene(robot=robot, contact_forces=sensor), command_manager=NS(get_command=lambda _: tensor([[0.5, 0, 0]]).repeat(n, 1)), num_envs=n, device="cpu", step_dt=.02)
    return env


def gait_object(ns, env, cfg):
    obj = ns["GaitReward"].__new__(ns["GaitReward"])
    params = cfg["rewards"]["feet_gait"]["params"]
    for key in ("std", "command_name", "max_err", "velocity_threshold", "command_threshold"):
        setattr(obj, key, params[key])
    obj.synced_feet_pairs = [[FEET.index(name) for name in pair] for pair in params["synced_feet_pair_names"]]
    obj.contact_sensor = env.scene["contact_forces"]
    obj.asset = env.scene["robot"]
    return obj


def call_gait(obj, env, cfg):
    return obj(env, **cfg["rewards"]["feet_gait"]["params"])


def gait_audit(ns, cfg):
    weight = cfg["rewards"]["feet_gait"]["weight"]
    dt = cfg["sim"]["dt"]
    decimation = cfg["decimation"]
    step_dt = dt * decimation
    records = []
    for period in (.2, .4, 1.):
        for pattern in ("all_four_synchronous", "ideal_diagonal_alternation"):
            half_steps = round(period / 2 / dt)
            env = make_env()
            env.step_dt = step_dt
            sensor = env.scene["contact_forces"]
            sensor.compute_first_contact = lambda interval: ns["compute_first_contact"](sensor, interval)
            gait = gait_object(ns, env, cfg)
            gait_values, air_values, timer_values = [], [], []
            # Six periods; discard two to remove initial no-history transients.
            for k in range(1, 12 * half_steps + 1):
                # t=k*dt samples contact held on ((k-1)*dt, k*dt].
                # First half is stance for FL/RR; switch on the next physics sample.
                first_half = ((k - 1) // half_steps) % 2 == 0
                contact = [first_half] * 4 if pattern == "all_four_synchronous" else [first_half, not first_half, not first_half, first_half]
                sensor._timestamp = tensor([k * dt])
                sensor._timestamp_last_update = tensor([(k - 1) * dt])
                sensor.data.net_forces_w.zero_()
                sensor.data.net_forces_w[0, :, 2] = tensor(contact) * 10
                ns["update_contact_timer"](sensor, slice(None))
                if k > 4 * half_steps and k % decimation == 0:
                    gait_values.append(float(call_gait(gait, env, cfg)[0]) * weight)
                    air_cfg = cfg["rewards"]["feet_air_time"]
                    air_values.append(float(ns["feet_air_time"](env, command_name="base_velocity", sensor_cfg=NS(name="contact_forces", body_ids=list(range(4))), threshold=air_cfg["params"]["threshold"])[0]) * air_cfg["weight"])
                    timer_values.append(float(sensor.data.current_air_time[0, 0] + sensor.data.current_contact_time[0, 0]))
            assert np.isclose(np.mean(gait_values), .5) if pattern == "ideal_diagonal_alternation" else np.mean(gait_values) < .5
            # Production function evaluated at continuous phase t, uniform in each half-cycle.
            def continuous_value(t):
                s = sensor.data
                s.current_air_time.zero_()
                s.current_contact_time.zero_()
                if pattern == "all_four_synchronous":
                    s.current_air_time[:] = t
                else:
                    s.current_air_time[:, [0, 3]] = t
                    s.current_contact_time[:, [1, 2]] = t
                return float(call_gait(gait, env, cfg)[0]) * weight
            continuous_mean = quad(continuous_value, 0, period / 2, points=[.2] if period / 2 > .2 else None, epsabs=1e-12)[0] / (period / 2)
            if pattern == "all_four_synchronous":
                analytic_discrete = weight * np.exp(-8 * np.minimum(np.square(timer_values), gait.max_err ** 2) / gait.std)
                np.testing.assert_allclose(gait_values, analytic_discrete, atol=1e-12, rtol=1e-12)
            records.append({"period_s": period, "pattern": pattern, "continuous_uniform_phase_mean_weighted": continuous_mean, "discrete_mean_weighted": float(np.mean(gait_values)), "discrete_mean_actual_step_contribution": float(np.mean(gait_values)) * step_dt, "discrete_air_time_mean_weighted": float(np.mean(air_values)), "discrete_mean_gait_plus_air_weighted": float(np.mean(gait_values) + np.mean(air_values)), "samples": len(gait_values), "one_cycle_weighted_gait_samples": gait_values[:round(period / step_dt)], "one_cycle_active_timer_s": timer_values[:round(period / step_dt)]})
    return records


def parse_urdf():
    root = ET.parse(URDF).getroot()
    joints = {}
    for item in root.findall("joint"):
        origin = item.find("origin")
        local = np.eye(4)
        if origin is not None:
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        axis = item.find("axis")
        limit = item.find("limit")
        joints[item.get("name")] = {"name": item.get("name"), "parent": item.find("parent").get("link"), "child": item.find("child").get("link"), "type": item.get("type"), "local": local, "axis": np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else None, "bounds": (float(limit.get("lower")), float(limit.get("upper"))) if limit is not None and limit.get("lower") is not None else None}
    return joints


def fk(joints, q, link="FL_foot"):
    by_child = {j["child"]: j for j in joints.values()}
    chain = []
    while link != "trunk":
        j = by_child[link]
        chain.append(j)
        link = j["parent"]
    transform = np.eye(4)
    for j in reversed(chain):
        transform = transform @ j["local"]
        if j["type"] == "revolute":
            r = np.eye(4)
            r[:3, :3] = Rotation.from_rotvec(j["axis"] * q[j["name"]]).as_matrix()
            transform = transform @ r
    return transform[:3, 3]


def posture_audit(ns, cfg):
    joints = parse_urdf()
    defaults = cfg["scene"]["robot"]["init_state"]["joint_pos"]
    q0 = {name: next(value for expr, value in defaults.items() if re.fullmatch(expr, name)) for name, j in joints.items() if j["type"] == "revolute"}
    names = list(q0)
    moving = ["FL_thigh_joint", "FL_calf_joint"]
    x0 = np.array([q0[name] for name in moving])
    target = cfg["rewards"]["feet_height_body"]["params"]["target_height"]
    z0 = fk(joints, q0)[2]
    def pose(x):
        return q0 | dict(zip(moving, x))
    def height(x):
        return fk(joints, pose(x))[2]
    solutions = []
    for offset in ([0, 0], [.4, -.4], [-.4, -.4], [.8, -.8], [-.8, .8]):
        fit = minimize(lambda x: float(np.sum((x - x0) ** 2)), x0 + offset, method="SLSQP", bounds=[joints[n]["bounds"] for n in moving], constraints={"type": "eq", "fun": lambda x: height(x) - target}, options={"ftol": 1e-13, "maxiter": 300})
        if fit.success and abs(height(fit.x) - target) < 1e-8:
            solutions.append(fit)
    assert solutions, "IK optimizer found no feasible pose"
    best = min(solutions, key=lambda result: result.fun)
    # The two pitch axes are parallel. URDF offsets after the thigh and calf axes
    # give a global conservative bound on ||d z / d (q_thigh,q_calf)||.
    l1 = np.linalg.norm(joints["FL_calf_joint"]["local"][:3, 3])
    foot_joint = next(j for j in joints.values() if j["child"] == "FL_foot")
    l2 = np.linalg.norm(foot_joint["local"][:3, 3])
    lipschitz = float(np.hypot(l1 + l2, l2))
    delta_z = target - z0
    weight_height = abs(cfg["rewards"]["feet_height_body"]["weight"])
    weight_posture = abs(cfg["rewards"]["joint_pos_penalty"]["weight"])
    assert weight_height * 2 * abs(delta_z) * lipschitz < weight_posture
    assert abs(delta_z) / lipschitz <= np.sqrt(best.fun)
    rows = []
    for speed in (.5, 10.):
        env = make_env()
        env.scene["contact_forces"].data.current_air_time[0, 0] = .1
        data = env.scene["robot"].data
        data.default_joint_pos = tensor([[q0[name] for name in names]])
        data.root_pos_w = torch.zeros(1, 3, dtype=torch.float64)
        data.root_quat_w = tensor([[1, 0, 0, 0]])
        data.root_lin_vel_w = torch.zeros(1, 3, dtype=torch.float64)
        data.body_lin_vel_w = torch.zeros(1, 4, 3, dtype=torch.float64)
        data.body_lin_vel_w[0, 0, 0] = speed
        for fraction in (0., .01, .1, .25, .5, 1.):
            q = pose(x0 + fraction * (best.x - x0))
            data.joint_pos = tensor([[q[name] for name in names]])
            data.body_pos_w = tensor([[fk(joints, q, foot).tolist() for foot in FEET]])
            posture_cfg = cfg["rewards"]["joint_pos_penalty"]
            kwargs = posture_cfg["params"] | {"asset_cfg": NS(name="robot", joint_ids=list(range(len(names))))}
            posture = float(ns["joint_pos_penalty"](env, **kwargs)[0]) * posture_cfg["weight"]
            height_cfg = cfg["rewards"]["feet_height_body"]
            kwargs = height_cfg["params"] | {"asset_cfg": NS(name="robot", body_ids=list(range(4))), "sensor_cfg": NS(name="contact_forces", body_ids=list(range(4)))}
            clearance = float(ns["feet_height_body"](env, **kwargs)[0]) * height_cfg["weight"]
            rows.append({"relative_horizontal_speed_mps": speed, "interpolation_fraction": fraction, "foot_z_m": height(x0 + fraction * (best.x - x0)), "joint_pos_penalty_weighted": posture, "feet_height_body_weighted": clearance, "sum_weighted": posture + clearance})
    return {"default_foot_z_m": float(z0), "target_foot_z_m": target, "lift_m": float(delta_z), "moving_joints": moving, "default_angles_rad": x0.tolist(), "optimized_angles_rad": best.x.tolist(), "optimized_displacement_norm_rad": float(np.sqrt(best.fun)), "optimizer": "SLSQP, five starts, height-only equality; numerical candidate, not a global optimality proof", "link_lengths_m": [float(l1), float(l2)], "global_z_lipschitz_bound_m_per_rad": lipschitz, "rigorous_joint_displacement_lower_bound_for_target_rad": float(abs(delta_z) / lipschitz), "posture_cost_lower_bound_for_target": float(weight_posture * abs(delta_z) / lipschitz), "maximum_height_gain_for_target": float(weight_height * delta_z ** 2), "height_gain_upper_bound_per_displacement_norm": float(weight_height * 2 * abs(delta_z) * lipschitz), "posture_cost_per_displacement_norm": weight_posture, "sampled_production_rewards": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/TQBot2_reward_tradeoff_audit_20261016")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = yaml.load(ENV.read_text(), Loader=PlainLoader)
    ns = {"torch": torch, "ManagerTermBase": object}
    lines = {"reward_functions": extract_functions(REWARDS, {"GaitReward", "joint_pos_penalty", "feet_height_body", "feet_air_time"}, ns)}
    math_ns = {"torch": torch}
    lines["math"] = extract_functions(MATH, {"quat_apply_inverse"}, math_ns)
    ns["math_utils"] = NS(quat_apply_inverse=math_ns["quat_apply_inverse"])
    lines["contact_sensor"] = extract_contact_timer(ns)
    result = {"scope": "Synthetic, CPU-only mathematical reward audit; no policy rollout, no simulator, no training", "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (REWARDS, URDF, ENV, SENSOR, MATH, MANAGER, Path(__file__))], "extracted_lines": lines, "parameters": {"sim_dt": cfg["sim"]["dt"], "decimation": cfg["decimation"], "reward_step_dt": cfg["sim"]["dt"] * cfg["decimation"], "gait": cfg["rewards"]["feet_gait"], "joint_pos_penalty": cfg["rewards"]["joint_pos_penalty"], "feet_height_body": cfg["rewards"]["feet_height_body"], "feet_air_time": cfg["rewards"]["feet_air_time"]}, "assumptions": ["Upright base and movement command [0.5,0,0] keep reward gates fully open.", "Gaits have 50% stance and 50% air; prescribed contact patterns, no dynamics or physical feasibility claim.", "Sensor is updated every .005 s; contact held on ((k-1)*dt,k*dt], evaluated at k*dt; policy reward evaluated after every fourth physics update. Two full periods are discarded, four averaged.", "Current air/contact timers and last-air-time update use installed ContactSensor production AST. Switching is aligned just after half-period endpoints; other contact phase offsets can change discrete samples.", "Continuous averages integrate uniform time through one half-period and directly call the frozen production GaitReward at each quadrature point.", "Only FL thigh/calf vary in posture experiment; hip and all other joints stay at the configured default, root is fixed, and only FL is flagged airborne.", "Foot-height reward depends on foot LINK ORIGIN in trunk frame, not ground clearance; its prescribed horizontal relative speed is held equal between compared inputs.", "The posture test controls the same reward inputs; it does not assert these inputs form a dynamically feasible trajectory.", "All displayed weighted values omit RewardManager's common .02 multiplier unless explicitly named actual_step_contribution. This common factor does not change comparisons.", "The installed IsaacLab sensor/math/manager sources are hashed but are not part of the frozen training snapshot; rewards and URDF are frozen training-source files."], "limitations": ["No checkpoint evaluated. This cannot establish which term caused current policy behavior or whether full expected return improves after a change.", "Other rewards (tracking, symmetry, torque, action rate, contacts) and future state values are excluded from the two-term posture comparison.", "Joint norm is over all selected joints: marginal cost away from the default can be lower or even decrease; the default-based proof is not a universal bound around arbitrary policy states.", "IK is numerical and only constrains foot z; foot x can change. Analytic lower bounds, not optimizer optimality, establish the stated two-term conflict.", "No physically valid contact forces or terrain are simulated; prescribed binary contact states are used only to test reward discrimination."], "gait": gait_audit(ns, cfg), "posture": posture_audit(ns, cfg)}
    result["assumptions"].extend(["Reward calculations use float64 on CPU for audit precision; training commonly uses float32.", "GaitReward constructor is bypassed to avoid simulator dependencies; its fields and resolved four-foot IDs are populated from frozen configuration, and __call__ plus both helper methods run unmodified."])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output / "audit.json"), "gait": [{k: r[k] for k in ("period_s", "pattern", "continuous_uniform_phase_mean_weighted", "discrete_mean_weighted", "discrete_air_time_mean_weighted")} for r in result["gait"]], "posture": {k: v for k, v in result["posture"].items() if k != "sampled_production_rewards"}}, indent=2))


if __name__ == "__main__":
    main()
