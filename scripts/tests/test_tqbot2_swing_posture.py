"""CPU reward-contract tests; execute production AST without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg


def production_symbol(filename, name):
    tree = ast.parse((MDP / filename).read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerTermBase}
    exec(compile(ast.fix_missing_locations(module), str(MDP / filename), "exec"), namespace)
    return namespace[name]


Penalty = production_symbol("swing_posture.py", "SwingJointPositionPenalty")
original_penalty = production_symbol("rewards.py", "joint_pos_penalty")
LEGS = ("FR", "FL", "RR", "RL")
JOINTS = [f"{leg}_{kind}_joint" for leg in LEGS for kind in ("hip", "thigh", "calf")]
MAPPING = {f"{leg}_foot": (f"{leg}_thigh_joint", f"{leg}_calf_joint") for leg in LEGS}
CONFIG = MDP.parent / "config/quadruped/tqbot2/standard_walk_env_cfg.py"


def fixture(batch=1, joint_names=None, body_names=None, joint_ids=slice(None), body_ids=slice(None), scale=.1):
    joint_names = list(JOINTS if joint_names is None else joint_names)
    body_names = list(["trunk"] + list(MAPPING) if body_names is None else body_names)
    reference = torch.tensor([0. if "hip" in n else .8 if "thigh" in n else -1.6446 for n in joint_names], dtype=torch.float64).repeat(batch, 1)
    robot = NS(joint_names=joint_names, data=NS(
        joint_pos=reference.clone(), default_joint_pos=reference,
        root_lin_vel_b=torch.zeros(batch, 3, dtype=torch.float64),
        projected_gravity_b=torch.tensor([[0., 0., -1.]], dtype=torch.float64).repeat(batch, 1),
    ))
    sensor = NS(body_names=body_names, cfg=NS(track_air_time=True), data=NS(
        current_air_time=torch.zeros(batch, len(body_names), dtype=torch.float64),
    ))
    command = torch.tensor([[.5, 0., 0.]], dtype=torch.float64).repeat(batch, 1)
    env = NS(device="cpu", scene={"robot": robot, "contact_forces": sensor}, command_manager=NS(get_command=lambda _: command))
    params = dict(command_name="base_velocity", asset_cfg=NS(name="robot", joint_ids=joint_ids),
                  stand_still_scale=5., velocity_threshold=.5, command_threshold=.1,
                  sensor_cfg=NS(name="contact_forces", body_ids=body_ids), foot_joint_names=MAPPING.copy(), swing_scale=scale)
    return env, params, command


def old(env, params):
    return original_penalty(env, **{k: v for k, v in params.items() if k not in ("sensor_cfg", "foot_joint_names", "swing_scale")})


def configured_fixture(batch=1):
    """Apply the actual StandardWalk override to resolved CPU fixture entities."""
    env, params, command = fixture(batch=batch)
    term_cfg = NS(params=params)
    tree = ast.parse(CONFIG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TQBot2StandardWalkRoughEnvCfg")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    nodes = [n for n in init.body if ast.unparse(n).startswith("self.rewards.joint_pos_penalty.")]
    context = {
        "self": NS(rewards=NS(joint_pos_penalty=term_cfg), foot_link_name=".*_foot"),
        "SwingJointPositionPenalty": Penalty,
        "SceneEntityCfg": lambda name, **_: NS(name=name, body_ids=slice(None)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG), "exec"), context)
    return env, term_cfg, command


def set_error(env, name, value, row=slice(None)):
    robot = env.scene["robot"]
    col = robot.joint_names.index(name)
    robot.data.joint_pos[row, col] = robot.data.default_joint_pos[row, col] + value


def airborne(env, foot, row=slice(None), time=.08):
    sensor = env.scene["contact_forces"]
    sensor.data.current_air_time[row, sensor.body_names.index(foot)] = time


class SwingPostureTests(unittest.TestCase):
    def assert_reward(self, env, params, expected):
        actual = Penalty(NS(params=params), env)(env, **params)
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-12)

    def test_all_support_matches_old_for_mixed_commands_and_tilts(self):
        env, params, command = fixture(batch=4)
        env.scene["robot"].data.joint_pos += torch.arange(48, dtype=torch.float64).reshape(4, 12) / 100
        command[:] = torch.tensor([[.5, 0., 0.], [0., 0., 0.], [.1, 0., 0.], [0., 0., .3]])
        env.scene["robot"].data.root_lin_vel_b[2, 0] = .6
        env.scene["robot"].data.projected_gravity_b[:, 2] = torch.tensor([-1., -.35, 0., .4])
        self.assert_reward(env, params, old(env, params))

    def test_scale_one_matches_old_even_with_airborne_feet(self):
        env, params, command = fixture(batch=3, scale=1.)
        env.scene["robot"].data.joint_pos += .2
        env.scene["contact_forces"].data.current_air_time[:] = .1
        command[1:] = 0.
        env.scene["robot"].data.root_lin_vel_b[2, 0] = .6
        self.assert_reward(env, params, old(env, params))

    def test_pure_swing_deviation_reduced_to_tenth(self):
        env, params, _ = fixture()
        set_error(env, "FL_thigh_joint", .3)
        set_error(env, "FL_calf_joint", -.4)
        airborne(env, "FL_foot")
        self.assert_reward(env, params, [.05])

    def test_hip_and_support_coefficients_preserved_in_mixed_norm(self):
        env, params, _ = fixture()
        set_error(env, "FL_thigh_joint", .3)
        set_error(env, "FL_calf_joint", -.4)
        set_error(env, "FL_hip_joint", .12)
        set_error(env, "RR_calf_joint", .16)
        airborne(env, "FL_foot")
        self.assert_reward(env, params, [math.sqrt(.05**2 + .12**2 + .16**2)])

    def test_stop_and_exact_thresholds_preserve_prior_despite_drift(self):
        env, params, command = fixture(batch=4)
        set_error(env, "FL_calf_joint", .6)
        airborne(env, "FL_foot")
        command[:] = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [0., 0., 0.], [.1001, 0., 0.]], dtype=torch.float64)
        env.scene["robot"].data.root_lin_vel_b[:, 0] = torch.tensor([.6, .5, 0., 0.])
        self.assert_reward(env, params, [.6, 3., 3., .06])

    def test_turn_command_allows_swing(self):
        env, params, command = fixture()
        command[:] = torch.tensor([0., 0., .2])
        set_error(env, "FR_thigh_joint", .4)
        airborne(env, "FR_foot")
        self.assert_reward(env, params, [.04])

    def test_name_mapping_survives_reordering_and_selected_subsets(self):
        names = list(reversed(JOINTS))
        bodies = ["RL_foot", "trunk", "FR_foot", "RR_foot", "FL_foot", "FL_thigh"]
        selected = [names.index(n) for n in JOINTS if "hip" not in n]
        env, params, _ = fixture(joint_names=names, body_names=bodies, joint_ids=selected, body_ids=[4, 0, 3, 2])
        set_error(env, "FL_thigh_joint", .3)
        set_error(env, "FL_calf_joint", .4)
        set_error(env, "FR_calf_joint", .12)
        set_error(env, "RL_hip_joint", 1.)  # outside the selected term
        airborne(env, "FL_foot")
        self.assert_reward(env, params, [.13])

    def test_contact_and_command_changes_and_reset_are_read_each_step(self):
        env, params, command = fixture(batch=2)
        set_error(env, "FL_calf_joint", .5)
        term = Penalty(NS(params=params), env)
        airborne(env, "FL_foot", row=0)
        torch.testing.assert_close(term(env, **params), torch.tensor([.05, .5], dtype=torch.float64))
        airborne(env, "FL_foot", row=0, time=0.)
        airborne(env, "FL_foot", row=1)
        torch.testing.assert_close(term(env, **params), torch.tensor([.5, .05], dtype=torch.float64))
        command[:] = 0.
        torch.testing.assert_close(term(env, **params), torch.tensor([2.5, 2.5], dtype=torch.float64))

    def test_inputs_not_mutated_and_upright_gate_preserved(self):
        env, params, _ = fixture(batch=3)
        set_error(env, "FL_calf_joint", .6)
        airborne(env, "FL_foot")
        robot = env.scene["robot"]
        robot.data.projected_gravity_b[:, 2] = torch.tensor([-.7, -.35, .1], dtype=torch.float64)
        before = (robot.data.joint_pos.clone(), robot.data.default_joint_pos.clone())
        self.assert_reward(env, params, [.06, .03, 0.])
        torch.testing.assert_close(robot.data.joint_pos, before[0])
        torch.testing.assert_close(robot.data.default_joint_pos, before[1])

    def test_invalid_scale_rejected(self):
        for scale in (-.1, 1.1, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                env, params, _ = fixture(scale=scale)
                with self.assertRaisesRegex(ValueError, "swing_scale"):
                    Penalty(NS(params=params), env)

    def test_missing_tracking_or_names_and_duplicate_joints_rejected(self):
        cases = ("tracking", "foot", "joint", "duplicate", "empty", "selection")
        for case in cases:
            with self.subTest(case=case):
                env, params, _ = fixture()
                if case == "tracking":
                    env.scene["contact_forces"].cfg.track_air_time = False
                elif case == "foot":
                    params["sensor_cfg"].body_ids = [1, 2, 3]
                elif case == "joint":
                    params["foot_joint_names"]["FL_foot"] = ("missing_joint",)
                elif case == "duplicate":
                    params["foot_joint_names"]["FL_foot"] = ("FR_thigh_joint",)
                elif case == "empty":
                    params["foot_joint_names"] = {}
                else:
                    params["asset_cfg"].joint_ids = [0, 0] + list(range(1, 12))
                with self.assertRaises(ValueError):
                    Penalty(NS(params=params), env)

    def test_configured_swing_exempts_hip_thigh_and_calf(self):
        env, cfg, _ = configured_fixture(batch=3)
        for row, kind in enumerate(("hip", "thigh", "calf")):
            set_error(env, f"FL_{kind}_joint", .6, row=row)
        airborne(env, "FL_foot")
        actual = cfg.weight * cfg.func(cfg, env)(env, **cfg.params)
        torch.testing.assert_close(actual, torch.zeros(3, dtype=torch.float64))

    def test_configured_mixed_support_cost_and_all_swing_zero(self):
        env, cfg, _ = configured_fixture(batch=3)
        set_error(env, "FL_hip_joint", .5)
        set_error(env, "FL_thigh_joint", .6)
        set_error(env, "FL_calf_joint", .8)
        set_error(env, "RR_thigh_joint", .3)
        set_error(env, "RR_calf_joint", .4)
        airborne(env, "FL_foot", row=0)
        for leg in LEGS:
            airborne(env, f"{leg}_foot", row=2)
        actual = cfg.weight * cfg.func(cfg, env)(env, **cfg.params)
        # Row 0: only the support leg's 3-4-5 triangle remains. Row 1: all
        # support, so exactly 20% of the old negative-weight penalty. Row 2: no
        # remaining posture coordinates, hence zero even for hip deviations.
        expected = torch.tensor([-.1, -.2 * old(env, cfg.params)[1].item(), 0.], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_configured_stop_preserves_gating_at_twenty_percent_weight(self):
        env, cfg, command = configured_fixture(batch=2)
        set_error(env, "FL_calf_joint", .6)
        airborne(env, "FL_foot")
        command[:] = 0.
        env.scene["robot"].data.root_lin_vel_b[1, 0] = .6
        actual = cfg.weight * cfg.func(cfg, env)(env, **cfg.params)
        torch.testing.assert_close(actual, torch.tensor([-.6, -.12], dtype=torch.float64), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
