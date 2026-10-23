"""CPU contract tests of production terrain clearance, without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/foot_ground.py"


class ManagerTermBase:
    def __init__(self, cfg, env):
        pass


def load_production():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in ("terrain_surface_at_feet", "SwingFootClearance")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                        type_ignores=[])
    namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerTermBase,
                 "RayCaster": NS(meshes={"/terrain": object()})}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


class Scene(dict):
    pass


def fixture(batch=1, terrain_type="plane", asset_ids=None, sensor_ids=None):
    dtype = torch.float64
    names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    position = torch.zeros(batch, 4, 3, dtype=dtype)
    position[..., 2] = .062  # 5 cm foot-sphere clearance with a 1.2 cm radius.
    velocity = torch.zeros_like(position)
    velocity[..., 0] = .5
    robot = NS(body_names=names, data=NS(
        body_link_pos_w=position, body_link_lin_vel_w=velocity,
        root_link_lin_vel_w=torch.zeros(batch, 3, dtype=dtype),
        root_link_pos_w=torch.tensor([[0., 0., .25]], dtype=dtype).repeat(batch, 1),
        projected_gravity_b=torch.tensor([[0., 0., -1.]], dtype=dtype).repeat(batch, 1),
    ))
    sensor = NS(body_names=names.copy(), cfg=NS(track_air_time=True),
                data=NS(current_air_time=torch.full((batch, 4), .1, dtype=dtype)))
    scene = Scene(robot=robot, contact_forces=sensor,
                  height_scanner=NS(cfg=NS(mesh_prim_paths=["/terrain"])))
    scene.terrain = NS(cfg=NS(terrain_type=terrain_type))
    command = torch.tensor([[.5, 0., 0.]], dtype=dtype).repeat(batch, 1)
    env = NS(scene=scene, command_manager=NS(get_command=lambda _: command))
    params = dict(command_name="base_velocity", target_clearance=.05, error_scale=.05, tanh_mult=2., foot_radius=.012,
                  asset_cfg=NS(name="robot", body_ids=list(range(4)) if asset_ids is None else asset_ids),
                  sensor_cfg=NS(name="contact_forces", body_ids=list(range(4)) if sensor_ids is None else sensor_ids))
    return env, params, command


class SwingClearanceTests(unittest.TestCase):
    def setUp(self):
        self.production = load_production()
        self.Cost = self.production["SwingFootClearance"]

    def evaluate(self, env, params):
        return self.Cost(NS(params=params), env)(env, **params)

    def assert_cost(self, env, params, expected):
        torch.testing.assert_close(self.evaluate(env, params), torch.as_tensor(expected, dtype=torch.float64),
                                   rtol=1e-11, atol=1e-12)

    def test_plane_penalizes_only_clearance_below_five_cm(self):
        env, params, _ = fixture(batch=5)
        heights = torch.tensor([.02, .04, .05, .051, .08], dtype=torch.float64)
        env.scene["robot"].data.body_link_pos_w[..., 2] = heights[:, None] + .012
        self.assert_cost(env, params, [4 * .6**2 * math.tanh(1.),
                                       4 * .2**2 * math.tanh(1.), 0., 0., 0.])

    def test_high_feet_do_not_add_cost_or_cancel_low_foot_shortfall(self):
        env, params, _ = fixture()
        env.scene["robot"].data.body_link_pos_w[0, :, 2] = torch.tensor(
            [.032, .092, .112, .062], dtype=torch.float64)
        self.assert_cost(env, params, [.6**2 * math.tanh(1.)])

    def test_root_height_does_not_fake_clearance(self):
        env, params, _ = fixture(batch=2)
        env.scene["robot"].data.body_link_pos_w[..., 2] = .025
        env.scene["robot"].data.root_link_pos_w[:, 2] = torch.tensor([.25, .14])
        cost = 4 * ((.025 - .012 - .05) / .05)**2 * math.tanh(1.)
        self.assert_cost(env, params, [cost, cost])

    def test_slope_local_normal_distance_radius_and_tangent_speed(self):
        env, params, _ = fixture(batch=3, terrain_type="generator")
        theta = math.radians(30)
        normal = torch.tensor([-math.sin(theta), 0., math.cos(theta)], dtype=torch.float64)
        tangent = torch.tensor([math.cos(theta), 0., math.sin(theta)], dtype=torch.float64)
        data = env.scene["robot"].data
        data.body_link_pos_w[..., 0] = .7
        data.body_link_pos_w[..., 2] = .7 * math.tan(theta) + .062 / math.cos(theta)
        data.body_link_pos_w[1, :, 2] += .02 / math.cos(theta)
        data.body_link_pos_w[2, :, 2] -= .02 / math.cos(theta)
        # An added normal velocity must not alter the tangent-speed weighting.
        data.body_link_lin_vel_w[:] = .5 * tangent + 2. * normal

        def raycast(starts, directions, mesh, max_dist, return_normal):
            torch.testing.assert_close(starts, data.body_link_pos_w + torch.tensor([0., 0., 1.]))
            self.assertTrue((directions[..., 2] == -1.).all())
            self.assertEqual(max_dist, 5.)
            self.assertTrue(return_normal)
            hits = starts.clone()
            hits[..., 2] = hits[..., 0] * math.tan(theta)
            # Test the production helper's upward-normal orientation too.
            return hits, None, -normal.expand_as(hits).clone(), None

        self.production["raycast_mesh"] = raycast
        self.assert_cost(env, params, [0., 0., 4 * (.02 / .05)**2 * math.tanh(1.)])

    def test_raised_surface_uses_local_height_not_world_origin(self):
        env, params, _ = fixture(terrain_type="generator")
        env.scene["robot"].data.body_link_pos_w[..., 2] += .3

        def raycast(starts, directions, mesh, **kwargs):
            hits = starts.clone()
            hits[..., 2] = .3
            return hits, None, -directions, None

        self.production["raycast_mesh"] = raycast
        self.assert_cost(env, params, [0.])

    def test_stance_stop_turn_and_upright_gates(self):
        env, params, command = fixture(batch=5)
        env.scene["robot"].data.body_link_pos_w[..., 2] = .032
        env.scene["contact_forces"].data.current_air_time[0] = 0.
        command[1] = 0.
        command[2] = torch.tensor([0., 0., .3])
        env.scene["robot"].data.projected_gravity_b[3, 2] = -.35
        env.scene["robot"].data.projected_gravity_b[4, 2] = .5
        cost = 4 * (.03 / .05)**2 * math.tanh(1.)
        self.assert_cost(env, params, [0., 0., cost, cost / 2, 0.])

    def test_relative_tangent_motion_gate(self):
        env, params, _ = fixture(batch=3)
        data = env.scene["robot"].data
        data.body_link_pos_w[..., 2] = .032
        data.root_link_lin_vel_w[0, 0] = .5  # common translation does not count as swing motion
        data.body_link_lin_vel_w[1] = torch.tensor([0., 0., .8])  # purely normal motion
        data.body_link_lin_vel_w[2] = torch.tensor([.5, 0., .8])
        self.assert_cost(env, params, [0., 0., 4 * (.03 / .05)**2 * math.tanh(1.)])

    def test_invalid_raycast_ignored_without_nan(self):
        env, params, _ = fixture(terrain_type="generator")
        env.scene["robot"].data.body_link_pos_w[..., 2] = .032

        def raycast(starts, directions, mesh, **kwargs):
            hits = starts.clone()
            hits[..., 2] = 0.
            hits[:, 1, :] = float("inf")
            normals = -directions
            normals[:, 2, :] = float("nan")
            return hits, None, normals, None

        self.production["raycast_mesh"] = raycast
        self.assert_cost(env, params, [2 * (.03 / .05)**2 * math.tanh(1.)])

    def test_selected_indices_follow_matching_sensor_order(self):
        env, params, _ = fixture(asset_ids=[2, 0], sensor_ids=[1, 3])
        env.scene["contact_forces"].body_names = ["FL_foot", "RR_foot", "RL_foot", "FR_foot"]
        env.scene["robot"].data.body_link_pos_w[:, 2, 2] = .032
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .042
        env.scene["contact_forces"].data.current_air_time[:, 1] = 0.
        self.assert_cost(env, params, [(.02 / .05)**2 * math.tanh(1.)])

    def test_error_scale_changes_strength_without_moving_height_target(self):
        env, params, _ = fixture(batch=2)
        env.scene["robot"].data.body_link_pos_w[1, :, 2] -= .03
        original = self.evaluate(env, params)
        params["error_scale"] = .10
        weaker = self.evaluate(env, params)
        torch.testing.assert_close(weaker, original / 4)
        self.assertEqual(weaker[0].item(), 0.)

    def test_configured_two_cm_swing_has_calibrated_cost(self):
        env, params, _ = fixture()
        env.scene["robot"].data.body_link_pos_w[:, 0, 2] = .032
        env.scene["contact_forces"].data.current_air_time[:, 1:] = 0.
        config = SOURCE.parent.parent / "config/quadruped/tqbot2/standard_walk_env_cfg.py"
        tree = ast.parse(config.read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(ast.unparse(t) == "self.rewards.feet_ground_clearance" for t in node.targets))
        args = {kw.arg: kw.value for kw in assignment.value.keywords}
        weight = ast.literal_eval(args["weight"])
        configured = dict(zip((ast.literal_eval(k) for k in args["params"].keys), args["params"].values))
        for name in ("target_clearance", "error_scale", "tanh_mult", "foot_radius"):
            params[name] = ast.literal_eval(configured[name])
        actual = weight * self.evaluate(env, params)
        # One foot at 2 cm pays 0.18 with full speed weight, rather than 0.0045.
        torch.testing.assert_close(actual, torch.tensor([-.18 * math.tanh(1.)], dtype=torch.float64))
        self.assertEqual(params["target_clearance"], .05)

    def test_slice_selection_supported(self):
        env, params, _ = fixture(asset_ids=slice(None), sensor_ids=slice(None))
        self.assert_cost(env, params, [0.])

    def test_mismatched_order_or_missing_phase_tracking_rejected(self):
        env, params, _ = fixture(sensor_ids=[1, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "ordering mismatch"):
            self.Cost(NS(params=params), env)
        env, params, _ = fixture()
        env.scene["contact_forces"].cfg.track_air_time = False
        with self.assertRaisesRegex(ValueError, "phase tracking"):
            self.Cost(NS(params=params), env)

    def test_live_contact_and_commands_without_stale_state(self):
        env, params, command = fixture()
        env.scene["robot"].data.body_link_pos_w[..., 2] = .032
        term = self.Cost(NS(params=params), env)
        self.assertGreater(term(env, **params).item(), 0.)
        env.scene["contact_forces"].data.current_air_time[:] = 0.
        self.assertEqual(term(env, **params).item(), 0.)
        env.scene["contact_forces"].data.current_air_time[:] = .1
        command[:] = 0.
        self.assertEqual(term(env, **params).item(), 0.)

    def test_invalid_parameters_rejected(self):
        for name, values in (("target_clearance", [0., -.1, float("nan")]),
                             ("error_scale", [0., -.05, float("inf"), float("nan")]),
                             ("tanh_mult", [0., -1., float("inf")]),
                             ("foot_radius", [-.1, float("inf")])):
            for value in values:
                with self.subTest(name=name, value=value):
                    env, params, _ = fixture()
                    params[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        self.Cost(NS(params=params), env)


if __name__ == "__main__":
    unittest.main()
