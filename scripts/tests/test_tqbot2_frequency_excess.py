"""CPU contact-history and one-sided cadence tests, without starting Isaac Sim."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp"
SOURCE = MDP / "frequency_excess.py"
DTYPE = torch.float64
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


def production_symbols():
    tree = ast.parse(SOURCE.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *definitions,
    ], type_ignores=[])
    namespace = {"torch": torch, "math": math, "ManagerTermBase": ManagerTermBase}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["frequency_excess_cost"], namespace["FootFrequencyExcess"]


cost, Reward = production_symbols()


class Fixture:
    def __init__(self, batch=1, body_names=FEET, params=None, history_length=4, update_period=.005):
        self.command = torch.zeros(batch, 3, dtype=DTYPE)
        self.command[:, 0] = .5
        gravity = torch.zeros_like(self.command)
        gravity[:, 2] = -1.
        self.asset = NS(data=NS(projected_gravity_b=gravity))
        self.sensor = NS(
            body_names=body_names, cfg=NS(history_length=history_length, update_period=update_period),
            data=NS(net_forces_w_history=torch.zeros(batch, history_length, len(body_names), 3, dtype=DTYPE)),
        )
        self.env = NS(
            scene={"robot": self.asset, "foot_cycle_contacts": self.sensor},
            command_manager=NS(get_command=lambda name: self.command),
            common_step_counter=0, step_dt=.02, physics_dt=.005,
        )
        self.params = dict(
            threshold_frequency=2., frequency_scale=2., command_name="base_velocity", command_threshold=.1,
            confirmation_time=.01, force_threshold=1., asset_cfg=NS(name="robot"),
            sensor_cfg=NS(name="foot_cycle_contacts", body_ids=[body_names.index(name) for name in FEET]),
        )
        self.params.update(params or {})
        self.term = Reward(NS(params=self.params), self.env)
        self.ticks = 0

    def evaluate(self):
        return self.term(self.env, **self.params)

    def push_forces(self, forces):
        history = self.sensor.data.net_forces_w_history
        history = history.roll(1, dims=1)
        history[:, 0] = forces
        self.sensor.data.net_forces_w_history = history
        self.ticks += 1
        if self.ticks % 4 == 0:
            self.env.common_step_counter += 1
            return self.evaluate().clone()

    def advance(self, contact, ticks):
        contacts = torch.as_tensor(contact, dtype=DTYPE).expand(self.command.shape[0], 4)
        forces = torch.zeros_like(self.sensor.data.net_forces_w_history[:, 0])
        forces[:, self.term.foot_ids, 2] = contacts * 20.
        result = None
        for _ in range(ticks):
            value = self.push_forces(forces)
            if value is not None:
                result = value
        return result

    def periodic(self, period_ticks=50, total_ticks=800, stance_ticks=None, phase=0):
        stance_ticks = period_ticks // 2 if stance_ticks is None else stance_ticks
        values = []
        for k in range(total_ticks):
            value = self.advance((k + phase) % period_ticks < stance_ticks, 1)
            if value is not None:
                values.append(value)
        return torch.stack(values)


class ExcessKernelTests(unittest.TestCase):
    def test_weighted_scale_and_one_sided_threshold(self):
        frequencies = torch.tensor([1., 2., 3., 4., 6.], dtype=DTYPE)
        periods = frequencies.reciprocal()[:, None].repeat(1, 4)
        actual = -.5 * cost(periods, torch.ones_like(periods, dtype=torch.bool), 2., 2.)
        torch.testing.assert_close(actual, torch.tensor([0., 0., -.125, -.5, -1.5], dtype=DTYPE))

    def test_fixed_four_foot_denominator_and_missing_history(self):
        periods = torch.tensor([[.25, float("nan"), 0., float("inf")]], dtype=DTYPE)
        measured = torch.tensor([[True, False, False, False]])
        self.assertEqual(cost(periods, measured, 2., 2.).item(), .25)
        self.assertEqual(cost(periods, torch.zeros_like(measured), 2., 2.).item(), 0.)

    def test_continuity_at_quadratic_to_linear_transition(self):
        frequencies = torch.tensor([3.999999, 4., 4.000001], dtype=DTYPE)
        periods = frequencies.reciprocal()[:, None].repeat(1, 4)
        values = cost(periods, torch.ones_like(periods, dtype=torch.bool), 2., 2.)
        torch.testing.assert_close(values, torch.ones_like(values), atol=1.1e-6, rtol=0.)
        self.assertTrue((values[1:] > values[:-1]).all())


class ExcessContactTests(unittest.TestCase):
    def test_actual_config_connects_four_substeps_to_negative_cost(self):
        config = MDP.parent / "config/quadruped/tqbot2/standard_walk_env_cfg.py"
        tree = ast.parse(config.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TQBot2StandardWalkRoughEnvCfg")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        targets = ("self.scene.foot_cycle_contacts", "self.rewards.feet_frequency_excess")
        nodes = [n for n in init.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in targets]
        configured = NS(scene=NS(), rewards=NS(), sim=NS(dt=.005), decimation=4, foot_link_name=".*_foot")
        namespace = dict(self=configured, ContactSensorCfg=lambda **kw: NS(**kw), RewTerm=lambda **kw: NS(**kw),
                         SceneEntityCfg=lambda name, **kw: NS(name=name, **kw), FootFrequencyExcess=Reward)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(config), "exec"), namespace)
        cfg = configured.rewards.feet_frequency_excess
        self.assertEqual(cfg.weight, -.5)
        self.assertIs(cfg.func, Reward)
        self.assertEqual(cfg.params["sensor_cfg"].name, "foot_cycle_contacts")
        sensor = configured.scene.foot_cycle_contacts
        self.assertEqual(sensor.history_length, 4)
        f = Fixture(history_length=sensor.history_length, update_period=sensor.update_period,
                    params={k: v for k, v in cfg.params.items() if k not in ("sensor_cfg", "asset_cfg")})
        f.periodic(50)
        self.assertAlmostEqual(cfg.weight * f.evaluate().item(), -.5)

    def test_target_and_fast_cadence_are_measured_from_full_cycles(self):
        for ticks, expected in ((100, 0.), (80, .0625), (50, 1.), (32, 3.25)):
            with self.subTest(ticks=ticks):
                f = Fixture()
                values = f.periodic(ticks)
                torch.testing.assert_close(values[-50:], torch.full_like(values[-50:], expected))
                torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, ticks * .005))

    def test_sub_80ms_cycles_and_all_control_alignment_offsets_are_kept(self):
        for ticks in (4, 5, 12):
            for phase in range(4):
                with self.subTest(ticks=ticks, phase=phase):
                    f = Fixture()
                    f.periodic(ticks, total_ticks=240, stance_ticks=2, phase=phase)
                    self.assertTrue(f.term.measured.all())
                    torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, ticks * .005))
                    self.assertGreater(f.evaluate().item(), 1.)
                    self.assertGreaterEqual(f.term.completed_cycles.min().item(), 240 // ticks - 3)

    def test_duty_factor_does_not_set_frequency(self):
        for stance in (2, 15, 40, 48):
            f = Fixture()
            f.periodic(50, stance_ticks=stance)
            self.assertAlmostEqual(f.evaluate().item(), 1.)

    def test_initial_contact_and_first_air_to_contact_event_have_no_period(self):
        f = Fixture()
        self.assertEqual(f.advance(True, 200).item(), 0.)
        f.advance(False, 40)
        self.assertEqual(f.advance(True, 60).item(), 0.)
        self.assertFalse(f.term.measured.any())
        f.advance(False, 40)
        f.advance(True, 4)
        self.assertTrue(f.term.measured.all())
        torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, .5))

    def test_single_sample_dropout_and_contact_spike_do_not_erase_fast_cost(self):
        for contact in (True, False):
            f = Fixture()
            f.periodic()
            f.advance(contact, 40)
            old_periods = f.term.periods.clone()
            old_cycles = f.term.completed_cycles.clone()
            f.advance(not contact, 1)
            f.advance(contact, 3)
            torch.testing.assert_close(f.term.periods, old_periods)
            torch.testing.assert_close(f.term.completed_cycles, old_cycles)
            self.assertAlmostEqual(f.evaluate().item(), 1.)

    def test_waiting_never_clears_cost_but_a_new_slow_cycle_does(self):
        f = Fixture()
        f.periodic()
        self.assertAlmostEqual(f.advance(False, 1200).item(), 1.)
        f.periodic(100, total_ticks=400)
        self.assertEqual(f.evaluate().item(), 0.)
        torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, .5))

    def test_one_fast_foot_contributes_one_quarter_without_affecting_others(self):
        f = Fixture()
        for k in range(800):
            f.advance([k % 50 < 25, k % 100 < 50, True, True], 1)
        self.assertAlmostEqual(f.evaluate().item(), .25)
        torch.testing.assert_close(f.term.measured, torch.tensor([[True, True, False, False]]))

    def test_stopping_clears_history_and_restart_needs_new_cycles(self):
        f = Fixture()
        f.periodic()
        f.command.zero_()
        self.assertEqual(f.advance(True, 4).item(), 0.)
        self.assertFalse(f.term.measured.any())
        f.command[:, 0] = .5
        self.assertEqual(f.advance(True, 4).item(), 0.)

    def test_selected_order_duplicate_calls_and_partial_reset(self):
        f = Fixture(batch=2, body_names=("extra", *reversed(FEET)))
        f.periodic()
        periods, counts = f.term.periods.clone(), f.term.completed_cycles.clone()
        for _ in range(3):
            f.evaluate()
        torch.testing.assert_close(f.term.periods, periods)
        torch.testing.assert_close(f.term.completed_cycles, counts)
        f.term.reset(torch.tensor([0]))
        torch.testing.assert_close(f.evaluate(), torch.tensor([0., 1.], dtype=DTYPE))
        f.advance(True, 4)
        self.assertFalse(f.term.measured[0].any())
        self.assertTrue(f.term.measured[1].all())
        f.term.reset()
        self.assertEqual(f.evaluate().sum().item(), 0.)

    def test_upright_factor_does_not_corrupt_frequency_history(self):
        f = Fixture(batch=3)
        f.periodic()
        f.asset.data.projected_gravity_b[:, 2] = torch.tensor([-.7, -.35, 1.])
        torch.testing.assert_close(f.evaluate(), torch.tensor([1., .5, 0.], dtype=DTYPE))
        self.assertTrue(f.term.measured.all())

    def test_large_global_step_keeps_substep_period_precision(self):
        f = Fixture()
        f.env.common_step_counter = 100_000_000
        f.term._last_processed_step = f.env.common_step_counter
        f.periodic(5, total_ticks=240, stance_ticks=2)
        torch.testing.assert_close(f.term.periods, torch.full_like(f.term.periods, .025))

    def test_bad_forces_and_missing_control_step_are_reported(self):
        for bad in (float("nan"), float("inf")):
            f = Fixture()
            f.sensor.data.net_forces_w_history[0, 0, 0, 0] = bad
            f.env.common_step_counter = 1
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                f.evaluate()
        f = Fixture()
        f.env.common_step_counter = 2
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            f.evaluate()

    def test_insufficient_history_and_invalid_config_are_rejected(self):
        for kwargs in ({"history_length": 3}, {"update_period": .02}):
            with self.assertRaises(ValueError):
                Fixture(**kwargs)
        for name, value in (("confirmation_time", .005), ("confirmation_time", .012),
                            ("threshold_frequency", 0.), ("frequency_scale", float("nan")),
                            ("force_threshold", -1.), ("command_threshold", -.1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                Fixture(params={name: value})


if __name__ == "__main__":
    unittest.main()
