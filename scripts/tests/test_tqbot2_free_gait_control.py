"""CPU configuration/guard tests; no Isaac application, CUDA or robot simulation."""
import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/tqbot2"
package = ModuleType("tqbot2_cpu_control")
package.__path__ = [str(CONFIG)]
sys.modules[package.__name__] = package
policy = importlib.import_module("tqbot2_cpu_control.free_gait_policy")
fresh = importlib.import_module("tqbot2_cpu_control.fresh_training")
spec = importlib.util.spec_from_file_location("control_launcher", ROOT / "scripts/TQBot2_train_control.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def fixture():
    contract = json.loads((ROOT / "docs/experiments/TQBot2_V4_reward_contract.json").read_text())
    rewards = NS(**{k: NS(weight=float(v["weight"]), func=v["func"], params=deepcopy(v["params"]))
                    for k, v in contract["rewards"].items()})
    for k in policy.INACTIVE_GUARDS:
        setattr(rewards, k, None)
    return NS(rewards=rewards, timing=(.005, 4), terminations=contract["terminations"],
              commands=contract["commands"], protection=NS(pd=(20., .5), torque_margin=.9, collision=True))


class ControlTests(unittest.TestCase):
    def test_exact_seven_active_terms_removed(self):
        cfg = fixture()
        self.assertEqual(policy.active_reward_names(cfg.rewards), policy.V4_ACTIVE)
        policy.apply_free_gait(cfg.rewards)
        self.assertEqual(policy.active_reward_names(cfg.rewards), policy.V4_ACTIVE - policy.REMOVED.keys())
        self.assertEqual(len(policy.REMOVED), 7)
        self.assertEqual(len(policy.active_reward_names(cfg.rewards)), 14)

    def test_retained_rewards_and_nonreward_constraints_are_identical(self):
        cfg, before = fixture(), fixture()
        policy.apply_free_gait(cfg.rewards)
        for name in policy.V4_ACTIVE - policy.REMOVED.keys():
            self.assertEqual(vars(getattr(cfg.rewards, name)), vars(getattr(before.rewards, name)))
        for field in ("timing", "terminations", "commands"):
            self.assertEqual(getattr(cfg, field), getattr(before, field))
        self.assertEqual(vars(cfg.protection), vars(before.protection))

    def test_misleading_contact_event_and_swing_posture_terms_are_removed(self):
        cfg = fixture()
        self.assertIn("feet_contact_without_cmd", policy.REMOVED)
        self.assertIn("joint_pos_penalty", policy.REMOVED)
        policy.apply_free_gait(cfg.rewards)
        self.assertIsNone(cfg.rewards.feet_contact_without_cmd)
        self.assertIsNone(cfg.rewards.joint_pos_penalty)
        self.assertIsNotNone(cfg.rewards.feet_slide)
        self.assertIsNotNone(cfg.rewards.contact_forces)
        self.assertIsNotNone(cfg.rewards.stand_still)

    def test_unreviewed_active_reward_is_rejected(self):
        cfg = fixture()
        cfg.rewards.new_phase_cost = NS(weight=1.)
        with self.assertRaises(ValueError):
            policy.apply_free_gait(cfg.rewards)

    def test_v4_source_matches_frozen_original_modulo_class_name(self):
        source = (CONFIG / "reward_v4_env_cfg.py").read_text().replace(
            "class TQBot2RewardV4EnvCfg(", "class TQBot2StandardWalkRoughEnvCfg(")
        expected = "0295159f9dcbbb70c8fd59b8f24985c2067ce93672d988b07037c766d24c57c7"
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)
        ast.parse((CONFIG / "free_gait_env_cfg.py").read_text())

    def test_fresh_launcher_has_disabled_load_selectors(self):
        args = NS(variant="free", num_envs=1024, iterations=12000, seed=42,
                  device="cuda:0", run_name="portable_test")
        command = launcher.build_command(args)
        self.assertNotIn("--resume", command)
        self.assertNotIn("--warm_start", command)
        self.assertEqual(command[command.index("--load_run")+1], "__NO_RESUME__")
        self.assertEqual(command[command.index("--checkpoint")+1], "__NO_CHECKPOINT__")
        self.assertIn("agent.resume=false", command)
        self.assertIn("1024", command)
        self.assertIn(launcher.TASKS["free"], command)

    def test_fresh_entry_calls_guard_before_any_checkpoint_load(self):
        text = (ROOT / "scripts/reinforcement_learning/rsl_rl/train_tqbot2_ablation.py").read_text()
        self.assertLess(text.index("    audit_fresh_start("), text.index("        runner.load("))
        ast.parse(text)

    def test_audit_refuses_resume_and_nonempty_optimizer_before_writing(self):
        runner = NS(current_learning_iteration=0, alg=NS(optimizer=NS(state={})))
        with self.assertRaises(ValueError):
            fresh.audit_fresh_start(runner, None, NS(resume=True), NS(resume=True, warm_start=None), ".")
        runner.alg.optimizer.state = {"unexpected": 1}
        with self.assertRaises(ValueError):
            fresh.audit_fresh_start(runner, None,
                                    NS(resume=False, load_run="__NO_RESUME__", load_checkpoint="__NO_CHECKPOINT__"),
                                    NS(resume=False, warm_start=None), ".")


if __name__ == "__main__":
    unittest.main()
