"""Validate, briefly train, and evaluate the blind-box skill in one Isaac Sim session."""

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--walking_checkpoint", type=Path)
parser.add_argument("--curriculum_state", type=Path)
parser.add_argument("--baseline_report", type=Path, help="Reuse a matching, measured walking baseline")
parser.add_argument("--history_adapter_only", action="store_true")
parser.add_argument("--validation_only", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_heights", type=float, nargs="+", default=[0.04, 0.08, 0.12, 0.16])
parser.add_argument("--trials_per_env", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2 import blind_box_mdp as box
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.agents.blind_box_ppo_cfg import (
    TQBot2BlindBoxPPORunnerCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.agents.blind_box_warm_start import (
    lift_walking_actor,
    train_history_adapter_only,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.blind_box_geometry import feet_on_top
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.blind_box_env_cfg import (
    TQBot2BlindBoxEnvCfg,
)


def validate(env):
    actor_terms = env.observation_manager.active_terms["policy"]
    assert actor_terms == ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos",
                           "joint_vel", "actions", "joint_tracking_error"], actor_terms
    assert env.cfg.observations.policy.history_length == 12
    assert env.observation_manager.group_obs_dim["policy"] == (684,)
    obs, _ = env.reset()
    assert torch.isfinite(obs["policy"]).all()
    # Change privileged terrain metadata: actor inputs must remain exactly equal.
    saved = env.scene.terrain.heights.clone()
    for term in actor_terms:
        cfg = getattr(env.observation_manager.cfg.policy, term)
        before = cfg.func(env, **cfg.params).clone()
        env.scene.terrain.heights += 0.123
        after = cfg.func(env, **cfg.params)
        env.scene.terrain.heights.copy_(saved)
        assert torch.equal(before, after), f"Terrain leaks into actor term {term}"
    for _ in range(100):
        obs, reward, _, _, _ = env.step(torch.randn(env.num_envs, 12, device=env.device) * 0.5)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all()
    # Partial history reset must not retain a previous episode's joint feedback.
    ids = torch.tensor([0], device=env.device)
    env.observation_manager.reset(env_ids=ids)
    reset_obs = env.observation_manager.compute(update_history=True)
    joint_history = reset_obs["policy"][0, 108:252].reshape(12, 12)
    assert torch.equal(joint_history, joint_history[-1].expand_as(joint_history))
    # Robot joint command clipping is applied after scaling and offset.
    limits = env.scene["robot"].data.soft_joint_pos_limits
    term = env.action_manager.get_term("joint_pos")
    ids = term._joint_ids
    term.process_actions(torch.full((env.num_envs, 12), 1e3, device=env.device))
    assert (term.processed_actions <= limits[:, ids, 1] + 1e-5).all()
    term.process_actions(torch.full((env.num_envs, 12), -1e3, device=env.device))
    assert (term.processed_actions >= limits[:, ids, 0] - 1e-5).all()
    # Positive physical control: a standing robot placed on top should succeed;
    # the same standing robot on the approach ground must not succeed.
    env.cfg.box_evaluation_height = 0.1
    env.reset()
    zeros = torch.zeros(env.num_envs, 12, device=env.device)
    for _ in range(50):
        env.step(zeros)
        assert not env.termination_manager.get_term("box_success").any()
    robot = env.scene["robot"]
    root = robot.data.root_state_w.clone()
    root[:, 0] = env.scene.env_origins[:, 0] + 1.6
    root[:, 2] += 0.1
    root[:, 7:] = 0
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    passed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for _ in range(100):
        env.step(zeros)
        passed |= env.termination_manager.get_term("box_success")
    assert passed.all(), f"Known standing-on-top control failed in {int((~passed).sum())} environments"
    env.cfg.box_evaluation_height = None
    env.reset()
    print("PASS: 684 blind actor inputs; no geometry leakage; 100 finite physics steps; joint limits", flush=True)
    return {"policy_terms": actor_terms, "actor_observations": 684, "history_frames": 12,
            "finite_physics_steps": 100, "geometry_leakage_check": "passed", "joint_target_limits": "passed",
            "partial_history_reset": "passed", "physical_top_positive_control": "passed"}


@torch.inference_mode()
def evaluate(env, policy, heights=None):
    results = []
    for height in (args.eval_heights if heights is None else heights):
        env.cfg.box_evaluation_height = height
        counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        outcomes = {key: 0 for key in ("success", "failure", "timeout")}
        stopped_count = 0
        top_counts = {key: 0 for key in ("reached_two_top_feet", "reached_four_top_feet", "reached_supported_top")}
        max_top = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        supported_seen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        eval_seed = args.seed + 1000 + round(height * 1000)
        torch.manual_seed(eval_seed)
        obs, _ = env.reset()
        begin = time.monotonic()
        for _ in range(env.max_episode_length * args.trials_per_env + 1):
            action = policy(obs) if policy else torch.zeros(env.num_envs, 12, device=env.device)
            obs, _, terminated, truncated, _ = env.step(action)
            s, terrain = box.state(env), env.scene.terrain
            top = feet_on_top(box.foot_positions(env), terrain.heights, terrain.lengths, s.radii).sum(dim=1)
            max_top = torch.maximum(max_top, top)
            contacts = env.scene["contact_forces"].data.net_forces_w[:, s.sensor_ids, 2] > 2.0
            supported_seen |= (top == 4) & (contacts.sum(dim=1) >= 3) & (
                env.scene["robot"].data.projected_gravity_b[:, 2] < -0.85
            )
            selected = (terminated | truncated) & (counts < args.trials_per_env)
            success = env.termination_manager.get_term("box_success") & selected
            failure = env.termination_manager.get_term("box_failure") & selected & ~success
            outcomes["success"] += int(success.sum())
            stopped_count += int((selected & box.state(env).last_stopped).sum())
            outcomes["failure"] += int(failure.sum())
            outcomes["timeout"] += int((selected & ~success & ~failure).sum())
            top_counts["reached_two_top_feet"] += int((selected & (max_top >= 2)).sum())
            top_counts["reached_four_top_feet"] += int((selected & (max_top == 4)).sum())
            top_counts["reached_supported_top"] += int((selected & supported_seen).sum())
            max_top[selected] = 0
            supported_seen[selected] = False
            counts[selected] += 1
            if (counts >= args.trials_per_env).all():
                break
        total = int(counts.sum())
        results.append({"height_m": height, "episodes": total, "first_attempt_seed": eval_seed,
                        **outcomes, **top_counts,
                        "stopped_on_top": stopped_count,
                        "success_rate": outcomes["success"] / max(total, 1),
                        "wall_seconds": round(time.monotonic() - begin, 2)})
        print(f"EVAL {results[-1]}", flush=True)
    env.cfg.box_evaluation_height = None
    return results


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg = TQBot2BlindBoxEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.log_dir = str(args.output.resolve())
    cfg.sim.device = args.device
    if args.curriculum_state:
        cfg.box_curriculum_resume_path = str(args.curriculum_state.resolve())
    # The default buffers target thousands of environments. Reserve capacities
    # proportional to this experiment rather than occupying another run's VRAM.
    cfg.sim.physx.gpu_max_rigid_contact_count = max(2**18, args.num_envs * 1024)
    cfg.sim.physx.gpu_max_rigid_patch_count = max(2**15, args.num_envs * 128)
    cfg.sim.physx.gpu_found_lost_pairs_capacity = max(2**18, args.num_envs * 1024)
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = max(2**18, args.num_envs * 1024)
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = max(2**18, args.num_envs * 1024)
    runner_cfg = TQBot2BlindBoxPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = args.device
    runner_cfg.logger = "tensorboard"
    runner_cfg.max_iterations = args.iterations
    adapter_only = args.history_adapter_only or args.walking_checkpoint is not None
    if args.checkpoint and (args.checkpoint.parent / "report.json").exists():
        previous = json.loads((args.checkpoint.parent / "report.json").read_text())
        adapter_only |= previous.get("history_adapter_only", False)
    if adapter_only:
        runner_cfg.algorithm.learning_rate = 1e-4
        runner_cfg.algorithm.schedule = "fixed"
        runner_cfg.algorithm.entropy_coef = 0.001
    walking_state = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor_state, critic_state = saved["actor_state_dict"], saved["critic_state_dict"]
        if actor_state["mlp.0.weight"].shape[1] != 684:
            raise ValueError("Full resumes require a blind-box checkpoint; use --walking_checkpoint for a walker")
        for name, model_state in (("actor", actor_state), ("critic", critic_state)):
            setattr(runner_cfg.policy, f"{name}_hidden_dims",
                    [model_state[f"mlp.{index}.weight"].shape[0] for index in (0, 2, 4)])
            setattr(runner_cfg.policy, f"{name}_obs_normalization",
                    any(key.startswith("obs_normalizer.") for key in model_state))
    if args.walking_checkpoint:
        if args.checkpoint:
            raise ValueError("Select either a full checkpoint or a walking actor initialization")
        walking_state = torch.load(args.walking_checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        runner_cfg.policy.actor_hidden_dims = [walking_state[f"mlp.{index}.weight"].shape[0] for index in (0, 2, 4)]
        runner_cfg.policy.actor_obs_normalization = False
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, importlib.metadata.version("rsl-rl-lib"))
    dump_yaml(str(args.output / "params" / "env.yaml"), cfg)
    dump_yaml(str(args.output / "params" / "agent.yaml"), runner_cfg)
    env = gym.make("RobotLab-Isaac-Climb-BlindBox-TQBot2-v0", cfg=cfg).unwrapped
    try:
        report = {"seed": args.seed, "num_envs": args.num_envs, "validation": validate(env)}
        (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.validation_only:
            return
        if args.baseline_report:
            report["zero_action_baseline"] = json.loads(args.baseline_report.read_text())["zero_action_baseline"]
            report["zero_action_baseline_source"] = str(args.baseline_report.resolve())
        else:
            report["zero_action_baseline"] = evaluate(env, None, [min(args.eval_heights)])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        with torch.inference_mode():
            env.reset()
        print("Creating PPO runner", flush=True)
        # Isaac Lab lazily caches tensors during inference-mode physics steps.
        # Resets after evaluation must use the same mode, without creating the
        # trainable model parameters inside inference mode.
        with torch.inference_mode():
            wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(args.output), device=args.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint))
            report["checkpoint"] = str(args.checkpoint.resolve())
            report["loaded_iteration"] = runner.current_learning_iteration
        if walking_state:
            actor = runner.alg.get_policy()
            actor.load_state_dict(lift_walking_actor(walking_state, actor.state_dict()))
            report["walking_checkpoint"] = str(args.walking_checkpoint.resolve())
            report["walking_checkpoint_sha256"] = hashlib.sha256(args.walking_checkpoint.read_bytes()).hexdigest()
            shutil.copy2(args.walking_checkpoint, args.output / "initial_walking_checkpoint.pt")
            if args.baseline_report:
                baseline = json.loads(args.baseline_report.read_text())
                if (baseline["walking_checkpoint_sha256"] != report["walking_checkpoint_sha256"]
                    or baseline["seed"] != args.seed or baseline["num_envs"] != args.num_envs
                    or [item["height_m"] for item in baseline["walking_baseline"]] != args.eval_heights
                    or any(item["episodes"] != args.num_envs * args.trials_per_env
                           for item in baseline["walking_baseline"])):
                    raise ValueError("Baseline checkpoint, seed, environment count, or evaluation cases do not match")
                manifest = json.loads((args.baseline_report.parent / "source_snapshot/sha256.json").read_text())
                for name in ("blind_box_env_cfg.py", "blind_box_mdp.py", "blind_box_geometry.py"):
                    file = Path(box.__file__).parent / name
                    relative = str(file.relative_to(Path(__file__).resolve().parents[3]))
                    if manifest[relative] != hashlib.sha256(file.read_bytes()).hexdigest():
                        raise ValueError(f"Baseline task definition differs: {name}")
                report["walking_baseline"] = baseline["walking_baseline"]
                report["walking_baseline_source"] = str(args.baseline_report.resolve())
            else:
                report["walking_baseline"] = evaluate(env, runner.get_inference_policy(device=args.device))
            # Fine-tuning needs smaller perturbations than the rough-terrain
            # walker's trained exploration distribution. Deterministic baseline
            # actions are unchanged by this reset.
            with torch.no_grad():
                actor.distribution.std_param.fill_(0.2)
            report["fine_tune_initial_action_std"] = 0.2
        if adapter_only:
            train_history_adapter_only(runner.alg.get_policy(), runner.alg.optimizer)
        report["history_adapter_only"] = adapter_only
        if not args.curriculum_state:
            env.scene.terrain.course = box.BoxCourse()
        with torch.inference_mode():
            env.reset()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        start = time.monotonic()
        if args.iterations:
            runner.learn(args.iterations, init_at_random_ep_len=False)
        report["training_iterations"] = args.iterations
        report["training_seconds"] = round(time.monotonic() - start, 2)
        report["curriculum"] = vars(env.scene.terrain.course).copy()
        report["evaluation"] = evaluate(env, runner.get_inference_policy(device=args.device))
        report["actor_uses"] = "IMU, joint positions/velocities, desired joint targets, previous actions, user command"
        report["status"] = "complete"
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        # Kit shutdown can exit before Python prints an uncaught traceback.
        traceback.print_exc()
        raise
    finally:
        env.close()


try:
    main()
finally:
    app.close()
