"""Check implicit TQBot2 walking actuators and Go2-style rewards in actual PhysX."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rough", action="store_true", help="Also exercise terrain raycasts on generated terrain")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)


def main():
    cfg = TQBot2StandardWalkRoughEnvCfg()
    from isaaclab.actuators import ImplicitActuatorCfg
    from robot_lab.assets.tqbot import TQBOT2_CFG
    from robot_lab.assets.tqbot_walk_stance import (
        WALK_DEFAULT_FOOT_Z_TRUNK, WALK_FOOT_HEIGHT_INCREMENT,
        WALK_FOOT_HEIGHT_TARGET_TRUNK, WALK_JOINT_POS,
    )
    from robot_lab.assets.tqbot_collision import spawn_tqbot2_walk_with_leg_collisions
    terrains = cfg.scene.terrain.terrain_generator.sub_terrains
    assert len(terrains) == 6 and "high_friction_flat" not in terrains
    assert abs(sum(t.proportion for t in terrains.values()) - 1.) < 1e-12
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        assert terrains[name].step_height_range == (.035, .161)
    assert terrains["boxes"].grid_height_range == (.035, .14)
    assert cfg.scene.robot.spawn.func is spawn_tqbot2_walk_with_leg_collisions
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    assert cfg.actions.joint_pos.clip == {".*": (-100., 100.)}
    for name, actuator in cfg.scene.robot.actuators.items():
        source = TQBOT2_CFG.actuators[name]
        assert isinstance(actuator, ImplicitActuatorCfg) and isinstance(source, ImplicitActuatorCfg)
        assert actuator.effort_limit_sim == source.effort_limit_sim * .90
        assert actuator.stiffness == 20. and actuator.damping == .5
        assert source.stiffness == 14. and source.damping == .6
        for param in ("joint_names_expr", "velocity_limit_sim",
                      "armature", "friction", "dynamic_friction", "viscous_friction"):
            assert getattr(actuator, param) == getattr(source, param)
    from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.agents.rsl_rl_ppo_cfg import (
        TQBot2StandardWalkPPORunnerCfg,
    )
    agent = TQBot2StandardWalkPPORunnerCfg()
    assert agent.policy.actor_obs_normalization is False and agent.policy.critic_obs_normalization is False
    assert agent.algorithm.learning_rate == .001 and agent.algorithm.schedule == "adaptive"
    assert agent.algorithm.entropy_coef == .01 and agent.resume is False
    assert agent.experiment_name == "tqbot2_standard_walk"
    assert cfg.foot_link_name == ".*_foot"
    assert cfg.events.randomize_foot_material.params["static_friction_range"] == (.3, 1.)
    assert cfg.events.randomize_foot_material.params["dynamic_friction_range"] == (.3, .8)
    slopes = {n: cfg.scene.terrain.terrain_generator.sub_terrains[n].slope_range
              for n in ("hf_pyramid_slope", "hf_pyramid_slope_inv")}
    assert all(abs(r[1] - math.tan(math.radians(30))) < 1e-12 for r in slopes.values())
    num_envs = 16
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    if args.rough:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
        cfg.scene.terrain.max_init_terrain_level = 1
    else:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.height_scanner = None
        cfg.scene.height_scanner_base = None
        cfg.observations.critic.height_scan = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.terrain_breakdown = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        robot = env.scene["robot"]
        feet = [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]
        assert len(robot.body_names) == 17 and all(name in robot.body_names for name in feet)
        contact_term = env.reward_manager.get_term_cfg("undesired_contacts")
        contact_sensor = env.scene[contact_term.params["sensor_cfg"].name]
        penalized_contact_bodies = [contact_sensor.body_names[i]
                                    for i in contact_term.params["sensor_cfg"].body_ids]
        assert set(penalized_contact_bodies) == {"trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip"}
        assert "illegal_contact" not in env.termination_manager.active_terms
        # Exercise the resolved reward with isolated contact forces. A calf or
        # thigh contact must be free; trunk and hip contacts must still count.
        forces = contact_sensor.data.net_forces_w_history
        saved_forces = forces.clone()
        try:
            for body_name in contact_sensor.body_names:
                forces.zero_()
                forces[:, :, contact_sensor.body_names.index(body_name), 2] = 10.
                expected = float(body_name in penalized_contact_bodies)
                torch.testing.assert_close(contact_term.func(env, **contact_term.params),
                                           torch.full((num_envs,), expected, device=env.device))
        finally:
            forces.copy_(saved_forces)
        foot_ids, _ = robot.find_bodies(feet, preserve_order=True)
        calf_ids, _ = robot.find_bodies([n.replace("foot", "calf") for n in feet], preserve_order=True)
        from isaaclab.utils.math import quat_apply
        offset = torch.tensor([0., 0., -.153], device=env.device).expand(num_envs, 4, 3)
        expected_feet = robot.data.body_link_pos_w[:, calf_ids] + quat_apply(robot.data.body_link_quat_w[:, calf_ids], offset)
        torch.testing.assert_close(robot.data.body_link_pos_w[:, foot_ids], expected_feet, atol=2e-5, rtol=0.)
        assert cfg.scene.robot.init_state.joint_pos == WALK_JOINT_POS
        assert TQBOT2_CFG.init_state.joint_pos["F.*_thigh_joint"] == .8
        assert TQBOT2_CFG.init_state.joint_pos[".*_calf_joint"] == -1.5
        # Before physics integration, nominal joints and upright root must
        # reproduce the URDF-derived stance in the actual root link frame.
        from isaaclab.utils.math import quat_apply_inverse
        foot_in_trunk = quat_apply_inverse(
            robot.data.root_link_quat_w[:, None, :].expand(-1, 4, -1),
            robot.data.body_link_pos_w[:, foot_ids] - robot.data.root_link_pos_w[:, None, :],
        )
        torch.testing.assert_close(foot_in_trunk[..., 2],
                                  torch.full_like(foot_in_trunk[..., 2], WALK_DEFAULT_FOOT_Z_TRUNK),
                                  atol=2e-5, rtol=0.)
        actual_default_foot_z = foot_in_trunk[0, :, 2].tolist()
        import xml.etree.ElementTree as ET
        source = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
        source_mass = sum(float(m.get("value")) for m in source.findall("link/inertial/mass"))
        torch.testing.assert_close(robot.data.default_mass.sum(dim=1),
                                   torch.full_like(robot.data.default_mass.sum(dim=1), source_mass), atol=1e-4, rtol=0.)
        torch.testing.assert_close(robot.data.default_mass[:, foot_ids],
                                   torch.full_like(robot.data.default_mass[:, foot_ids], .015), atol=1e-6, rtol=0.)
        from robot_lab.assets.tqbot_collision import WALK_SELF_COLLISION_PAIRS_WITH_FEET
        from pxr import Usd, UsdPhysics
        from itertools import combinations
        for index in range(num_envs):
            root = robot.stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
            bodies = {p.GetName(): p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
            trunk_shapes = {p.GetPath() for p in Usd.PrimRange(bodies["trunk"], Usd.TraverseInstanceProxies())
                            if p.HasAPI(UsdPhysics.CollisionAPI)}
            for left, right in combinations(bodies, 2):
                left_targets = UsdPhysics.FilteredPairsAPI(bodies[left]).GetFilteredPairsRel().GetTargets()
                right_targets = UsdPhysics.FilteredPairsAPI(bodies[right]).GetFilteredPairsRel().GetTargets()
                if "trunk" in (left, right):
                    excluded = trunk_shapes.issubset(right_targets if left == "trunk" else left_targets)
                else:
                    excluded = bodies[right].GetPath() in left_targets or bodies[left].GetPath() in right_targets
                assert excluded == (frozenset((left, right)) not in WALK_SELF_COLLISION_PAIRS_WITH_FEET)
        thigh_ids = [i for i, n in enumerate(robot.joint_names) if n.endswith("_thigh_joint")]
        assert len(thigh_ids) == 4
        limits = robot.data.joint_pos_limits[:, thigh_ids]
        torch.testing.assert_close(limits[..., 0], torch.full_like(limits[..., 0], .2))
        torch.testing.assert_close(limits[..., 1], torch.full_like(limits[..., 1], 3.2))
        soft = robot.data.soft_joint_pos_limits[:, thigh_ids]
        torch.testing.assert_close(soft[..., 0], torch.full_like(soft[..., 0], .33))
        torch.testing.assert_close(soft[..., 1], torch.full_like(soft[..., 1], 3.07))
        assert obs["policy"].shape == (num_envs, 45)
        from isaaclab.actuators import ImplicitActuator
        assert robot._has_implicit_actuators
        assert all(isinstance(a, ImplicitActuator) for a in robot.actuators.values())
        torch.testing.assert_close(robot.data.default_joint_stiffness,
                                   torch.full_like(robot.data.default_joint_stiffness, 20.))
        torch.testing.assert_close(robot.data.default_joint_damping,
                                   torch.full_like(robot.data.default_joint_damping, .5))
        for actuator in robot.actuators.values():
            drive_ids = actuator.joint_indices.cpu() if isinstance(actuator.joint_indices, torch.Tensor) else actuator.joint_indices
            torch.testing.assert_close(robot.root_physx_view.get_dof_max_forces()[:, drive_ids],
                                       actuator.effort_limit_sim.cpu())
            torch.testing.assert_close(robot.root_physx_view.get_dof_stiffnesses()[:, drive_ids],
                                       actuator.stiffness.cpu())
            torch.testing.assert_close(robot.root_physx_view.get_dof_dampings()[:, drive_ids],
                                       actuator.damping.cpu())
            assert ((actuator.stiffness >= 10.) & (actuator.stiffness <= 40.)).all()
            assert ((actuator.damping >= .25) & (actuator.damping <= 1.)).all()
        friction = robot.root_physx_view.get_dof_friction_properties()
        torch.testing.assert_close(friction, friction.new_tensor([.4, .4, 0.]).expand_as(friction))
        assert torch.count_nonzero(robot.root_physx_view.get_dof_friction_coefficients()) == 0
        assert "high_friction_flat" not in (cfg.scene.terrain.terrain_generator.sub_terrains if args.rough else {})
        assert "high_friction_flat_material" not in env.event_manager.active_terms.get("startup", [])
        assert "high_friction_flat_exit" not in env.termination_manager.active_terms
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512, 256, 128],
                         activation="elu", obs_normalization=False,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}).to(env.device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["actor_state_dict"])
        actor.eval()
        action = env.action_manager.get_term("joint_pos")
        for sign in (-1, 1):
            raw = torch.full((num_envs, 12), 100. * sign, device=env.device)
            action.process_actions(raw)
            expected = torch.clamp(raw * action._scale + action._offset, -100., 100.)
            torch.testing.assert_close(action.processed_actions, expected)
            joint_ids = [i for i, n in enumerate(action._joint_names) if n.endswith("_thigh_joint")]
            targets = action.processed_actions[:, joint_ids]
            if sign < 0:
                assert (targets < .6).all()
            else:
                assert (targets > 3.2).all()
        env.reset()
        state = robot.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        half = num_envs // 2
        state[half:, 2] = env.scene.env_origins[half:, 2] + .12
        state[half:, 3:7] = torch.tensor([0., 1., 0., 0.], device=env.device)
        state[:, 7:] = 0.
        ids = torch.arange(num_envs, device=env.device)
        robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(state[:, 7:], env_ids=ids)
        command = env.command_manager.get_term("base_velocity")
        command.vel_command_b[:] = 0.
        command.is_heading_env[:] = False
        command.is_standing_env[:] = True
        command.vel_command_b[:half, 0] = .5
        command.is_standing_env[:half] = False
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        obs = env.observation_manager.compute()
        peaks = {"torque_nm": 0., "joint_speed_rad_s": 0.}
        with torch.no_grad():
            for _ in range(300):
                obs, reward, _, _, _ = env.step(actor(obs, stochastic_output=False))
                assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(action.processed_actions).all()
                for actuator in robot.actuators.values():
                    assert torch.isfinite(actuator.applied_effort).all()
                    assert (actuator.applied_effort.abs() <= actuator.effort_limit + 1e-5).all()
                assert torch.isfinite(env.reward_manager._step_reward).all()
                peaks["torque_nm"] = max(peaks["torque_nm"], robot.data.applied_torque.abs().max().item())
                peaks["joint_speed_rad_s"] = max(peaks["joint_speed_rad_s"], robot.data.joint_vel.abs().max().item())
        recovery_terms = {"recovery_effort", "recovery_torque_change", "recovery_joint_velocity"}
        assert recovery_terms.isdisjoint(env.reward_manager.active_terms)
        assert env.reward_manager.get_term_cfg("feet_gait").weight == .5
        from robot_lab.tasks.manager_based.locomotion.velocity.mdp.foot_ground import SupportFootSlip
        slide = env.reward_manager.get_term_cfg("feet_slide")
        assert isinstance(slide.func, SupportFootSlip) and slide.weight == -.1
        assert "feet_height_body" in env.reward_manager.active_terms
        height_term = env.reward_manager.get_term_cfg("feet_height_body")
        assert height_term.weight == -5. and height_term.params["target_height"] == WALK_FOOT_HEIGHT_TARGET_TRUNK
        assert abs(height_term.params["target_height"] - WALK_DEFAULT_FOOT_Z_TRUNK - WALK_FOOT_HEIGHT_INCREMENT) < 1e-12
        sensor = env.scene["contact_forces"]
        air_time = sensor.data.current_air_time.clone()
        try:
            sensor.data.current_air_time.zero_()
            torch.testing.assert_close(height_term.func(env, **height_term.params),
                                       torch.zeros(num_envs, device=env.device))
            sensor.data.current_air_time.fill_(.1)
            unmasked = dict(height_term.params, sensor_cfg=None)
            torch.testing.assert_close(height_term.func(env, **height_term.params),
                                       height_term.func(env, **unmasked))
        finally:
            sensor.data.current_air_time.copy_(air_time)
        removed_terms = {"feet_swing_clearance", "walking_energy_efficiency", "walking_joule_proxy"}
        assert removed_terms.isdisjoint(env.reward_manager.active_terms)
        subset = torch.tensor([1, 7], device=env.device)
        env._reset_idx(subset)
        torch.testing.assert_close(robot.data.soft_joint_pos_limits[:, thigh_ids, 0],
                                   torch.full_like(soft[..., 0], .33))
        assert torch.isfinite(env.observation_manager.compute()["policy"]).all()
        report = dict(checkpoint=str(args.checkpoint.resolve()), device=env.device, num_envs=num_envs,
                      physics_thigh_limits=[.2, 3.2], soft_thigh_limits=[.33, 3.07],
                      extra_thigh_target_clipping_removed_verified=True,
                      actuator_type="ImplicitActuator", actuator_effort_limits_verified=True,
                      physx_drive_gains_verified=True, joint_friction_verified=True,
                      recovery_costs_disabled_verified=True, partial_reset_verified=True,
                      feet_gait_weight=.5, steps=300, nominal_slope_ranges=slopes,
                      terrain="generated" if args.rough else "plane",
                      feet_height_body_target_m=WALK_FOOT_HEIGHT_TARGET_TRUNK, feet_height_body_weight=-5.,
                      default_foot_z_trunk_m=actual_default_foot_z,
                      foot_height_increment_m=WALK_FOOT_HEIGHT_INCREMENT,
                      foot_height_reference_frame="trunk root link",
                      walking_stance_verified=True,
                      extra_energy_terms_disabled_verified=True, swing_clearance_disabled_verified=True,
                      high_friction_flat_removed_verified=True,
                      support_contact_point_slip_verified=True, swing_only_height_verified=True,
                      training_nominal_pd={"stiffness": 20., "damping": .5},
                      training_randomized_pd_ranges={"stiffness": [10., 40.], "damping": [.25, 1.]},
                      leg_contact_penalty_removed_verified=True,
                      penalized_contact_bodies=penalized_contact_bodies,
                      non_gain_motor_parameters_preserved_verified=True, training_torque_scale=.90,
                      training_effort_limits_nm={"hip_thigh": 23.7 * .90, "calf": 35.55 * .90},
                      shared_robot_config_unchanged_verified=True, independent_feet_verified=True,
                      foot_offset_m=[0., 0., -.153], foot_mass_kg=.015, total_mass_kg=source_mass,
                      self_collision_pairs_checked=num_envs * 136,
                      foot_friction_lower_bound=.3, ppo_learning_rate=.001, ppo_schedule="adaptive",
                      ppo_entropy_coef=.01, obs_normalization=False, resume=False,
                      lower_step_terrain_ranges_verified=True, selective_self_collision_config_verified=True,
                      peaks=peaks, upright_fraction_at_end=float((-robot.data.projected_gravity_b[:, 2] > .866).float().mean()),
                      note="Integration smoke test; walking and 30-degree climbing performance not established")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print("TQBot2 standard walking integration checks passed", flush=True)
    finally:
        env.close()


main()
app.close()
