"""Verify TQBot2's new friction model in cloned PhysX articulations.

Run with the Isaac Lab Python environment and --headless --device cuda:0.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from robot_lab.tasks.manager_based.locomotion.recover.config.tqbot2.recover_env_cfg import TQBot2RecoverEnvCfg
from robot_lab.tasks.manager_based.locomotion.roll.config.quadruped.tqbot2.hip_only_env_cfg import TQBot2RollHipOnlyEnvCfg
from robot_lab.tasks.manager_based.locomotion.roll.config.quadruped.tqbot2.left_thigh_cycle_env_cfg import TQBot2RollLeftThighCycleEnvCfg
from robot_lab.tasks.manager_based.locomotion.roll.config.quadruped.tqbot2.rough_env_cfg import TQBot2RollRoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.rough_env_cfg import TQBot2RoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg


def main():
    torch.set_num_threads(2)
    configs = {
        "walk_standard": TQBot2StandardWalkRoughEnvCfg(),
        "walk_implicit": TQBot2RoughEnvCfg(),
        "roll": TQBot2RollRoughEnvCfg(),
        "recover": TQBot2RecoverEnvCfg(),
        "hip_only": TQBot2RollHipOnlyEnvCfg(),
        "left_thigh": TQBot2RollLeftThighCycleEnvCfg(),
    }
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene_cfg = InteractiveSceneCfg(num_envs=4, env_spacing=20.0)
    for index, (name, cfg) in enumerate(configs.items()):
        for actuator in cfg.scene.robot.actuators.values():
            assert (actuator.friction, actuator.dynamic_friction, actuator.viscous_friction) == (0.4, 0.4, 0.0)
            assert (actuator.stiffness, actuator.damping) == (14.0, 0.6)
        robot_cfg = cfg.scene.robot.replace(prim_path=f"{{ENV_REGEX_NS}}/{name}")
        robot_cfg.spawn.fix_base = True
        robot_cfg.spawn.rigid_props.disable_gravity = True
        robot_cfg.init_state.pos = (2.0 * index, 0.0, 2.0)
        setattr(scene_cfg, name, robot_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    report = {"device": args.device, "num_envs": scene.num_envs, "robots": {}}

    def verify(name, robot, expected_new):
        legacy = robot.root_physx_view.get_dof_friction_coefficients()
        new = robot.root_physx_view.get_dof_friction_properties()
        assert legacy.shape == (4, robot.num_joints)
        assert torch.count_nonzero(legacy) == 0, (name, legacy)
        expected = torch.tensor(expected_new, device=new.device).expand_as(new)
        assert torch.allclose(new, expected, atol=1e-7, rtol=0), (name, new)
        return {"legacy": legacy.cpu().tolist(), "new_properties": new.cpu().tolist()}

    for name, robot in scene.articulations.items():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
        robot.set_joint_position_target(robot.data.default_joint_pos)
        report["robots"][name] = {
            "joints": robot.joint_names,
            "actuators": {group: type(actuator).__name__ for group, actuator in robot.actuators.items()},
            "initial": verify(name, robot, [0.4, 0.4, 0.0]),
        }
    for _ in range(8):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    scene.reset()
    for name, robot in scene.articulations.items():
        report["robots"][name]["after_steps_and_reset"] = verify(name, robot, [0.4, 0.4, 0.0])
        # Removing the new friction must not reactivate a retained legacy value.
        robot.write_joint_friction_coefficient_to_sim(0.0, 0.0, 0.0)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    for name, robot in scene.articulations.items():
        report["robots"][name]["new_zero_control"] = verify(name, robot, [0.0, 0.0, 0.0])
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("TQBOT2_NEW_FRICTION_SMOKE_OK", flush=True)
    # Follow ManagerBasedEnv's shutdown order for an in-memory physics stage.
    del scene
    if sim.cfg.create_stage_in_memory:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        sim.stop()
        sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
