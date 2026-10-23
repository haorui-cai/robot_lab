"""Slope curriculum with unchanged 35 observations / 6 actions for warm starts."""

from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.roll.mdp import slopes

from .left_thigh_cycle_env_cfg import TQBot2RollLeftThighCycleEnvCfg


@configclass
class TQBot2RollLeftThighSlopeEnvCfg(TQBot2RollLeftThighCycleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.class_type = slopes.RollingSlopeTerrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.events.randomize_reset_base.func = slopes.reset_slope_rolling
        self.terminations.invalid_rolling_pose = TerminationTermCfg(func=slopes.invalid_slope_pose)
        self.terminations.terrain_out_of_bounds = TerminationTermCfg(func=slopes.slope_out_of_bounds, time_out=True)
        self.curriculum.slopes = CurriculumTermCfg(func=slopes.slope_curriculum)
