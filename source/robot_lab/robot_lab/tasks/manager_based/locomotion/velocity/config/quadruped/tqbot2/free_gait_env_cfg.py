"""Free-gait control: V4 plant/task, no gait-style/timing rewards, flat/slopes only."""
from isaaclab.utils import configclass
from isaaclab.terrains import MeshPlaneTerrainCfg

from .free_gait_policy import apply_free_gait
from .free_gait_terrain import configure_free_gait_terrain
from .reward_v4_env_cfg import TQBot2RewardV4EnvCfg


@configclass
class TQBot2FreeGaitEnvCfg(TQBot2RewardV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_free_gait(self.rewards)
        configure_free_gait_terrain(self.scene.terrain.terrain_generator, MeshPlaneTerrainCfg(proportion=.2))
