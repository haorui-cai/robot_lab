"""Free-gait control: same V4 plant/task, without gait-style/timing rewards."""
from isaaclab.utils import configclass

from .free_gait_policy import apply_free_gait
from .reward_v4_env_cfg import TQBot2RewardV4EnvCfg


@configclass
class TQBot2FreeGaitEnvCfg(TQBot2RewardV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_free_gait(self.rewards)
