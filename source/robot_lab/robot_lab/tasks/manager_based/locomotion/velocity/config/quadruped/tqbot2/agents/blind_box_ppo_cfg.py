"""Separate runner configuration; existing walking checkpoints are unchanged."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import TQBot2RoughPPORunnerCfg


@configclass
class TQBot2BlindBoxPPORunnerCfg(TQBot2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "tqbot2_blind_box"
        self.max_iterations = 10000
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.policy.init_noise_std = 0.7
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.algorithm.gamma = 0.995
