# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TQBot2RollPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 12000
    save_interval = 100
    experiment_name = "tqbot2_roll_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TQBot2RollHipOnlyPPORunnerCfg(TQBot2RollPPORunnerCfg):
    experiment_name = "tqbot2_roll_hip_only"
    run_name = "curled_thigh168_calf282"


@configclass
class TQBot2RollHipOnlyCyclePPORunnerCfg(TQBot2RollPPORunnerCfg):
    experiment_name = "tqbot2_roll_hip_only_cycle"
    run_name = "phase_yaw_curriculum_seeded_roll"
    clip_actions = 1.0


@configclass
class TQBot2RollLeftThighCyclePPORunnerCfg(TQBot2RollHipOnlyCyclePPORunnerCfg):
    experiment_name = "tqbot2_roll_left_thigh_cycle"
    run_name = "left_thigh_168_245_phase_yaw_curriculum"


@configclass
class TQBot2RollLeftThighSlopePPORunnerCfg(TQBot2RollLeftThighCyclePPORunnerCfg):
    experiment_name = "tqbot2_roll_left_thigh_slope"
    run_name = "up10_down30_yaw5_assist_decay"
