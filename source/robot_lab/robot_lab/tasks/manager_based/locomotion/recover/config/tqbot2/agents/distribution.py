# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Bounded exploration noise for the clipped Recover action space (RSL-RL 5)."""

import math

import torch
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from torch import nn
from torch.distributions import Normal


class BoundedGaussianDistribution(GaussianDistribution):
    """Gaussian policy with smoothly bounded, learnable per-joint std.

    PPO log probabilities, entropy and KL all see the same bounded Gaussian.
    This does not modify the distribution after sampling or edit RSL-RL itself.
    Inference/export remain the Gaussian mean, as in the standard distribution.
    """

    def __init__(self, output_dim: int, init_std: float = 0.35, min_std: float = 0.03, max_std: float = 0.6):
        if not 0 < min_std < init_std < max_std:
            raise ValueError("Require 0 < min_std < init_std < max_std")
        Distribution.__init__(self, output_dim)
        self.min_std = min_std
        self.max_std = max_std
        ratio = (init_std - min_std) / (max_std - min_std)
        self.noise_logit = nn.Parameter(torch.full((output_dim,), math.log(ratio / (1.0 - ratio))))
        self._distribution = None

    def update(self, mlp_output: torch.Tensor) -> None:
        std = self.min_std + (self.max_std - self.min_std) * self.noise_logit.sigmoid()
        self._distribution = Normal(mlp_output, std.expand_as(mlp_output))
