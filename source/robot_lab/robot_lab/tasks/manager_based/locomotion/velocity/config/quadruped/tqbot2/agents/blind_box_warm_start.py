"""Lift a 45-input walking actor into a 12-frame actor without changing its output."""

import torch


def latest_walking_columns(history_length=12):
    columns, offset = [], 0
    for dimension in (3, 3, 3, 12, 12, 12):
        begin = offset + (history_length - 1) * dimension
        columns.extend(range(begin, begin + dimension))
        offset += history_length * dimension
    return columns


def lift_walking_actor(walking_state, target_state):
    """Preserve all hidden layers; only the latest six observation terms have weights.

    This migration is deliberately strict. It supports the current non-normalized
    TQBot2 MLP walking actor, with matching joint order/scales and three hidden layers.
    The critic and PPO optimizer start fresh for the new reward/task.
    """
    if set(walking_state) != set(target_state):
        raise ValueError("Walking and target actors must use matching non-normalized MLP/distribution layouts")
    key = "mlp.0.weight"
    if walking_state[key].shape[1] != 45 or target_state[key].shape[1] != 684:
        raise ValueError("Expected the TQBot2 45-input walker and 684-input blind-box actor")
    result = {}
    for name, target in target_state.items():
        old = walking_state[name].to(device=target.device, dtype=target.dtype)
        if name == key:
            if old.shape[0] != target.shape[0]:
                raise ValueError("Hidden dimensions must match the walking actor")
            expanded = torch.zeros_like(target)
            expanded[:, latest_walking_columns()] = old
            result[name] = expanded
        else:
            if old.shape != target.shape:
                raise ValueError(f"Incompatible actor parameter {name}: {old.shape} vs {target.shape}")
            result[name] = old.clone()
    return result


def train_history_adapter_only(actor, optimizer=None):
    """Keep the walking prior fixed and learn only previously zero history columns."""
    if not isinstance(actor.obs_normalizer, torch.nn.Identity):
        raise ValueError("The walking history adapter requires an unnormalized actor")
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name == "mlp.0.weight")
    weight = actor.mlp[0].weight
    columns = latest_walking_columns()
    mask = torch.ones_like(weight)
    mask[:, columns] = 0
    weight.register_hook(lambda gradient: gradient * mask)
    # A resumed Adam optimizer can otherwise move masked columns through momentum.
    if optimizer is not None:
        for value in optimizer.state.get(weight, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                value[:, columns] = 0
