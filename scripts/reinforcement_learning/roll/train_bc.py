#!/usr/bin/env python3
"""Simple behavior-cloning trainer for TQBot2 rolling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return 0.6 * self.net(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BC policy from heuristic roll dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to heuristic dataset .npz file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt checkpoint path.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset)
    obs = torch.from_numpy(data["obs"]).float()
    actions = torch.from_numpy(data["actions"]).float()
    dataset = TensorDataset(obs, actions)

    val_size = max(int(0.1 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = BCPolicy(obs_dim=obs.shape[1], action_dim=actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(x), y).item() for x, y in val_loader]
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"[epoch {epoch + 1:03d}] train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce any model checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "obs_dim": obs.shape[1],
            "action_dim": actions.shape[1],
            "best_val": best_val,
        },
        args.output,
    )
    print(f"Saved BC checkpoint to {args.output} (best val={best_val:.6f})")


if __name__ == "__main__":
    main()
