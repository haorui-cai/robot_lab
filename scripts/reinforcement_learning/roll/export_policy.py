#!/usr/bin/env python3
"""Export BC checkpoint to TorchScript and ONNX for deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


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
    parser = argparse.ArgumentParser(description="Export BC policy to deployable formats.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input BC checkpoint (.pt).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported files.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = BCPolicy(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, ckpt["obs_dim"], dtype=torch.float32)

    ts_path = args.output_dir / "roll_policy_jit.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))

    onnx_path = args.output_dir / "roll_policy.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        onnx_status = "ok"
    except Exception as exc:  # pragma: no cover
        onnx_status = f"failed: {exc}"

    meta = {
        "obs_dim": ckpt["obs_dim"],
        "action_dim": ckpt["action_dim"],
        "jit": str(ts_path),
        "onnx": str(onnx_path),
        "onnx_status": onnx_status,
    }
    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    tiny_mlp = {
        "obs_dim": ckpt["obs_dim"],
        "action_dim": ckpt["action_dim"],
        "layers": [
            {
                "weight": layer.weight.detach().cpu().tolist(),
                "bias": layer.bias.detach().cpu().tolist(),
            }
            for layer in linear_layers
        ],
        "output_scale": 0.6,
        "activation": "elu_tanh_last",
    }
    (args.output_dir / "roll_policy_tinymlp.json").write_text(json.dumps(tiny_mlp), encoding="utf-8")
    txt_path = args.output_dir / "roll_policy_tinymlp.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"obs_dim {ckpt['obs_dim']}\n")
        f.write(f"action_dim {ckpt['action_dim']}\n")
        f.write(f"num_layers {len(linear_layers)}\n")
        f.write("output_scale 0.6\n")
        for idx, layer in enumerate(linear_layers):
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy()
            rows, cols = weight.shape
            f.write(f"layer {idx} rows {rows} cols {cols}\n")
            for r in range(rows):
                f.write(" ".join(f"{v:.8e}" for v in weight[r]) + "\n")
            f.write("bias\n")
            f.write(" ".join(f"{v:.8e}" for v in bias) + "\n")
            f.write("endlayer\n")

    (args.output_dir / "roll_policy_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
