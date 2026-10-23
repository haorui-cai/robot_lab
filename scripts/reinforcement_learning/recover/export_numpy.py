"""Export the Recover actor for dependency-light Gazebo evaluation."""
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("output")
args = parser.parse_args()
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
ids = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
arrays = {}
for n, i in enumerate(ids):
    arrays[f"w{n}"] = state[f"mlp.{i}.weight"].numpy()
    arrays[f"b{n}"] = state[f"mlp.{i}.bias"].numpy()
assert arrays["w0"].shape[1] == 42 and arrays[f"w{len(ids)-1}"].shape[0] == 12
x = np.random.default_rng(42).normal(size=(100, 42)).astype(np.float32)
y, reference = x.copy(), torch.from_numpy(x)
for n in range(len(ids)):
    y = y @ arrays[f"w{n}"].T + arrays[f"b{n}"]
    reference = torch.nn.functional.linear(reference, torch.from_numpy(arrays[f"w{n}"]), torch.from_numpy(arrays[f"b{n}"]))
    if n < len(ids)-1:
        y = np.where(y > 0, y, np.expm1(np.minimum(y, 0)))
        reference = torch.nn.functional.elu(reference)
np.testing.assert_allclose(y, reference.numpy(), atol=2e-5, rtol=2e-5)
np.savez(args.output, **arrays)
print("Exported actor; NumPy/PyTorch inference parity passed on 100 observations")
