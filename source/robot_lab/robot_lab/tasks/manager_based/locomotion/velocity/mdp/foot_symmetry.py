"""URDF reference/calibration and tensor math for diagonal foot displacement."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def urdf_foot_positions(urdf_path, joint_positions, foot_names):
    """Batched FK of link origins, expressed in the URDF root link frame.

    joint_positions maps exact joint names to equally sized 1-D arrays. This
    walking calibration supports fixed and revolute joints only.
    """
    root = ET.parse(urdf_path).getroot()
    pending = list(root.findall("joint"))
    roots = {link.get("name") for link in root.findall("link")} - {
        joint.find("child").get("link") for joint in pending
    }
    if len(roots) != 1:
        raise ValueError("Foot symmetry requires a single URDF root link")
    count = len(next(iter(joint_positions.values())))
    identity = np.broadcast_to(np.eye(4), (count, 4, 4))
    poses = {roots.pop(): identity}
    while pending:
        before = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            transform = identity.copy()
            origin = joint.find("origin")
            if origin is not None:
                transform[:, :3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:, :3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                angles = np.asarray(joint_positions[joint.get("name")], dtype=float)
                rotation = Rotation.from_rotvec(angles[:, None] * axis / np.linalg.norm(axis)).as_matrix()
                transform[:, :3, :3] = transform[:, :3, :3] @ rotation
            elif joint.get("type") != "fixed":
                raise ValueError(f"Unsupported calibration joint: {joint.get('name')}")
            poses[joint.find("child").get("link")] = poses[parent] @ transform
            pending.remove(joint)
        if len(pending) == before:
            raise ValueError("Unresolved URDF joint topology")
    return np.stack([poses[name][:, :3, 3] for name in foot_names], axis=1)


def calibrate_diagonal_displacement(urdf_path, foot_names, reference_joints,
                                    perturbation_rad=0.05, sample_count=8192, seed=42):
    """Match mean old/new costs on independent uniform joint perturbations.

    Feet must be ordered by diagonal pairs (FR, RL, FL, RR). All three joint
    coordinates per leg contribute to the old cost; only foot X/Z contribute
    to the new cost. No physics state, policy actions or global RNG is changed.
    """
    if len(foot_names) != 4 or len(set(foot_names)) != 4 or any(not n.endswith("_foot") for n in foot_names):
        raise ValueError("Calibration requires four distinct named feet in diagonal-pair order")
    if perturbation_rad <= 0 or sample_count < 2:
        raise ValueError("Calibration requires positive perturbations and at least two samples")
    names = [f"{foot.removesuffix('_foot')}_{kind}_joint"
             for foot in foot_names for kind in ("hip", "thigh", "calf")]
    reference = np.array([reference_joints[name] for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    samples = reference + rng.uniform(-perturbation_rad, perturbation_rad, (sample_count, len(names)))
    # Include any fixed-at-reference articulated joints outside the four legs.
    positions = {name: np.full(sample_count + 1, value) for name, value in reference_joints.items()}
    positions.update({name: np.r_[reference[i], samples[:, i]] for i, name in enumerate(names)})
    feet = urdf_foot_positions(urdf_path, positions, foot_names)
    displacement = (feet[1:] - feet[0])[..., (0, 2)]
    new_cost = np.square(displacement[:, 0::2] - displacement[:, 1::2]).sum(axis=-1).mean(axis=-1)
    angles = samples.reshape(sample_count, 4, 3)
    old_cost = np.square(angles[:, 0::2] - angles[:, 1::2]).sum(axis=-1).mean(axis=-1)
    old_mean, new_mean = float(old_cost.mean()), float(new_cost.mean())
    if not np.isfinite(old_mean + new_mean) or min(old_mean, new_mean) <= 0:
        raise ValueError("Degenerate foot-symmetry calibration")
    return {
        "foot_names": list(foot_names), "reference_positions_body": feet[0].tolist(),
        "scale": old_mean / new_mean,
        "old_cost_mean": old_mean, "foot_cost_mean_m2": new_mean,
        "perturbation_rad": perturbation_rad, "sample_count": sample_count, "seed": seed,
        "urdf_sha256": hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
    }


def diagonal_displacement_cost(foot_positions_body, reference_positions_body):
    """Unweighted mean squared X/Z displacement mismatch for two foot pairs."""
    delta = (foot_positions_body - reference_positions_body)[..., (0, 2)]
    return (delta[:, 0::2] - delta[:, 1::2]).square().sum(dim=-1).mean(dim=-1)
