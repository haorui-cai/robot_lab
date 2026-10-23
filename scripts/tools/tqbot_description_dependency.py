"""Resolve and build the pinned private TQBot model dependency."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPENDENCY = REPO / "external/tqbot_description"
MODEL = REPO / "source/robot_lab/data/Robots/TQBot/TQBot2_description"
BUILT_MODEL = DEPENDENCY / "build/isaac/TQBot2_description"


def rebuild():
    if not (DEPENDENCY / "tools/build_isaac.py").is_file():
        raise RuntimeError("Initialize the private dependency: git submodule update --init external/tqbot_description")
    expected = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files", "--stage", "--", "external/tqbot_description"], text=True
    ).split()
    actual = subprocess.check_output(["git", "-C", str(DEPENDENCY), "rev-parse", "HEAD"], text=True).strip()
    if len(expected) < 2 or expected[0] != "160000" or actual != expected[1]:
        raise RuntimeError("Private description checkout does not match the recorded submodule commit")
    dirty = subprocess.check_output(["git", "-C", str(DEPENDENCY), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("Private description sources have local changes; commit and update the dependency pin first")
    if not MODEL.is_symlink() or MODEL.resolve() != BUILT_MODEL.resolve():
        raise RuntimeError("Model path must be the versioned link to the private generated model; preserve any local copy before migrating")
    subprocess.run([sys.executable, str(DEPENDENCY / "tools/build_isaac.py")], check=True)
    return MODEL


if __name__ == "__main__":
    print(rebuild())
