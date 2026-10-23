"""Rebuild the explicit profiles provided by the pinned private description."""
import argparse

from tqbot_description_dependency import rebuild


def build():
    return rebuild()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-left-thighs", action="store_true", help="Compatibility option; both rolling profiles are generated")
    parser.parse_args()
    print(build())


if __name__ == "__main__":
    main()
