"""Require byte-identical arrays, metadata, and RNG across current trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(baseline, candidate):
    groups = {}
    for enabled, configurations, states in ((False, 8, 60), (True, 2, 15)):
        counts = {
            "state_pairs": 0,
            "array_pairs": 0,
            "metadata_pairs": 0,
            "rng_pairs": 0,
        }
        for first in sorted(baseline.glob("*/step*/state_*.npz")):
            if ("four_heuristics" in first.parts[-3]) != enabled:
                continue
            second = candidate / first.relative_to(baseline)
            with (
                np.load(first, allow_pickle=False) as left,
                np.load(second, allow_pickle=False) as right,
            ):
                if set(left.files) != set(right.files):
                    raise AssertionError((second, "array keys"))
                for key in left.files:
                    if not (
                        left[key].dtype == right[key].dtype
                        and left[key].shape == right[key].shape
                        and left[key].tobytes() == right[key].tobytes()
                    ):
                        raise AssertionError((second, key))
                    counts["array_pairs"] += 1
            counts["state_pairs"] += 1
            agent = first.stem.removeprefix("state_")
            for kind in ("metadata", "rng"):
                left_path = first.parent / f"{kind}_{agent}.json"
                right_path = candidate / left_path.relative_to(baseline)
                if left_path.read_bytes() != right_path.read_bytes():
                    raise AssertionError(right_path)
                counts[f"{kind}_pairs"] += 1
        expected = {
            "state_pairs": states,
            "array_pairs": states * 22,
            "metadata_pairs": states,
            "rng_pairs": states,
        }
        if counts != expected:
            raise AssertionError((counts, expected))
        groups["four_heuristics" if enabled else "standard"] = {
            "status": "PASS",
            "configurations": configurations,
            "boundaries_per_configuration": 5,
            **counts,
        }
    for pattern in (
        "*/step*/state_*.npz",
        "*/step*/metadata_*.json",
        "*/step*/rng_*.json",
    ):
        left = {path.relative_to(baseline) for path in baseline.glob(pattern)}
        right = {path.relative_to(candidate) for path in candidate.glob(pattern)}
        if left != right:
            raise AssertionError((pattern, left ^ right))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-omitted", type=Path, required=True)
    parser.add_argument("--new-explicit-current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "PASS",
        "comparison": {
            "new_omitted": compare(args.baseline, args.new_omitted),
            "new_explicit_current": compare(args.baseline, args.new_explicit_current),
        },
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
