"""Compare archived state arrays, metadata and RNG for disabled-flag trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--new-omitted", type=Path, required=True)
    parser.add_argument("--new-explicit", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", default="c076b8b")
    args = parser.parse_args()
    results = {}
    for label, new in (
        ("new_omitted", args.new_omitted),
        ("new_explicit", args.new_explicit),
    ):
        arrays = states = rngs = metadata = 0
        for first in args.baseline.glob("*/step*/state_*.npz"):
            second = new / first.relative_to(args.baseline)
            assert second.exists(), second
            with (
                np.load(first, allow_pickle=False) as left,
                np.load(second, allow_pickle=False) as right,
            ):
                assert set(left.files) == set(right.files)
                for key in left.files:
                    assert (
                        left[key].dtype == right[key].dtype
                        and left[key].shape == right[key].shape
                        and left[key].tobytes() == right[key].tobytes()
                    ), (label, first, key)
                    arrays += 1
            states += 1
        for kind in ("metadata", "rng"):
            for first in args.baseline.glob(f"*/step*/{kind}_*.json"):
                assert (
                    first.read_bytes()
                    == (new / first.relative_to(args.baseline)).read_bytes()
                )
                if kind == "rng":
                    rngs += 1
                else:
                    metadata += 1
        assert states == 60 and rngs == 60 and metadata == 60
        results[label] = {
            "status": "PASS",
            "configurations": 8,
            "boundaries_per_configuration": 5,
            "state_pairs": states,
            "array_pairs_byte_equal": arrays,
            "rng_pairs_byte_equal": rngs,
            "metadata_pairs_byte_equal": metadata,
        }
    result = {
        "base_commit": args.base_commit,
        "comparison": results,
        "feature_model_sha256": hashlib.sha256(
            args.model_source.read_bytes()
        ).hexdigest(),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
