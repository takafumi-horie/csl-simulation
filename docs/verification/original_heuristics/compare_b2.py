"""Compare the frozen external B2 adapter with production at each RNG boundary."""

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from legacy_csl.config import OriginalHeuristics
from legacy_csl.data import load_dataset
from legacy_csl.model import CslModel
from legacy_csl.runner import accept_words
from legacy_csl.state_io import save_state

ADAPTER_SHA256 = "0a0cddafcdcac09fe7ad78656a85c9dbf9a7c670c7180c37d49b0c343b940dba"


def compare_saved(left_root, right_root, trial, step):
    arrays = 0
    for agent in ("A", "B"):
        with (
            np.load(left_root / f"state_{agent}.npz", allow_pickle=False) as left,
            np.load(right_root / f"state_{agent}.npz", allow_pickle=False) as right,
        ):
            assert set(left.files) == set(right.files)
            for key in left.files:
                assert (
                    left[key].dtype == right[key].dtype
                    and left[key].shape == right[key].shape
                    and left[key].tobytes() == right[key].tobytes()
                ), (trial, step, agent, key)
                arrays += 1
        for key in ("metadata", "rng"):
            assert (left_root / f"{key}_{agent}.json").read_bytes() == (
                right_root / f"{key}_{agent}.json"
            ).read_bytes(), (trial, step, agent, key)
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adapter-sha256", default=ADAPTER_SHA256)
    parser.add_argument("--trials", type=int, nargs="+", default=[0, 2, 10])
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or any(trial < 0 for trial in args.trials):
        parser.error("steps must be positive and trials nonnegative")
    # The external audit checkout is read-only, including Python bytecode caches.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.adapter_dir.resolve()))
    adapter = importlib.import_module("exploratory_phase2_driver")
    assert Path(adapter.__file__).resolve().parent == args.adapter_dir.resolve()
    assert (
        hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
        == args.adapter_sha256
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    datasets = {agent: load_dataset(args.data_root, agent) for agent in ("A", "B")}
    compared = 0
    for trial in args.trials:
        adapter_args = SimpleNamespace(
            condition="B2current_init",
            trial=trial,
            iterations=args.steps,
            data_root=args.data_root,
            original_root=args.original_root,
        )
        config = adapter.experiment_config(adapter_args, args.output_root)
        old = adapter.initialize_models(adapter_args, datasets, config)[0]
        enabled = replace(
            config,
            original_heuristics=OriginalHeuristics(
                theta_mean_samples=100,
                inference_theta="raw",
                gaussian_mean_samples=100,
                empty_component_mean="original_uniform",
            ),
        )
        seeds = np.random.SeedSequence(0).spawn(trial + 1)[trial].spawn(2)
        new = {
            agent: CslModel(datasets[agent], enabled, rng=np.random.default_rng(seed))
            for agent, seed in zip(("A", "B"), seeds, strict=True)
        }
        paired = (("adapter", old), ("production", new))
        proposals = {
            tag: {
                agent: tuple(row.copy() for row in model.state.words)
                for agent, model in models.items()
            }
            for tag, models in paired
        }
        for step in range(args.steps + 1):
            boundary = args.output_root / f"trial{trial:02}" / f"step{step}"
            for tag, models in paired:
                target = boundary / tag
                for agent, model in models.items():
                    save_state(model.state, target, agent)
                    (target / f"rng_{agent}.json").write_text(
                        json.dumps(
                            model.rng.bit_generator.state,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
            compared += compare_saved(
                boundary / "adapter", boundary / "production", trial, step
            )
            if step == args.steps:
                break
            for tag, models in paired:
                for listener, speaker in (("A", "B"), ("B", "A")):
                    model = models[listener]
                    model.step(accept_words(model, proposals[tag][speaker], "mh"))
                    proposals[tag][listener] = model.draw_words()
    model_source = Path(sys.modules[CslModel.__module__].__file__)
    result = {
        "status": "PASS",
        "adapter_sha256": args.adapter_sha256,
        "trials": args.trials,
        "boundaries_per_trial": args.steps + 1,
        "agents": 2,
        "arrays_byte_equal": compared,
        "metadata_and_rng_byte_equal": True,
        "feature_model_sha256": hashlib.sha256(model_source.read_bytes()).hexdigest(),
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
