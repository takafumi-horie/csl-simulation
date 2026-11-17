"""Compare the production profile with independently translated original code.

Run with this checkout's src on PYTHONPATH; original sources and real data are
explicit inputs. Result JSON contains per-boundary array/RNG fingerprints.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import sklearn

from reference_oracle import (
    ATTRS,
    original_inputs,
    rng_payload,
    run_reference,
)

from legacy_csl.config import load_config
from legacy_csl.data import load_dataset, load_datasets
from legacy_csl.original import initialize_original_models
from legacy_csl.runner import accept_words, learn
from legacy_csl.state_io import load_state


def pack(rows):
    return (np.concatenate(rows).astype(np.int64),
            np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype(np.int64))


def native_payload(model):
    state = model.state
    result = {"z_a": state.z["a"].copy(), "theta": state.theta.copy()}
    for attr in ATTRS:
        for native, mapping in (("pi", state.pi), ("mean", state.means),
                                ("covariance", state.covariances)):
            result[f"{native}_{attr}"] = mapping[attr].copy()
        if attr != "a":
            result[f"z_{attr}_flat"], result[f"z_{attr}_offsets"] = pack(state.z[attr])
    result["words_flat"], result["word_offsets"] = pack(state.words)
    result["attributes"] = np.asarray(state.attributes, dtype="U1")
    result["vocabulary"] = np.asarray(state.vocabulary, dtype="U20")
    return result


def fingerprints(payload):
    return {
        owner: {
            key: {"shape": list(array.shape), "dtype": array.dtype.str,
                  "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            for key, array in arrays.items()
        }
        for owner, arrays in payload.items()
    }


def assert_equal(expected, actual, location, output):
    assert expected.keys() == actual.keys(), location
    for owner, arrays in expected.items():
        assert arrays.keys() == actual[owner].keys(), (location, owner)
        for key, array in arrays.items():
            other = actual[owner][key]
            equal = (array.shape == other.shape and array.dtype == other.dtype
                     and array.tobytes() == other.tobytes())
            if not equal:
                np.savez_compressed(output / "first_mismatch.npz",
                                    expected=array, actual=other)
                delta = (float(np.max(np.abs(array - other)))
                         if array.shape == other.shape and array.dtype.kind in "fiu"
                         else None)
                raise AssertionError(f"{location}/{owner}/{key}: byte mismatch, max_abs={delta}")


def compare_readers(inputs, datasets, data_root):
    records = []
    for index, (agent, dataset) in enumerate(zip(("A", "B"), datasets)):
        values = inputs[index * 6:(index + 1) * 6]
        assert np.array_equal(dataset.num_objects, values[0])
        assert np.array_equal(dataset.attentions, values[5])
        current = load_dataset(data_root, agent)
        for column, attr in enumerate(ATTRS, 1):
            expected = np.asarray(values[column]) if attr == "a" else np.concatenate(values[column])
            actual = dataset.flatten(attr)
            assert expected.dtype == actual.dtype
            assert expected.shape == actual.shape
            assert expected.tobytes() == actual.tobytes(), (agent, attr)
            truth = dataset.true_categories[attr]
            current_truth = current.true_categories[attr]
            truth = truth if attr == "a" else np.concatenate(truth)
            current_truth = current_truth if attr == "a" else np.concatenate(current_truth)
            assert np.array_equal(truth, current_truth)
            records.append({"agent": agent, "attribute": attr,
                            "shape": list(actual.shape),
                            "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                            "truth_unchanged": True,
                            "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 10])
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--native-output", type=Path,
                        help="also run the native runner and check saved state/RNG; new directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = load_config(args.config)
    assert config.original_heuristics.execution_profile == "original"
    config = replace(config, paths=replace(config.paths, data_root=args.data_root.resolve()))
    inputs = original_inputs(args.original_root, args.data_root)
    datasets = load_datasets(args.data_root, ("A", "B"), execution_profile="original")
    root = Path(__file__).resolve().parents[3]
    result = {"passed": False, "iterations": args.iterations, "seeds": args.seeds,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "reader_comparisons": compare_readers(inputs, datasets, args.data_root),
              "trials": [], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_sha256": {
                  str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((root / "src/legacy_csl").glob("*.py"))},
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "oracle_script_sha256": hashlib.sha256(
                  Path(__file__).with_name("reference_oracle.py").read_bytes()).hexdigest()}
    if args.native_output is not None:
        args.native_output.mkdir(parents=True, exist_ok=False)
    try:
        for seed in args.seeds:
            records = {}

            def reference(stage, iteration, payload):
                key = f"{iteration}/{stage}"
                assert key not in records
                records[key] = payload

            manifest = run_reference(
                args.original_root, inputs, seed, args.iterations, reference,
                args.output / f"translated_seed{seed}.py.txt")
            trial_config = replace(config, run=replace(config.run, seed=seed, experiments=1))
            rng = np.random.RandomState(seed)
            before = rng_payload(np.random.get_state())
            models = initialize_original_models(datasets, trial_config, rng=rng, experiment_id=0)
            trial = {"seed": seed, "reference": manifest, "boundaries": []}
            result["trials"].append(trial)

            def compare(stage, iteration, proposal=None):
                key = f"{iteration}/{stage}"
                actual = {agent: native_payload(model)
                          for agent, model in zip(("A", "B"), models)}
                if proposal is not None:
                    actual["proposal"] = {"words": np.asarray(proposal, dtype=np.int64)}
                actual["rng"] = rng_payload(rng.get_state())
                assert_equal(records.pop(key), actual, f"seed{seed}/{key}", args.output)
                trial["boundaries"].append({"stage": stage, "iteration": iteration,
                                            "fingerprints": fingerprints(actual)})

            compare("initial", 0)
            for iteration in range(1, args.iterations + 1):
                for index, agent in enumerate(("A", "B")):
                    proposal = models[1 - index].draw_words()
                    compare(f"proposal_{agent}", iteration, proposal)
                    accepted = accept_words(models[index], proposal, "mh")
                    models[index].state.words = accepted
                    compare(f"accepted_{agent}", iteration)
                    models[index].step(accepted)
                    compare(f"updated_{agent}", iteration)
            assert not records, records.keys()
            if args.native_output is not None:
                native_config = replace(
                    trial_config,
                    run=replace(trial_config.run, name=f"seed{seed}", iterations=args.iterations),
                    paths=replace(trial_config.paths, output_root=args.native_output.resolve()))
                directory, = learn(native_config)
                saved = {agent: native_payload(SimpleNamespace(state=load_state(directory, agent)))
                         for agent in ("A", "B")}
                expected = {agent: native_payload(model)
                            for agent, model in zip(("A", "B"), models)}
                assert_equal(expected, saved, f"native_runner/seed{seed}", args.output)
                compressed = directory / "rng_boundaries.jsonl.gz"
                if compressed.exists():
                    with gzip.open(compressed, "rt", encoding="utf-8") as stream:
                        boundaries = [json.loads(line) for line in stream]
                else:
                    boundaries = [json.loads(line) for line in
                                  (directory / "rng_boundaries.jsonl").read_text().splitlines()]
                by_stage = {(item["iteration"], item["stage"]): item
                            for item in trial["boundaries"]}
                for item in boundaries:
                    stage = ("initial" if item["boundary"] == "initialized"
                             else item["boundary"].replace("after_", "updated_"))
                    state = (item["algorithm"], np.asarray(item["keys"], dtype=np.uint32),
                             item["position"], item["has_gaussian"], item["cached_gaussian"])
                    assert fingerprints({"rng": rng_payload(state)})["rng"] == (
                        by_stage[(item["iteration"], stage)]["fingerprints"]["rng"])
                assert len(boundaries) == 1 + 2 * args.iterations
                provenance = json.loads((directory / "execution_provenance.json").read_text())
                files = [entry for group in provenance["inputs"] for entry in group["files"]]
                for entry in files:
                    assert entry["sha256"] == hashlib.sha256(
                        (args.data_root / entry["path"]).read_bytes()).hexdigest()
                trial["native_runner"] = {"passed": True,
                                          "checkpoint_arrays": sum(map(len, saved.values())),
                                          "rng_boundaries": len(boundaries),
                                          "input_hashes_verified": len(files)}
            assert_equal({"rng": before}, {"rng": rng_payload(np.random.get_state())},
                         "global_rng_noninterference", args.output)
            trial["global_rng_unchanged"] = True
            print(f"seed {seed}: {len(trial['boundaries'])} boundaries passed", flush=True)
        result["passed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
