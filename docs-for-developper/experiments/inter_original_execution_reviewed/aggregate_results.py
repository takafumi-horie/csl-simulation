"""Independently compute final-state ARI and compare frozen reference artifacts.

No learning is performed. Native numerical arrays must match all ten frozen
original states exactly; string/word metadata use the explicit mapping below.
"""

import argparse
import csv
import gzip
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np
from sklearn.metrics import adjusted_rand_score

from legacy_csl.config import load_config
from legacy_csl.data import ATTRIBUTES, load_datasets

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRIALS = range(10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def flatten(rows):
    return np.asarray(rows) if isinstance(rows, np.ndarray) else np.concatenate(rows)


def select(rows, dataset, attr):
    if attr == "a":
        return np.asarray(rows)
    return np.asarray(
        [row[index] for row, index in zip(rows, dataset.attentions, strict=True)]
    )


def assert_bytes(left, right, context):
    if not (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    ):
        raise AssertionError(
            (
                context,
                "dtype/shape/bytes differ",
                left.dtype,
                right.dtype,
                left.shape,
                right.shape,
            )
        )


def native_state(directory, agent, dataset):
    with np.load(directory / f"state_{agent}.npz", allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    metadata = read_json(directory / f"metadata_{agent}.json")
    for key, value in saved.items():
        if not np.isfinite(value).all():
            raise AssertionError((directory, agent, key, "nonfinite"))
    expected_offsets = np.r_[0, np.cumsum(dataset.num_objects)]
    assignments = {"a": saved["z_a"]}
    if saved["z_a"].shape != (30,):
        raise AssertionError("Expected 30 action assignments")
    for attr in ATTRIBUTES[1:]:
        np.testing.assert_array_equal(saved[f"z_{attr}_offsets"], expected_offsets)
        assignments[attr] = tuple(
            np.split(saved[f"z_{attr}_flat"], expected_offsets[1:-1])
        )
    for attr in ATTRIBUTES:
        z = flatten(assignments[attr])
        if not np.issubdtype(z.dtype, np.integer) or not ((z >= 0) & (z < 10)).all():
            raise AssertionError((agent, attr, "invalid assignments"))
        np.testing.assert_allclose(saved[f"pi_{attr}"].sum(), 1, rtol=0, atol=1e-12)
        covariance = saved[f"covariance_{attr}"]
        if not (np.linalg.eigvalsh(covariance) > 0).all():
            raise AssertionError((agent, attr, "nonpositive covariance"))
    np.testing.assert_array_equal(saved["word_offsets"], np.arange(31) * 4)
    np.testing.assert_allclose(saved["theta"].sum(axis=1), 1, rtol=0, atol=1e-12)
    if saved["theta"].shape != (40, 13):
        raise AssertionError("Expected theta shape (40,13)")
    if metadata["vocabulary"] != list(map(str, range(13))):
        raise AssertionError("Expected canonical vocabulary IDs")
    canonical = {
        key: value
        for key, value in saved.items()
        if not key.endswith("_offsets") and key != "words_flat"
    }
    canonical.update(
        words=saved["words_flat"].reshape(30, 4),
        attributes=np.asarray(metadata["attributes"], dtype="<U1"),
        vocabulary=np.asarray(list(map(int, metadata["vocabulary"])), dtype=np.int64),
    )
    # The reference stores words as a matrix and metadata as arrays. All other
    # numerical arrays share their native names, dtypes, shapes, and bytes.
    return assignments, canonical


def metrics(assignments, datasets):
    rows = []
    for agent, dataset in datasets.items():
        for attr in ATTRIBUTES:
            truth, inferred = dataset.true_categories[attr], assignments[agent][attr]
            for scope in ("all", "attended"):
                left = (
                    flatten(truth) if scope == "all" else select(truth, dataset, attr)
                )
                right = (
                    flatten(inferred)
                    if scope == "all"
                    else select(inferred, dataset, attr)
                )
                rows.append(
                    {
                        "agent": agent,
                        "attribute": attr,
                        "scope": scope,
                        "ari": float(adjusted_rand_score(left, right)),
                    }
                )
    return rows


def add_agent_means(records):
    extra = []
    for condition in sorted({row["condition"] for row in records}):
        for trial in TRIALS:
            for scope in ("all", "attended"):
                for attr in ATTRIBUTES:
                    pair = [
                        row["ari"]
                        for row in records
                        if (
                            row["condition"],
                            row["trial"],
                            row["scope"],
                            row["attribute"],
                        )
                        == (condition, trial, scope, attr)
                    ]
                    if len(pair) != 2:
                        raise AssertionError((condition, trial, scope, attr, pair))
                    extra.append(
                        {
                            "condition": condition,
                            "trial": trial,
                            "scope": scope,
                            "agent": "AB_mean",
                            "attribute": attr,
                            "ari": statistics.mean(pair),
                        }
                    )
    return records + extra


def summarize(records):
    rows = []
    for condition in sorted({row["condition"] for row in records}):
        for scope in ("attended", "all"):
            for agent in ("A", "B", "AB_mean"):
                for attr in ATTRIBUTES:
                    selected = sorted(
                        (
                            row
                            for row in records
                            if (
                                row["condition"],
                                row["scope"],
                                row["agent"],
                                row["attribute"],
                            )
                            == (condition, scope, agent, attr)
                        ),
                        key=lambda row: row["trial"],
                    )
                    if [row["trial"] for row in selected] != list(TRIALS):
                        raise AssertionError("Incomplete trial set")
                    values = [row["ari"] for row in selected]
                    rows.append(
                        {
                            "condition": condition,
                            "scope": scope,
                            "agent": agent,
                            "attribute": attr,
                            "n": 10,
                            "mean": statistics.mean(values),
                            "sample_sd": statistics.stdev(values),
                            "values": values,
                        }
                    )
    return rows


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "execution_manifest.json"
    )
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument(
        "--four-heuristics-results",
        type=Path,
        default=HERE.parent / "inter_original_heuristics_12ef90f/results.json",
    )
    parser.add_argument(
        "--paper-reference",
        type=Path,
        default=HERE.parent / "inter_original_heuristics_12ef90f/paper_reference.json",
    )
    parser.add_argument("--destination", type=Path, default=HERE)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if (
        manifest["status"] != "complete"
        or not manifest["source_and_input_postrun_verified"]
    ):
        raise AssertionError("The native experiment must complete successfully first")
    if sorted(row["trial"] for row in manifest["runs"]) != list(TRIALS):
        raise AssertionError("Expected exactly ten completed trials")
    if any(
        row["exit_code"] != 0 or row["status"] != "complete" for row in manifest["runs"]
    ):
        raise AssertionError("A trial failed")
    for name, expected in manifest["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise AssertionError((name, "source changed"))
    if sha(manifest["config_path"]) != manifest["config_sha256"]:
        raise AssertionError("Configuration changed")
    config = load_config(Path(manifest["config_path"]))
    if (config.run.seed, config.run.experiments, config.run.iterations) != (0, 10, 500):
        raise AssertionError("Unexpected experiment settings")
    output = Path(manifest["output_root"])
    if load_config(output / "resolved_config.yaml") != config:
        raise AssertionError("Native resolved configuration differs")
    datasets = dict(
        zip(
            ("A", "B"),
            load_datasets(
                config.paths.data_root, ("A", "B"), execution_profile="original"
            ),
            strict=True,
        )
    )
    reference_summary = read_json(args.reference_summary)
    four = read_json(args.four_heuristics_results)
    paper = read_json(args.paper_reference)
    records, validations, reference_pairs = [], [], 0
    with tarfile.open(args.reference_archive, "r:gz") as bundle:
        for trial in TRIALS:
            directory = output / f"exp{trial:03}"
            status = read_json(
                args.manifest.parent / "trial_status" / f"exp{trial:03}.json"
            )
            if status["status"] != "complete" or status["effective_seed"] != trial:
                raise AssertionError((trial, "worker status"))
            for name, expected in status["output_sha256"].items():
                if sha(directory / name) != expected:
                    raise AssertionError((trial, name, "output changed"))
            with (directory / "analysis/history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            if [int(row["iteration"]) for row in history] != list(range(1, 501)):
                raise AssertionError((trial, "history length/order"))
            provenance = read_json(directory / "execution_provenance.json")
            if (
                provenance["effective_seed"],
                provenance["experiment_id"],
                provenance["execution_profile"],
            ) != (trial, trial, "original"):
                raise AssertionError((trial, "provenance"))
            with gzip.open(directory / "rng_boundaries.jsonl.gz", "rt") as stream:
                boundaries = [json.loads(row) for row in stream]
            expected_boundaries = [(0, "initialized")] + [
                (iteration, f"after_{agent}")
                for iteration in range(1, 501)
                for agent in ("A", "B")
            ]
            if [
                (row["iteration"], row["boundary"]) for row in boundaries
            ] != expected_boundaries:
                raise AssertionError((trial, "RNG boundary sequence"))
            prefix = f"outputs/ari-controls/mhr/trial{trial:02}"
            frozen = json.load(bundle.extractfile(f"{prefix}/result.json"))
            if (frozen["condition"], frozen["trial"], frozen["iterations"]) != (
                "mhr",
                trial,
                500,
            ):
                raise AssertionError((trial, "reference trial identity"))
            assignments = {}
            for agent in ("A", "B"):
                assignments[agent], canonical = native_state(
                    directory, agent, datasets[agent]
                )
                stream = io.BytesIO(
                    bundle.extractfile(f"{prefix}/original_state_{agent}.npz").read()
                )
                with np.load(stream, allow_pickle=False) as reference:
                    if set(reference.files) != set(canonical):
                        raise AssertionError((trial, agent, "reference keys differ"))
                    for key in reference.files:
                        assert_bytes(
                            reference[key], canonical[key], (trial, agent, key)
                        )
                        reference_pairs += 1
                for attr in ATTRIBUTES:
                    with (directory / "csv" / f"z{attr}_{agent}.csv").open(
                        newline=""
                    ) as stream:
                        csv_z = np.asarray(
                            [int(value) for row in csv.reader(stream) for value in row]
                        )
                    np.testing.assert_array_equal(
                        csv_z, flatten(assignments[agent][attr])
                    )
                    final_csv = (
                        directory / "iterations/iter0500/csv" / f"z{attr}_{agent}.csv"
                    )
                    if (
                        final_csv.read_bytes()
                        != (directory / "csv" / final_csv.name).read_bytes()
                    ):
                        raise AssertionError(
                            (trial, agent, attr, "final snapshot differs")
                        )
            native = read_json(directory / "analysis/metrics.json")
            trial_values = metrics(assignments, datasets)
            for row in trial_values:
                agent, attr, scope = row["agent"], row["attribute"], row["scope"]
                expected = frozen["history"][-1]["metrics"][agent][attr][scope]
                summary_row = next(
                    item
                    for item in reference_summary["metrics"]
                    if (item["agent"], item["attribute"], item["metric"])
                    == (agent, attr, scope)
                )
                if row["ari"] != expected or expected != summary_row["values"][trial]:
                    raise AssertionError(
                        (trial, agent, attr, scope, "reference ARI differs")
                    )
                if scope == "attended":
                    key = f"ari_truth_{agent}_{attr}"
                    if row["ari"] != float(native[key]) or row["ari"] != float(
                        history[-1][key]
                    ):
                        raise AssertionError((trial, key, "native final ARI differs"))
                records.append({"condition": "native_original", "trial": trial, **row})
                records.append({"condition": "frozen_original", "trial": trial, **row})
            validations.append(
                {
                    "trial": trial,
                    "effective_seed": trial,
                    "status": "PASS",
                    "reference_arrays_equal": 40,
                    "history_rows": 500,
                    "rng_boundaries": len(boundaries),
                    "output_sha256": status["output_sha256"],
                }
            )
            print(
                f"PASS exp{trial:03}: frozen full state and independent final ARI",
                flush=True,
            )
    if reference_pairs != 400:
        raise AssertionError((reference_pairs, "expected 400 state-array pairs"))
    for row in four["trial_metrics"]:
        if row["agent"] in ("A", "B"):
            records.append({"condition": "four_heuristics", **row})
    records = add_agent_means(records)
    result = {
        "status": "PASS",
        "completed_utc": datetime.now(UTC).isoformat(),
        "execution_manifest_sha256": sha(args.manifest),
        "aggregator_sha256": sha(__file__),
        "native_output_root": str(output),
        "runtime": manifest["runtime"],
        "sample_sd_ddof": 1,
        "trials": list(TRIALS),
        "iterations": 500,
        "summary": summarize(records),
        "trial_metrics": records,
        "validations": validations,
        "reference_array_pairs_byte_equal": reference_pairs,
        "reference_ari_values_exact": 160,
        "paper_reference": paper,
        "comparators": {
            "original": {
                "archive": str(args.reference_archive.resolve()),
                "archive_sha256": sha(args.reference_archive),
                "summary_sha256": sha(args.reference_summary),
                "seeds": list(TRIALS),
                "reused_reference_trials": [0, 1, 2],
                "added_reference_trials": list(range(3, 10)),
            },
            "four_heuristics": {
                "results_sha256": sha(args.four_heuristics_results),
                "git_head": four["git_head"],
                "seed_rule": "SeedSequence(0).spawn(10), then spawn(2)",
                "paired_trajectories": False,
            },
            "paper": {"reference_sha256": sha(args.paper_reference)},
        },
        "interpretation": [
            "Mean and sample SD use all ten trials (ddof=1); no best-trial or best-iteration selection.",
            "Attended evaluates 30 scenes/objects; all evaluates 55 objects for p/o/c and 30 scenes for a.",
            "AB_mean averages A/B within each trial; it does not create 20 independent samples.",
            "Frozen original shares modern seeds 0..9; four-heuristic spawned seeds are not paired trajectories.",
            "Paper truth is user-confirmed; paper object subset and A/B aggregation remain unresolved.",
            "Exact agreement is with the fixed modernized original, not historical Python 2 or paper ARI.",
        ],
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    with (args.destination / "results.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(args.destination / "trial_metrics.csv", records)
    write_csv(args.destination / "summary.csv", result["summary"])
    print(
        "PASS: all ten trials aggregated with exact frozen-state comparison", flush=True
    )


if __name__ == "__main__":
    main()
