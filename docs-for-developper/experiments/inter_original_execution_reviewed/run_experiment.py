"""Launch ten native original-profile trials and preserve their provenance.

Use --prepare-only for configuration/input checks without starting learning.
No audit adapter or original source is imported by this launcher.
"""

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for thread_key in THREAD_KEYS:
    os.environ[thread_key] = "1"

# NumPy and its consumers must load after the numeric thread limits are fixed.
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from legacy_csl.config import dump_resolved_config, load_config  # noqa: E402
from legacy_csl.data import ATTRIBUTES, dataset_input_files, load_datasets  # noqa: E402
from legacy_csl.runner import _learn_original, validate_data  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resource_info():
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split("::")[-1].lstrip("/")
    limits = {}
    for directory in [cgroup, *cgroup.parents]:
        if str(directory).startswith("/sys/fs/cgroup"):
            for filename in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
                path = directory / filename
                if path.exists():
                    limits[str(path)] = path.read_text().strip()
    return {
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_limits": limits,
        "disk_free_bytes": shutil.disk_usage(REPO).free,
        "memory": [
            line
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))
        ],
    }


def fingerprint(array):
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def input_provenance(config):
    datasets = load_datasets(
        config.paths.data_root, ("A", "B"), execution_profile="original"
    )
    arrays, files = {}, {}
    for agent, dataset in zip(("A", "B"), datasets, strict=True):
        if dataset.true_categories is None:
            raise ValueError(f"Ground truth is required for agent {agent}")
        if dataset.num_documents != 30 or sum(dataset.num_objects) != 55:
            raise ValueError("This protocol requires 30 scenes and 55 objects")
        arrays[agent] = {
            "observations": {
                attr: fingerprint(dataset.flatten(attr)) for attr in ATTRIBUTES
            },
            "truth": {
                attr: fingerprint(
                    dataset.true_categories[attr]
                    if attr == "a"
                    else np.concatenate(dataset.true_categories[attr])
                )
                for attr in ATTRIBUTES
            },
            "attentions": fingerprint(dataset.attentions),
            "num_objects": fingerprint(dataset.num_objects),
        }
        for path in dataset_input_files(
            config.paths.data_root, agent, execution_profile="original"
        ):
            files[str(path.relative_to(config.paths.data_root))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha(path),
            }
    return {
        "data_root": str(config.paths.data_root),
        "arrays": arrays,
        "raw_files": files,
    }


def verify_sources(manifest):
    for name, expected in manifest["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise AssertionError(f"Source changed: {name}")
    if sha(manifest["config_path"]) != manifest["config_sha256"]:
        raise AssertionError("Effective configuration changed")


def worker(trial, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    verify_sources(manifest)
    config = load_config(Path(manifest["config_path"]))
    validate_data(config)
    seed = config.run.seed + trial
    directory = Path(manifest["output_root"]) / f"exp{trial:03}"
    directory.mkdir(parents=True, exist_ok=False)
    status_path = manifest_path.parent / "trial_status" / f"exp{trial:03}.json"
    status = {
        "trial": trial,
        "effective_seed": seed,
        "seed_rule": "run.seed + experiment_id",
        "pid": os.getpid(),
        "status": "running",
        "started_utc": now(),
        "output_directory": str(directory),
    }
    write_json(status_path, status)
    started = time.monotonic()
    try:
        _learn_original(config, directory, seed, experiment_id=trial)
        files = {}
        for agent in ("A", "B"):
            with np.load(directory / f"state_{agent}.npz", allow_pickle=False) as state:
                for key in state.files:
                    if not np.isfinite(state[key]).all():
                        raise AssertionError((trial, agent, key, "nonfinite"))
                np.testing.assert_allclose(state["theta"].sum(axis=1), 1, atol=1e-12)
                for attr in ATTRIBUTES:
                    np.testing.assert_allclose(state[f"pi_{attr}"].sum(), 1, atol=1e-12)
            for name in (f"state_{agent}.npz", f"metadata_{agent}.json"):
                files[name] = sha(directory / name)
        for name in (
            "execution_provenance.json",
            "rng_boundaries.jsonl.gz",
            "analysis/history.csv",
            "analysis/metrics.json",
        ):
            files[name] = sha(directory / name)
        verify_sources(manifest)
        status.update(status="complete", exit_code=0, output_sha256=files)
    except BaseException:
        status.update(status="failed", exit_code=1, traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(status_path, status)


def supervisor(args):
    artifact = args.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    if (artifact / "execution_manifest.json").exists():
        raise FileExistsError(artifact / "execution_manifest.json")
    source_config = REPO / "examples/inter_original_heuristics.yaml"
    config = load_config(source_config)
    config = replace(
        config,
        run=replace(
            config.run, name=args.run_name, experiments=10, iterations=500, seed=0
        ),
        paths=replace(
            config.paths,
            data_root=args.data_root.resolve(),
            output_root=args.output_parent.resolve(),
        ),
    )
    if (
        config.mode != "inter-csl"
        or config.original_heuristics.execution_profile != "original"
    ):
        raise ValueError("Expected the original INTER example")
    validate_data(config)
    config_path = artifact / "config.yaml"
    dump_resolved_config(config, config_path)
    root = config.paths.output_root / config.run.name
    plan = {
        "prepared_utc": now(),
        "config_path": str(config_path),
        "config_sha256": sha(config_path),
        "source_example_sha256": sha(source_config),
        "output_root": str(root),
        "trials": list(range(10)),
        "iterations": 500,
        "effective_seeds": list(range(10)),
        "max_parallel": args.max_parallel,
        "resources": resource_info(),
        "input_provenance": input_provenance(config),
    }
    write_json(artifact / "launch_plan.json", plan)
    if args.prepare_only:
        print(
            "Prepared configuration and launch_plan.json; no learning started",
            flush=True,
        )
        return 0
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (artifact / "trial_status").mkdir(exist_ok=False)
    dump_resolved_config(config, root / "resolved_config.yaml")
    paths = sorted((REPO / "src/legacy_csl").glob("*.py")) + [
        source_config,
        REPO / "pyproject.toml",
        Path(__file__).resolve(),
        HERE / "aggregate_results.py",
        HERE / "protocol.md",
    ]
    manifest = {
        **plan,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_sha256": {str(path.relative_to(REPO)): sha(path) for path in paths},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (*THREAD_KEYS, "PYTHONPATH", "MPLCONFIGDIR")
        },
        "supervisor_command": [sys.executable, *sys.argv],
        "started_utc": now(),
        "status": "running",
        "runs": [],
        "seed_rule": "run.seed + experiment_id",
    }
    manifest_path = artifact / "execution_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    write_json(manifest_path, manifest)
    started = time.monotonic()
    pending, active, failed = list(range(10)), [], False
    while pending or active:
        while pending and len(active) < args.max_parallel and not failed:
            trial = pending.pop(0)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(trial),
                "--manifest",
                str(manifest_path),
            ]
            stdout = root / "logs" / f"exp{trial:03}.stdout.log"
            stderr = root / "logs" / f"exp{trial:03}.stderr.log"
            with stdout.open("x") as out, stderr.open("x") as err:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            row = {
                "trial": trial,
                "effective_seed": trial,
                "pid": process.pid,
                "command": command,
                "stdout": str(stdout),
                "stderr": str(stderr),
                "started_utc": now(),
                "status": "running",
            }
            manifest["runs"].append(row)
            active.append((process, row))
            write_json(manifest_path, manifest)
        for process, row in active[:]:
            code = process.poll()
            if code is not None:
                row.update(
                    exit_code=code,
                    status="complete" if code == 0 else "failed",
                    finished_utc=now(),
                )
                active.remove((process, row))
                failed = failed or bool(code)
                write_json(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "event": "trial_complete",
                            "trial": row["trial"],
                            "exit_code": code,
                        }
                    ),
                    flush=True,
                )
        if failed and pending:
            manifest["not_started_after_failure"] = pending
            pending = []
        if pending or active:
            time.sleep(2)
    verify_sources(manifest)
    if input_provenance(config) != manifest["input_provenance"]:
        raise AssertionError("Input files or processed arrays changed")
    manifest.update(
        status="failed" if failed else "complete",
        finished_utc=now(),
        total_elapsed_seconds=time.monotonic() - started,
        source_and_input_postrun_verified=True,
    )
    write_json(manifest_path, manifest)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(10))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=HERE)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-parent", type=Path, default=REPO / "outputs")
    parser.add_argument("--run-name", default="inter_original_execution_reviewed")
    parser.add_argument("--max-parallel", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        if args.manifest is None:
            parser.error("--worker requires --manifest")
        worker(args.worker, args.manifest.resolve())
        return 0
    if args.data_root is None:
        parser.error("--data-root is required for preparation or launch")
    return supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
