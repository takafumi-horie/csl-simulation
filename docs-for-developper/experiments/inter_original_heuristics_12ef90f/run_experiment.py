"""Run ten native Inter-CSL trials without changing production inference or output."""

import argparse
import datetime
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

import numpy as np
import scipy
import sklearn

from legacy_csl.config import dump_resolved_config, load_config
from legacy_csl.data import load_dataset
from legacy_csl.runner import _learn_inter, validate_data
from legacy_csl.state_io import load_state

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXPECTED_HEAD = "12ef90fc2fc0ffec402087014ba85d1b09359fc5"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_sources(manifest):
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        == EXPECTED_HEAD
    )
    for name, expected in manifest["source_sha256"].items():
        assert sha(REPO / name) == expected, name
    assert sha(HERE / "config.yaml") == manifest["config_sha256"]
    assert sha(__file__) == manifest["runner_sha256"]


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
    memory = {
        line.split(":")[0]: line.split(":")[1].strip()
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))
    }
    return dict(
        cpu_count=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)),
        cgroup_limits=limits,
        memory=memory,
        disk_free_bytes=shutil.disk_usage(REPO).free,
    )


def input_provenance(config):
    arrays, raw = {}, {}
    for agent in ("A", "B"):
        dataset = load_dataset(config.paths.data_root, agent)
        arrays[agent] = {"observations": {}, "truth": {}}
        for attr in ("a", "p", "o", "c"):
            x = dataset.flatten(attr)
            y = dataset.true_categories[attr]
            y = np.asarray(y) if attr == "a" else np.concatenate(y)
            for kind, value in (("observations", x), ("truth", y)):
                arrays[agent][kind][attr] = dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                )
        arrays[agent]["attentions_sha256"] = hashlib.sha256(
            dataset.attentions.tobytes()
        ).hexdigest()
        compact = config.paths.data_root / f"agent{agent}.npz"
        if compact.exists():
            files = [compact]
        else:
            root = config.paths.data_root / f"agent{agent}"
            docs = sorted(root.glob("testss[0-9][0-9][0-9]"))
            if len(docs) >= 40:
                docs = docs[-30:]
            assert len(docs) == dataset.num_documents
            files = [
                root / "testss(001-040)" / f"z{a}_C.csv" for a in ("a", "p", "o", "c")
            ]
            for doc, count in zip(docs, dataset.num_objects, strict=True):
                files.extend(
                    doc / n
                    for n in ("action.csv", "target_object.txt", "object_center.txt")
                )
                files.extend(
                    doc / "image" / f"{prefix}_{index:02}.csv"
                    for index in range(1, int(count) + 1)
                    for prefix in ("object_CNN_PCA_40", "object_RGB_BoF_40")
                )
        for path in files:
            raw[str(path.relative_to(config.paths.data_root))] = dict(
                size_bytes=path.stat().st_size, sha256=sha(path)
            )
    return dict(data_root=str(config.paths.data_root), arrays=arrays, raw_files=raw)


def worker(trial):
    manifest = json.loads((HERE / "execution_manifest.json").read_text())
    verify_sources(manifest)
    config = load_config(HERE / "config.yaml")
    directory = config.paths.output_root / config.run.name / f"exp{trial:03}"
    directory.mkdir(parents=True, exist_ok=False)
    seed = np.random.SeedSequence(config.run.seed).spawn(config.run.experiments)[trial]
    status_path = HERE / "trial_status" / f"exp{trial:03}.json"
    status = dict(
        trial=trial,
        pid=os.getpid(),
        status="running",
        started_utc=now(),
        seed_entropy=seed.entropy,
        seed_spawn_key=list(seed.spawn_key),
        output_directory=str(directory),
    )
    write_json(status_path, status)
    started = time.monotonic()
    try:
        _learn_inter(config, directory, seed, experiment_id=trial)
        checks = {}
        for agent in ("A", "B"):
            state = load_state(directory, agent)
            assert np.isfinite(state.theta).all()
            np.testing.assert_allclose(state.theta.sum(axis=1), 1, atol=1e-12)
            for attr in ("a", "p", "o", "c"):
                for value in (
                    state.pi[attr],
                    state.means[attr],
                    state.covariances[attr],
                ):
                    assert np.isfinite(value).all()
                np.testing.assert_allclose(state.pi[attr].sum(), 1, atol=1e-12)
            checks[agent] = dict(
                checkpoint_sha256=sha(directory / f"state_{agent}.npz"),
                metadata_sha256=sha(directory / f"metadata_{agent}.json"),
                normalized_theta_pi=True,
            )
        assert (directory / "analysis" / "metrics.json").exists()
        assert (directory / "analysis" / "history.csv").exists()
        verify_sources(manifest)
        status.update(status="complete", exit_code=0, final_state_checks=checks)
    except BaseException:
        status.update(status="failed", exit_code=1, traceback=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(status_path, status)


def supervisor(max_parallel):
    config = load_config(HERE / "config.yaml")
    assert (
        config.run.seed == 0
        and config.run.iterations == 500
        and config.run.experiments == 10
    )
    assert config.mode == "inter-csl"
    validate_data(config)
    root = config.paths.output_root / config.run.name
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (HERE / "trial_status").mkdir(exist_ok=False)
    dump_resolved_config(config, root / "resolved_config.yaml")
    source_paths = sorted((REPO / "src/legacy_csl").glob("*.py")) + [
        REPO / "examples/inter_original_heuristics.yaml",
        REPO / "pyproject.toml",
    ]
    manifest = dict(
        head=EXPECTED_HEAD,
        resources=resource_info(),
        supervisor_command=[
            sys.executable,
            str(Path(__file__).resolve()),
            "--max-parallel",
            str(max_parallel),
        ],
        initial_head_verified=True,
        tracked_source_clean_at_launch=subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "src",
                "examples",
                "pyproject.toml",
            ],
            cwd=REPO,
        ).returncode
        == 0,
        initial_worktree_clean_at_request="Confirmed by root before experiment documents were added",
        source_sha256={str(p.relative_to(REPO)): sha(p) for p in source_paths},
        config_sha256=sha(HERE / "config.yaml"),
        runner_sha256=sha(__file__),
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
        ),
        environment={
            k: os.environ.get(k)
            for k in (
                "PYTHONPATH",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MPLCONFIGDIR",
            )
        },
        input_provenance=input_provenance(config),
        started_utc=now(),
        max_parallel=max_parallel,
        probe_trial=0,
        probe_iterations=25,
        output_root=str(root),
        seed_notice="Fresh native trial0..9: SeedSequence(0).spawn(10); distinct from prior confirmation trials10..19",
        commands=[],
        runs=[],
    )
    assert manifest["tracked_source_clean_at_launch"]
    verify_sources(manifest)
    path = HERE / "execution_manifest.json"
    assert not path.exists()
    write_json(path, manifest)
    started = time.monotonic()
    pending = list(range(10))
    active = []
    probe_finished = False
    failed = False
    while pending or active:
        limit = max_parallel if probe_finished else 1
        while pending and len(active) < limit and not failed:
            trial = pending.pop(0)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(trial),
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
            row = dict(
                trial=trial,
                pid=process.pid,
                status="launched",
                seed_entropy=0,
                seed_spawn_key=[trial],
                command=command,
                stdout=str(stdout),
                stderr=str(stderr),
                output_directory=str(root / f"exp{trial:03}"),
                started_elapsed_seconds=time.monotonic() - started,
            )
            manifest["runs"].append(row)
            active.append((process, row))
            write_json(path, manifest)
        marker = root / "exp000/iterations/iter0025/csv/word_length_B.csv"
        if not probe_finished and marker.exists():
            probe_finished = True
            manifest["probe"] = dict(
                iterations=25,
                elapsed_seconds=time.monotonic() - started,
                completed_iteration_csv_marker=str(marker),
                retained_as_trial0=True,
            )
            write_json(path, manifest)
            print(
                json.dumps(dict(event="probe_complete", **manifest["probe"])),
                flush=True,
            )
        for process, row in active[:]:
            code = process.poll()
            if code is not None:
                row.update(
                    exit_code=code,
                    status="complete" if code == 0 else "failed",
                    finished_elapsed_seconds=time.monotonic() - started,
                )
                active.remove((process, row))
                failed = failed or bool(code)
                write_json(path, manifest)
                print(
                    json.dumps(
                        dict(event="trial_complete", trial=row["trial"], exit_code=code)
                    ),
                    flush=True,
                )
        if failed and pending:
            manifest["not_started_after_failure"] = pending
            pending.clear()
        if pending or active:
            time.sleep(2)
    verify_sources(manifest)
    manifest.update(
        finished_utc=now(),
        total_elapsed_seconds=time.monotonic() - started,
        status="failed" if failed else "complete",
    )
    write_json(path, manifest)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(10))
    parser.add_argument("--max-parallel", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(key) == "1", key
    if args.worker is not None:
        worker(args.worker)
        return 0
    return supervisor(args.max_parallel)


if __name__ == "__main__":
    raise SystemExit(main())
