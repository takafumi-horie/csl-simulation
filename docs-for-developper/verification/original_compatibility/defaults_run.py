"""Archive 455b499 and run old/omitted/current parity in isolated processes.

Only the standard library is imported by this orchestrator. Supply a Python
3.11 interpreter and its numerical dependency directory explicitly if needed.
Outputs must be a new directory; existing records are never overwritten.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--base-commit", default="455b499")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    baseline = output / "baseline_source"
    baseline.mkdir()
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.base_commit}^{{commit}}"], cwd=repo, text=True
    ).strip()
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            commit,
            "src",
            "examples/single.yaml",
            "examples/inter.yaml",
        ],
        cwd=repo,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        # git archive contains only the explicitly requested repository paths.
        for member in stream.getmembers():
            target = (baseline / member.name).resolve()
            if not target.is_relative_to(baseline) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive entry: {member.name}")
        stream.extractall(baseline)
    examples = {}
    for name in ("single.yaml", "inter.yaml"):
        old = baseline / "examples" / name
        new = repo / "examples" / name
        if old.read_bytes() != new.read_bytes():
            raise AssertionError(f"Ordinary example changed: {name}")
        examples[name] = {"byte_equal": True, "sha256": sha256(new)}
    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(output / "mpl"),
    )
    for label, source, explicit in (
        ("old", baseline, False),
        ("new_omitted", repo, False),
        ("new_explicit_current", repo, True),
    ):
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path.resolve()) for path in args.dependency_path]
            + [str(source / "src")]
        )
        command = [
            str(args.python),
            str(scripts / "defaults_trajectory.py"),
            "--output-root",
            str(output / label),
        ]
        if explicit:
            command.append("--explicit-current")
        with (output / f"{label}.log").open("x") as log:
            subprocess.run(
                command,
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    comparison = output / "comparison.json"
    subprocess.run(
        [
            str(args.python),
            str(scripts / "defaults_compare.py"),
            "--baseline",
            str(output / "old"),
            "--new-omitted",
            str(output / "new_omitted"),
            "--new-explicit-current",
            str(output / "new_explicit_current"),
            "--output",
            str(comparison),
        ],
        cwd=output,
        env=env,
        check=True,
    )
    versions = subprocess.check_output(
        [
            str(args.python),
            "-c",
            "import json,platform,numpy,scipy,sklearn; "
            "print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,"
            "scipy=scipy.__version__,sklearn=sklearn.__version__)))",
        ],
        cwd=output,
        env=env,
        text=True,
    )
    result = {
        **json.loads(comparison.read_text()),
        "base_commit": commit,
        "environment": json.loads(versions),
        "blas_threads": 1,
        "seed": 21,
        "steps": 4,
        "ordinary_examples": examples,
        "feature_source_sha256": {
            str(path.relative_to(repo)): sha256(path)
            for path in sorted((repo / "src" / "legacy_csl").glob("*.py"))
        },
        "verification_script_sha256": {
            path.name: sha256(path) for path in sorted(scripts.glob("defaults_*.py"))
        },
    }
    with (output / "result.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {output / 'result.json'}")


if __name__ == "__main__":
    main()
