"""Independently audit native checkpoints/CSV histories and aggregate truth ARI.

Run after all ten native INTER trials have completed. No learning is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score

from legacy_csl.config import load_config
from legacy_csl.data import ATTRIBUTES, load_dataset
from legacy_csl.state_io import load_state

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
EXPECTED_HEAD = "12ef90fc2fc0ffec402087014ba85d1b09359fc5"
TRIALS = range(10)
ITERATIONS = 500


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def metric_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def flatten(rows):
    return np.concatenate(rows) if isinstance(rows, tuple) else np.asarray(rows)


def attended(rows, dataset, attr):
    if attr == "a":
        return np.asarray(rows)
    return np.asarray(
        [row[index] for row, index in zip(rows, dataset.attentions, strict=True)]
    )


def hash_dataset(dataset):
    return {
        "observations": {
            a: hashlib.sha256(dataset.flatten(a).tobytes()).hexdigest()
            for a in ATTRIBUTES
        },
        "truth": {
            a: hashlib.sha256(flatten(dataset.true_categories[a]).tobytes()).hexdigest()
            for a in ATTRIBUTES
        },
        "attentions": hashlib.sha256(dataset.attentions.tobytes()).hexdigest(),
        "num_objects": hashlib.sha256(dataset.num_objects.tobytes()).hexdigest(),
    }


def csv_names(agent):
    return (
        {
            f"{prefix}_{attr}_{agent}.csv"
            for attr in ATTRIBUTES
            for prefix in ("pi", "Mu", "Sig")
        }
        | {f"z{attr}_{agent}.csv" for attr in ATTRIBUTES}
        | {
            f"{prefix}_{agent}.csv"
            for prefix in ("theta", "uttered_words", "F", "W_list", "word_length")
        }
    )


def csv_assignments(directory, agent, dataset):
    assignments = {}
    for attr in ATTRIBUTES:
        rows = read_csv(directory / f"z{attr}_{agent}.csv")
        assert len(rows) == dataset.num_documents
        values = tuple(
            np.asarray([int(x) for x in row], dtype=np.int64) for row in rows
        )
        lengths = (
            np.ones(dataset.num_documents, dtype=int)
            if attr == "a"
            else dataset.num_objects
        )
        assert [len(row) for row in values] == list(lengths)
        assert all(((row >= 0) & (row < 10)).all() for row in values)
        assignments[attr] = np.concatenate(values) if attr == "a" else values
    return assignments


def independent_ari(assignments, datasets):
    """Do not call production analysis.category_metrics."""
    values = {}
    for agent, dataset in datasets.items():
        values[agent] = {}
        for attr in ATTRIBUTES:
            z, truth = assignments[agent][attr], dataset.true_categories[attr]
            assert flatten(z).shape == flatten(truth).shape
            values[agent][attr] = {
                "all": float(adjusted_rand_score(flatten(truth), flatten(z))),
                "attended": float(
                    adjusted_rand_score(
                        attended(truth, dataset, attr), attended(z, dataset, attr)
                    )
                ),
                "occupied": len(np.unique(flatten(z))),
            }
    native = {}
    for attr in ATTRIBUTES:
        native[f"ari_truth_{attr}"] = values["A"][attr]["attended"]
        for agent in ("A", "B"):
            native[f"ari_truth_{agent}_{attr}"] = values[agent][attr]["attended"]
        native[f"ari_agents_{attr}"] = float(
            adjusted_rand_score(
                attended(assignments["A"][attr], datasets["A"], attr),
                attended(assignments["B"][attr], datasets["B"], attr),
            )
        )
    return values, native


def assert_native_ari(expected, actual):
    for key, value in expected.items():
        assert math.isfinite(float(actual[key]))
        assert abs(value - float(actual[key])) < 1e-12, (key, value, actual[key])


def intermediate_native_ari(assignments, datasets):
    result = {}
    for attr in ATTRIBUTES:
        selected = {
            a: attended(z[attr], datasets[a], attr) for a, z in assignments.items()
        }
        for agent in ("A", "B"):
            truth = attended(
                datasets[agent].true_categories[attr], datasets[agent], attr
            )
            result[f"ari_truth_{agent}_{attr}"] = float(
                adjusted_rand_score(truth, selected[agent])
            )
        result[f"ari_truth_{attr}"] = result[f"ari_truth_A_{attr}"]
        result[f"ari_agents_{attr}"] = float(
            adjusted_rand_score(selected["A"], selected["B"])
        )
    return result


def validate_checkpoint(directory, agent, dataset):
    path = directory / f"state_{agent}.npz"
    with np.load(path, allow_pickle=False) as saved:
        for key in saved.files:
            array = saved[key]
            assert np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(), (
                path,
                key,
            )
            if key.startswith(("z_", "word")):
                assert np.issubdtype(array.dtype, np.integer), (path, key)
        for attr in ATTRIBUTES[1:]:
            assert np.array_equal(
                saved[f"z_{attr}_offsets"], np.r_[0, np.cumsum(dataset.num_objects)]
            )
        assert np.array_equal(saved["word_offsets"], np.arange(31) * 4)
    state = load_state(directory, agent)
    assert len(state.vocabulary) == 13 and len(set(state.vocabulary)) == 13
    assert len(state.words) == len(state.attributes) == 30
    assert state.theta.shape == (40, 13) and (state.theta >= 0).all()
    np.testing.assert_allclose(state.theta.sum(axis=1), 1, atol=1e-12, rtol=0)
    for words, attrs in zip(state.words, state.attributes, strict=True):
        assert words.shape == (4,) and ((words >= 0) & (words < 13)).all()
        assert len(attrs) == 4 and set(attrs) == set(ATTRIBUTES)
    for attr in ATTRIBUTES:
        dimension = dataset.flatten(attr).shape[1]
        assert state.pi[attr].shape == (10,) and (state.pi[attr] >= 0).all()
        np.testing.assert_allclose(state.pi[attr].sum(), 1, atol=1e-12, rtol=0)
        assert state.means[attr].shape == (10, dimension)
        assert state.covariances[attr].shape == (10, dimension, dimension)
        np.testing.assert_allclose(
            state.covariances[attr],
            state.covariances[attr].transpose(0, 2, 1),
            atol=1e-12,
            rtol=1e-12,
        )
        assert (np.linalg.eigvalsh(state.covariances[attr]) > 0).all()
    return state


def validate_final_csv(directory, agent, state, dataset):
    csv_dir = directory / "csv"
    expected_z = csv_assignments(csv_dir, agent, dataset)
    for attr in ATTRIBUTES:
        assert np.array_equal(flatten(expected_z[attr]), flatten(state.z[attr]))
        for name, expected in (("pi", state.pi[attr]), ("Mu", state.means[attr])):
            actual = np.loadtxt(csv_dir / f"{name}_{attr}_{agent}.csv", delimiter=",")
            assert np.array_equal(actual, expected)
        nonempty = [row for row in read_csv(csv_dir / f"Sig_{attr}_{agent}.csv") if row]
        actual = np.asarray(nonempty, dtype=float).reshape(
            state.covariances[attr].shape
        )
        assert np.array_equal(actual, state.covariances[attr])
    assert np.array_equal(
        np.loadtxt(csv_dir / f"theta_{agent}.csv", delimiter=","), state.theta
    )
    assert read_csv(csv_dir / f"F_{agent}.csv") == [
        list(row) for row in state.attributes
    ]
    assert read_csv(csv_dir / f"W_list_{agent}.csv") == [list(state.vocabulary)]
    assert read_csv(csv_dir / f"uttered_words_{agent}.csv") == [
        [str(x) for x in row] for row in state.words
    ]
    assert np.array_equal(
        np.loadtxt(csv_dir / f"word_length_{agent}.csv", delimiter=","), np.full(30, 4)
    )


def validate_trial(directory, datasets):
    states = {
        agent: validate_checkpoint(directory, agent, dataset)
        for agent, dataset in datasets.items()
    }
    expected_files = set.union(*(csv_names(agent) for agent in ("A", "B")))
    assert {p.name for p in (directory / "csv").iterdir()} == expected_files
    for agent, state in states.items():
        validate_final_csv(directory, agent, state, datasets[agent])
    values, independent = independent_ari(
        {a: state.z for a, state in states.items()}, datasets
    )
    native = json.loads((directory / "analysis/metrics.json").read_text())
    assert all(math.isfinite(float(v)) for v in native.values())
    assert_native_ari(independent, native)
    metric_csv = metric_rows(directory / "analysis/metrics.csv")
    assert len(metric_csv) == 1 and set(metric_csv[0]) == set(native)
    assert all(float(metric_csv[0][key]) == value for key, value in native.items())
    history = metric_rows(directory / "analysis/history.csv")
    assert [int(row["iteration"]) for row in history] == list(range(1, ITERATIONS + 1))
    assert all(math.isfinite(float(v)) for row in history for v in row.values())
    assert all(float(history[-1][key]) == value for key, value in native.items())
    folders = {x.name for x in (directory / "iterations").iterdir() if x.is_dir()}
    assert folders == {f"iter{i:04}" for i in range(1, ITERATIONS + 1)}
    for i, row in enumerate(history, 1):
        snapshot = directory / "iterations" / f"iter{i:04}" / "csv"
        assert {p.name for p in snapshot.iterdir()} == expected_files
        assignments = {
            agent: csv_assignments(snapshot, agent, dataset)
            for agent, dataset in datasets.items()
        }
        intermediate = intermediate_native_ari(assignments, datasets)
        assert_native_ari(intermediate, row)
    last = directory / "iterations/iter0500/csv"
    for name in expected_files:
        assert (last / name).read_bytes() == (directory / "csv" / name).read_bytes()
    return values, {
        "native_metrics": native,
        "history_rows": 500,
        "final_checkpoint_sha256": {
            a: digest(directory / f"state_{a}.npz") for a in ("A", "B")
        },
        "final_metadata_sha256": {
            a: digest(directory / f"metadata_{a}.json") for a in ("A", "B")
        },
        "final_csv_sha256": {
            name: digest(directory / "csv" / name) for name in sorted(expected_files)
        },
        "native_metrics_sha256": digest(directory / "analysis/metrics.json"),
        "native_history_sha256": digest(directory / "analysis/history.csv"),
        "snapshots_verified": 500,
        "csv_files_per_snapshot": 42,
        "intermediate_check": "all z CSVs and all truth/agent ARIs; other intermediate CSVs checked for presence",
        "final_check": "all numeric checkpoint arrays finite; shapes, theta/pi normalization, SPD covariance; all final CSVs match checkpoint",
    }


def summarize(rows):
    result = []
    for scope in ("attended", "all"):
        for agent in ("A", "B", "AB_mean"):
            for attr in ATTRIBUTES:
                vals = [
                    r["ari"]
                    for r in rows
                    if (r["scope"], r["agent"], r["attribute"]) == (scope, agent, attr)
                ]
                assert len(vals) == 10
                result.append(
                    {
                        "scope": scope,
                        "agent": agent,
                        "attribute": attr,
                        "n": 10,
                        "mean": statistics.mean(vals),
                        "sample_sd": statistics.stdev(vals),
                        "values": vals,
                    }
                )
    return result


def write_report(result, destination):
    paper = result["paper_reference"]["ari"]

    def find(scope, agent, attr):
        return next(
            r
            for r in result["summary"]
            if (r["scope"], r["agent"], r["attribute"]) == (scope, agent, attr)
        )

    def fmt(row):
        return f"{row['mean']:.6f} ± {row['sample_sd']:.6f}"

    labels = {"a": "動作", "p": "位置", "o": "形状", "c": "色"}
    text = """# 本番INTER・原典ヒューリスティクス版の10試行

本番実装 `12ef90f` のInter-CSL-PGMを、500反復×10試行で新規実行した。以下は最終checkpointから独立再計算した正解ラベルとのARIで、平均±標本標準偏差（ddof=1）を示す。良い試行だけの選別はしていない。

## 注視対象のARIと論文値

動作は30シーン、位置・形状・色は各シーンの注視対象30個で評価する。A/Bは分けて報告し、差は「今回の平均−論文値」である。

| 属性 | 論文Table 3 | 今回A | A−論文 | 今回B | B−論文 |
|---|---:|---:|---:|---:|---:|
"""
    for attr in ATTRIBUTES:
        a, b = find("attended", "A", attr), find("attended", "B", attr)
        text += f"| {labels[attr]} | {paper[attr]:.3f} | {fmt(a)} | {a['mean'] - paper[attr]:+.6f} | {fmt(b)} | {b['mean'] - paper[attr]:+.6f} |\n"
    near_max = max(
        abs(find("attended", agent, attr)["mean"] - paper[attr])
        for agent, attr in (("A", "a"), ("B", "a"), ("A", "o"), ("B", "o"), ("A", "c"))
    )
    largest_agent, largest_attr = max(
        ((agent, attr) for agent in ("A", "B") for attr in ATTRIBUTES),
        key=lambda pair: abs(find("attended", *pair)["mean"] - paper[pair[1]]),
    )
    color_b = find("attended", "B", "c")
    text += (
        f"\n動作・形状・Aの色の平均差は、絶対値で最大約{near_max:.3f}だった。"
        f"位置の差はA={find('attended', 'A', 'p')['mean'] - paper['p']:+.6f}、"
        f"B={find('attended', 'B', 'p')['mean'] - paper['p']:+.6f}で、"
        f"Bの色の差は{color_b['mean'] - paper['c']:+.6f}だった。"
        f"論文値からの絶対差が8個の主体別・属性別平均で最大だったのは"
        f"{largest_agent}の{labels[largest_attr]}である。"
        f"この10試行では位置とBの色に差が残る。Bの色は試行間で"
        f"{min(color_b['values']):.3f}〜{max(color_b['values']):.3f}、"
        f"標本標準偏差{color_b['sample_sd']:.3f}であり、全10試行を集計している。\n"
    )
    text += """
出典：[原論文Table 3](https://arxiv.org/html/2306.15837v1#S3.T3)、[確認した論文値](paper_reference.json)。論文表にはこのARIの標準偏差がない。正解ラベルはユーザー確認により論文使用と同一だが、論文が用いた対象集合とA/B集約方法は未確定であり、差を同一評価条件での性能差と断定しない。表のTrials欄を独立シードの反復数として解釈していない。

## 全物体とA/B単純平均（補足）

全物体の評価は位置・形状・色の55物体を使う。動作は引き続き30シーン。A/B単純平均は各試行で両主体のARIを平均してから10試行で集計した補助値で、20個の独立標本として扱わず、論文の集約方法とも同一視しない。

| 対象 | 属性 | A | B | A/B単純平均 |
|---|---|---:|---:|---:|
"""
    for scope, label in (
        ("all", "全55物体（動作30）"),
        ("attended", "注視対象（動作30）"),
    ):
        for attr in ATTRIBUTES:
            text += f"| {label} | {labels[attr]} | {fmt(find(scope, 'A', attr))} | {fmt(find(scope, 'B', attr))} | {fmt(find(scope, 'AB_mean', attr))} |\n"
    text += """
## 条件・解釈

今回の対象はINTERのみ。seed=0から`SeedSequence(0).spawn(10)`で作る試行0〜9を用い、各試行の子系列をA/Bに分ける。この集合は過去探索で使用した0〜2を含む。以前の探索で比較したB2の確認試行10〜19とは別集合で、その結果を今回の結果の代用にはしていない。探索の影響を受けた構成の再実行であり、独立した盲検確認や統計的に有意な再現を主張するものではない。

語彙13、各属性10カテゴリ、混合事前はカテゴリ当たり0.1（原典のα/Kに対応）、γ=0.1。学習θは実Dirichlet100標本平均、学習z/Fには未補正θ、占有カテゴリμは条件付きNormal100標本平均、空カテゴリμは位置[-0.5,0.5)・他属性[0,1)の一様配置を使う。Gaussian事前はm₀=0、κ₀=0.001、V₀=0.01I、ν₀=次元+2。更新順はz→π→Gaussian→F→θで、語生成・MH受理・前処理・通信は現行のまま。

原典初期化への切替はしていない。初期θは現行の1標本、初期Σは0.01I、KMeansなども現行のままである。標本平均は単一標本Gibbsと異なる更新分布であり、原典ヒューリスティクスの利用を原プログラム全体や論文値の完全再現とは呼ばない。

## 保存状態の検証と再集計

最終checkpoint20個の全数値配列の有限性、形状、θ/πの正規化、Σの正定値を確認した。全10試行の500行のhistoryと500個のCSV snapshot（各42ファイル）を確認し、各snapshotのzから正解ARIとA/B間ARIを独立再計算してnative historyと照合した。中間snapshotの他のCSVは存在を検証し、最終42CSVはcheckpointとの値の一致、反復500と最終出力のbyte一致を確認した。

再計算は本番の`analysis.category_metrics`を呼ばず、`state.z`/CSVと正解ラベルから`sklearn.metrics.adjusted_rand_score`で行った。全500反復の中間Gaussian/θ配列の数値検証を行ったという意味ではない。

```bash
PYTHONPATH=src python docs-for-developper/experiments/inter_original_heuristics_12ef90f/aggregate_results.py
```

実行時と同じ依存環境で実行する。[固定プロトコル](protocol.md)、[設定](config.yaml)、[実行・ソース/データ出自manifest](execution_manifest.json)、[独立集計スクリプト](aggregate_results.py)、[集計JSON](results.json)、[試行別CSV](trial_metrics.csv)を参照。生checkpointとnative解析結果の場所はmanifestに記録する。
"""
    text += (
        f"\n本番コミットは `{result['git_head']}`。学習実行全体は"
        f"{result['execution_elapsed_seconds']:.3f}秒（約8分50秒）だった。"
        "[実行後のソース・入力検証](postrun_validation.json)はPASSで、"
        "本番ソース・設定・runnerの不変性、408入力ファイルと処理済み観測・"
        "正解・注意対象のfingerprint一致を確認した。\n"
    )
    destination.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEST / "config.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=DEST / "execution_manifest.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_root or config.paths.output_root / config.run.name
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert config.mode == "inter-csl" and config.run.seed == 0
    assert config.run.experiments == 10 and config.run.iterations == ITERATIONS
    assert (
        config.interaction == "mh" and config.attribute_model.exclusivity == "distinct"
    )
    assert not (config.freeze.gmm or config.freeze.theta or config.freeze.attributes)
    assert (
        config.hyperparameters.vocabulary_size == 13
        and config.hyperparameters.gamma == 0.1
    )
    assert config.hyperparameters.word_length == 4
    assert set(config.hyperparameters.category_counts.as_mapping().values()) == {10}
    assert set(
        config.hyperparameters.gmm.mixture_concentration.as_mapping().values()
    ) == {0.1}
    for field, expected in (
        ("mean_prior", 0),
        ("mean_precision", 0.001),
        ("covariance_scale", 0.01),
        ("covariance_df_offset", 2),
    ):
        assert set(
            getattr(config.hyperparameters.gmm, field).as_mapping().values()
        ) == {expected}
    assert config.hyperparameters.gmm.kmeans_n_init == 10
    assert (
        config.word_model.theta_sampler == "dirichlet"
        and config.word_model.rescaling == "legacy_theta_sum"
    )
    h = config.original_heuristics
    assert (
        h.theta_mean_samples,
        h.inference_theta,
        h.gaussian_mean_samples,
        h.empty_component_mean,
    ) == (100, "raw", 100, "original_uniform")
    assert load_config(output / "resolved_config.yaml") == config
    manifest = json.loads(args.manifest.read_text())
    manifest_root = args.manifest.resolve().parent
    assert manifest["head"] == EXPECTED_HEAD and manifest["status"] == "complete"
    assert manifest["config_sha256"] == digest(args.config)
    postrun = json.loads((manifest_root / "postrun_validation.json").read_text())
    assert postrun["status"] == "PASS" and postrun["head"] == EXPECTED_HEAD
    assert postrun["source_config_runner_sha256_unchanged"]
    assert postrun["input_processed_observation_truth_attention_fingerprints_unchanged"]
    assert postrun["execution_manifest_sha256"] == digest(args.manifest)
    assert manifest["runner_sha256"] == digest(manifest_root / "run_experiment.py")
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    runs = manifest["runs"]
    assert len(runs) == 10 and {r["trial"] for r in runs} == set(TRIALS)
    assert all(r["exit_code"] == 0 and r["status"] == "complete" for r in runs)
    for row in runs:
        assert row["seed_entropy"] == 0 and row["seed_spawn_key"] == [row["trial"]]
        assert (
            Path(row["output_directory"])
            == Path(manifest["output_root"]) / f"exp{row['trial']:03}"
        )
    datasets = {a: load_dataset(config.paths.data_root, a) for a in ("A", "B")}
    assert all(
        d.num_documents == 30
        and sum(d.num_objects) == 55
        and d.true_categories is not None
        for d in datasets.values()
    )
    data_hashes = {a: hash_dataset(d) for a, d in datasets.items()}
    for agent in ("A", "B"):
        expected = manifest["input_provenance"]["arrays"][agent]
        for attr in ATTRIBUTES:
            assert (
                data_hashes[agent]["observations"][attr]
                == expected["observations"][attr]["sha256"]
            )
            assert (
                data_hashes[agent]["truth"][attr] == expected["truth"][attr]["sha256"]
            )
        assert data_hashes[agent]["attentions"] == expected["attentions_sha256"]
    for name, expected in manifest["input_provenance"]["raw_files"].items():
        path = config.paths.data_root / name
        assert (
            path.stat().st_size == expected["size_bytes"]
            and digest(path) == expected["sha256"]
        )
    records, validations = [], []
    for trial in TRIALS:
        directory = output / f"exp{trial:03}"
        values, validation = validate_trial(directory, datasets)
        worker = json.loads(
            (manifest_root / "trial_status" / f"exp{trial:03}.json").read_text()
        )
        assert worker["status"] == "complete" and worker["exit_code"] == 0
        for agent in ("A", "B"):
            assert (
                worker["final_state_checks"][agent]["checkpoint_sha256"]
                == validation["final_checkpoint_sha256"][agent]
            )
            assert (
                worker["final_state_checks"][agent]["metadata_sha256"]
                == validation["final_metadata_sha256"][agent]
            )
        validations.append(dict(trial=trial, **validation))
        for scope in ("attended", "all"):
            for attr in ATTRIBUTES:
                for agent in ("A", "B"):
                    records.append(
                        {
                            "trial": trial,
                            "scope": scope,
                            "agent": agent,
                            "attribute": attr,
                            "ari": values[agent][attr][scope],
                        }
                    )
                records.append(
                    {
                        "trial": trial,
                        "scope": scope,
                        "agent": "AB_mean",
                        "attribute": attr,
                        "ari": statistics.mean(
                            values[a][attr][scope] for a in ("A", "B")
                        ),
                    }
                )
        print(
            f"Validated exp{trial:03}: final checkpoint and all 500 z/history snapshots",
            flush=True,
        )
    paper_reference = json.loads((DEST / "paper_reference.json").read_text())
    summary = summarize(records)
    for row in summary:
        row["difference_from_paper"] = (
            row["mean"] - paper_reference["ari"][row["attribute"]]
            if row["scope"] == "attended"
            else None
        )
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    assert runtime == manifest["runtime"]
    # A later documentation-only commit may change HEAD; source bytes must not.
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / name) == expected
    result = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "git_head": EXPECTED_HEAD,
        "aggregation_git_head": head,
        "config_sha256": digest(args.config),
        "execution_manifest_sha256": digest(args.manifest),
        "execution_elapsed_seconds": manifest["total_elapsed_seconds"],
        "postrun_validation_sha256": digest(manifest_root / "postrun_validation.json"),
        "aggregator_sha256": digest(__file__),
        "native_output_root": str(output),
        "recorded_native_output_root": manifest["output_root"],
        "runtime": runtime,
        "trials": list(TRIALS),
        "seed": 0,
        "iterations": 500,
        "sample_sd_ddof": 1,
        "data_hashes": data_hashes,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted((ROOT / "src/legacy_csl").glob("*.py"))
        },
        "paper_reference": paper_reference,
        "summary": summary,
        "trial_metrics": records,
        "validations": validations,
        "validation_summary": {
            "all_passed": True,
            "completed_trials": 10,
            "final_checkpoints": 20,
            "history_rows": 5000,
            "csv_snapshots": 5000,
            "csv_files_present": 210000,
            "independently_verified_history_ari_fields": 80000,
            "final_native_ari_fields": 160,
            "raw_input_files_verified": len(manifest["input_provenance"]["raw_files"]),
        },
        "limitation": "Research-informed configuration rerun; not blinded confirmation. Paper object subset and A/B aggregation unresolved; no best-seed selection.",
    }
    (DEST / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (DEST / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_report(result, DEST / "report.md")
    print("All ten native trials independently validated and aggregated", flush=True)


if __name__ == "__main__":
    main()
