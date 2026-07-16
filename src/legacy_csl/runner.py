"""High-level learning, prediction, validation, and analysis workflows.

学習、予測、検証、解析の高水準ワークフロー。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .analysis import category_metrics, prediction_metrics, write_analysis
from .config import Config, dump_resolved_config
from .data import ATTRIBUTES, ObservationDataset, load_dataset, load_single_words
from .model import CslModel
from .state_io import load_state, save_state, write_metric_rows


def learn(config: Config) -> list[Path]:
    """Run all configured learning experiments.

    設定された全学習実験を実行する。

    Args:
        config: Validated standalone configuration.
            検証済みスタンドアローン設定。

    Returns:
        Experiment output directories.
        実験出力ディレクトリ一覧。
    """
    validate_data(config)
    root = config.paths.output_root / config.run.name
    root.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, root / "resolved_config.yaml")
    experiment_dirs: list[Path] = []
    seed_sequences = np.random.SeedSequence(config.run.seed).spawn(
        config.run.experiments
    )
    for experiment_id, seed_sequence in enumerate(seed_sequences):
        directory = root / f"exp{experiment_id:03}"
        directory.mkdir(parents=True, exist_ok=True)
        if config.mode == "single-csl":
            _learn_single(config, directory, seed_sequence)
        else:
            _learn_inter(config, directory, seed_sequence)
        experiment_dirs.append(directory)
    return experiment_dirs


def predict(config: Config) -> list[dict[str, object]]:
    """Predict Inter-CSL listener observations from speaker utterances.

    話者の発話から Inter-CSL 聞き手の観測を予測する。

    Args:
        config: Validated Inter-CSL configuration.
            検証済み Inter-CSL 設定。

    Returns:
        One metric row per experiment.
        実験ごとの指標行。

    Raises:
        ValueError: If called for Single-CSL.
            Single-CSL に対して呼び出した場合。
    """
    if config.mode != "inter-csl":
        raise ValueError("prediction is available only for inter-csl")
    dataset_a = load_dataset(config.paths.data_root, "A")
    dataset_b = load_dataset(config.paths.data_root, "B")
    root = config.paths.output_root / config.run.name
    rows: list[dict[str, object]] = []
    seed_sequences = np.random.SeedSequence(config.run.seed + 1_000_003).spawn(
        config.run.experiments
    )
    for experiment_id, seed_sequence in enumerate(seed_sequences):
        directory = root / f"exp{experiment_id:03}"
        rng_a, rng_b = [np.random.default_rng(seed) for seed in seed_sequence.spawn(2)]
        model_a = CslModel.from_state(
            dataset_a, config, load_state(directory, "A"), rng=rng_a
        )
        model_b = CslModel.from_state(
            dataset_b, config, load_state(directory, "B"), rng=rng_b
        )
        utterances = model_a.draw_words()
        predicted_z, predicted_observations = model_b.predict_observations(utterances)
        metrics: dict[str, object] = {"experiment": experiment_id}
        metrics.update(prediction_metrics(predicted_observations, dataset_b))
        prediction_dir = directory / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_dir / "prediction.npz",
            utterances=utterances,
            **{f"z_{key}": value for key, value in predicted_z.items()},
            **{f"x_{key}": value for key, value in predicted_observations.items()},
        )
        (prediction_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        rows.append(metrics)
    write_metric_rows(root / "prediction_summary.csv", rows)
    return rows


def analyze(config: Config) -> list[dict[str, object]]:
    """Rebuild standard analysis products from saved checkpoints.

    保存済みチェックポイントから標準解析成果物を再構築する。

    Args:
        config: Validated configuration used for learning.
            学習で使った検証済み設定。

    Returns:
        Final metrics for each experiment.
        実験ごとの最終指標。
    """
    root = config.paths.output_root / config.run.name
    rows: list[dict[str, object]] = []
    if config.mode == "single-csl":
        dataset = load_dataset(config.paths.data_root, "U")
        for experiment_id in range(config.run.experiments):
            directory = root / f"exp{experiment_id:03}"
            state = load_state(directory, "U")
            history = _load_history(directory / "analysis" / "history.csv")
            metrics: dict[str, object] = {"experiment": experiment_id}
            metrics.update(
                write_analysis(directory, state, dataset, history=history)
            )
            rows.append(metrics)
    else:
        dataset = load_dataset(config.paths.data_root, "A")
        for experiment_id in range(config.run.experiments):
            directory = root / f"exp{experiment_id:03}"
            state_a = load_state(directory, "A")
            state_b = load_state(directory, "B")
            history = _load_history(directory / "analysis" / "history.csv")
            metrics = {"experiment": experiment_id}
            metrics.update(
                write_analysis(
                    directory,
                    state_a,
                    dataset,
                    other_state=state_b,
                    history=history,
                )
            )
            rows.append(metrics)
    write_metric_rows(root / "analysis_summary.csv", rows)
    return rows


def validate_data(config: Config) -> None:
    """Validate required input datasets without running inference.

    推論せずに必要な入力データセットを検証する。

    Args:
        config: Validated configuration.
            検証済み設定。

    Raises:
        ValueError: If document counts, dimensions, or words are inconsistent.
            文書数、次元、単語列が不整合の場合。
    """
    agents = ("U",) if config.mode == "single-csl" else ("A", "B")
    datasets = [load_dataset(config.paths.data_root, agent) for agent in agents]
    if len({dataset.num_documents for dataset in datasets}) != 1:
        raise ValueError("all Inter-CSL agents must have the same document count")
    for dataset in datasets:
        for attr in ATTRIBUTES:
            values = np.asarray(dataset.observations[attr])
            if values.ndim != 2 or not np.all(np.isfinite(values)):
                raise ValueError(f"observations[{attr!r}] must be a finite D x P matrix")
    if config.mode == "single-csl":
        words = load_single_words(config.paths.data_root, datasets[0].num_documents)
        expected = (
            datasets[0].num_documents,
            config.hyperparameters.word_length,
        )
        if words.shape != expected:
            raise ValueError(
                f"Single-CSL words must have shape {expected}, got {words.shape}"
            )
        if np.any(words < 0) or np.any(
            words >= config.hyperparameters.vocabulary_size
        ):
            raise ValueError("Single-CSL words exceed the configured vocabulary")


def _learn_single(
    config: Config, directory: Path, seed_sequence: np.random.SeedSequence
) -> None:
    dataset = load_dataset(config.paths.data_root, "U")
    words = load_single_words(config.paths.data_root, dataset.num_documents)
    model = CslModel(
        dataset,
        config,
        rng=np.random.default_rng(seed_sequence),
        initial_words=words,
    )
    history: list[dict[str, object]] = []
    for iteration in range(config.run.iterations):
        model.step(words)
        row: dict[str, object] = {"iteration": iteration + 1}
        row.update(category_metrics(model.state, dataset))
        history.append(row)
    save_state(model.state, directory, "U")
    write_analysis(directory, model.state, dataset, history=history)


def _learn_inter(
    config: Config, directory: Path, seed_sequence: np.random.SeedSequence
) -> None:
    dataset_a = load_dataset(config.paths.data_root, "A")
    dataset_b = load_dataset(config.paths.data_root, "B")
    rng_a, rng_b = [np.random.default_rng(seed) for seed in seed_sequence.spawn(2)]
    model_a = CslModel(dataset_a, config, rng=rng_a)
    model_b = CslModel(dataset_b, config, rng=rng_b)
    proposal_a = model_a.draw_words()
    proposal_b = model_b.draw_words()
    history: list[dict[str, object]] = []
    for iteration in range(config.run.iterations):
        if config.interaction == "no_communication":
            model_a.step(model_a.state.words)
            model_a.state.words = model_a.draw_words()
            model_b.step(model_b.state.words)
            model_b.state.words = model_b.draw_words()
        else:
            accepted_a = _accept_words(model_a, proposal_b, config.interaction)
            model_a.step(accepted_a)
            proposal_a = model_a.draw_words()
            accepted_b = _accept_words(model_b, proposal_a, config.interaction)
            model_b.step(accepted_b)
            proposal_b = model_b.draw_words()
        row: dict[str, object] = {"iteration": iteration + 1}
        row.update(category_metrics(model_a.state, dataset_a, other_state=model_b.state))
        history.append(row)
    save_state(model_a.state, directory, "A")
    save_state(model_b.state, directory, "B")
    write_analysis(
        directory,
        model_a.state,
        dataset_a,
        other_state=model_b.state,
        history=history,
    )


def _accept_words(
    listener: CslModel,
    suggested_words: npt.ArrayLike,
    interaction: str,
) -> npt.NDArray[np.int64]:
    suggested = listener._validated_words(suggested_words)
    if interaction == "all_accept":
        return suggested
    current = listener.state.words.copy()
    current_theta = listener.effective_theta(current)
    suggested_theta = listener.effective_theta(suggested)
    for doc_id, attributes in enumerate(listener.state.attributes):
        for word_id, attr in enumerate(attributes):
            category = listener.global_category(attr, listener.state.z[attr][doc_id])
            old_probability = current_theta[category, current[doc_id, word_id]]
            new_probability = suggested_theta[category, suggested[doc_id, word_id]]
            ratio = min(1.0, float(new_probability / old_probability))
            if listener.rng.random() <= ratio:
                current[doc_id, word_id] = suggested[doc_id, word_id]
    return current


def _load_history(path: Path) -> list[dict[str, object]] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return [
            {
                key: int(value) if key == "iteration" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]
