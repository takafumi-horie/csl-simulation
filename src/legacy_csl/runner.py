"""High-level mathematically faithful CSL workflows.

数学的に旧実装と等価なCSL高水準workflow。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .analysis import category_metrics, prediction_metrics, write_analysis
from .config import Config, dump_resolved_config
from .data import ATTRIBUTES, load_dataset, load_single_words
from .model import CslModel, IntArray
from .state_io import load_state, save_state, write_metric_rows


def learn(config: Config) -> list[Path]:
    """Run all configured learning experiments.

    設定された全学習実験を実行する。

    Args:
        config: Validated standalone configuration.
            検証済みstandalone設定。

    Returns:
        Experiment output directories.
        実験出力ディレクトリ一覧。
    """
    validate_data(config)
    root = config.paths.output_root / config.run.name
    root.mkdir(parents=True, exist_ok=True)
    dump_resolved_config(config, root / "resolved_config.yaml")
    directories: list[Path] = []
    for experiment_id, seed in enumerate(
        np.random.SeedSequence(config.run.seed).spawn(config.run.experiments)
    ):
        directory = root / f"exp{experiment_id:03}"
        directory.mkdir(parents=True, exist_ok=True)
        if config.mode == "single-csl":
            _learn_single(config, directory, seed)
        else:
            _learn_inter(config, directory, seed)
        directories.append(directory)
    return directories


def predict(config: Config) -> list[dict[str, object]]:
    """Run the legacy Inter-CSL generative prediction procedure.

    legacy Inter-CSLの生成的予測手続きを実行する。

    Loaded training ``z`` and ``F`` are deliberately not reused. Prediction
    fixes canonical ``F=(a,p,o,c)``, samples speaker categories from ``pi``,
    samples words, samples listener categories from ``pi*theta``, and finally
    samples observations from the listener's full Gaussians.

    学習時の ``z`` と ``F`` は再利用しない。canonical ``F=(a,p,o,c)`` を固定し、
    話者カテゴリを ``pi``、単語、聞き手カテゴリを ``pi*theta``、観測を聞き手の
    full Gaussianから順にサンプルする。

    Args:
        config: Validated Inter-CSL configuration.
            検証済みInter-CSL設定。

    Returns:
        One metric row per experiment.
        実験ごとの指標行。
    """
    if config.mode != "inter-csl":
        raise ValueError("prediction is available only for inter-csl")
    if config.hyperparameters.word_length != len(ATTRIBUTES):
        raise ValueError("legacy Inter-CSL prediction requires word_length=4")
    dataset_a = load_dataset(config.paths.data_root, "A")
    dataset_b = load_dataset(config.paths.data_root, "B")
    root = config.paths.output_root / config.run.name
    rows: list[dict[str, object]] = []
    seeds = np.random.SeedSequence(config.run.seed + 1_000_003).spawn(
        config.run.experiments
    )
    for experiment_id, seed in enumerate(seeds):
        directory = root / f"exp{experiment_id:03}"
        rng_a, rng_b = [np.random.default_rng(child) for child in seed.spawn(2)]
        speaker = CslModel.from_state(
            dataset_a, config, load_state(directory, "A"), rng=rng_a
        )
        listener = CslModel.from_state(
            dataset_b, config, load_state(directory, "B"), rng=rng_b
        )
        canonical = [tuple(ATTRIBUTES) for _ in range(dataset_a.num_documents)]
        speaker.state.attributes = canonical
        speaker.state.words = tuple(
            np.zeros(len(ATTRIBUTES), dtype=np.int64)
            for _ in range(dataset_a.num_documents)
        )
        speaker.state.z = speaker._sample_z_prior(speaker.state.pi)
        utterances = speaker.draw_words()
        listener.state.attributes = list(canonical)
        predicted_z = listener.sample_categories_without_observations(utterances)
        predicted_observations = listener.sample_observations(predicted_z)
        metrics: dict[str, object] = {"experiment": experiment_id}
        metrics.update(prediction_metrics(predicted_observations, dataset_b))
        prediction_dir = directory / "prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        utterance_flat, utterance_offsets = _pack_words(utterances)
        np.savez_compressed(
            prediction_dir / "prediction.npz",
            utterances_flat=utterance_flat,
            utterance_offsets=utterance_offsets,
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

    保存済みcheckpointから標準解析成果物を再構築する。

    Args:
        config: Configuration used for learning.
            学習で使った設定。

    Returns:
        Final metrics for every experiment.
        全実験の最終指標。
    """
    root = config.paths.output_root / config.run.name
    rows: list[dict[str, object]] = []
    dataset = load_dataset(
        config.paths.data_root, "U" if config.mode == "single-csl" else "A"
    )
    for experiment_id in range(config.run.experiments):
        directory = root / f"exp{experiment_id:03}"
        state = load_state(directory, "U" if config.mode == "single-csl" else "A")
        other = load_state(directory, "B") if config.mode == "inter-csl" else None
        metrics: dict[str, object] = {"experiment": experiment_id}
        metrics.update(
            write_analysis(
                directory,
                state,
                dataset,
                other_state=other,
                history=_load_history(directory / "analysis" / "history.csv"),
            )
        )
        rows.append(metrics)
    write_metric_rows(root / "analysis_summary.csv", rows)
    return rows


def validate_data(config: Config) -> None:
    """Validate observations, truth needed by freeze, and Single words.

    観測、freezeに必要な正解、Single単語を検証する。

    Args:
        config: Validated configuration.
            検証済み設定。
    """
    agents = ("U",) if config.mode == "single-csl" else ("A", "B")
    datasets = [load_dataset(config.paths.data_root, agent) for agent in agents]
    if len({dataset.num_documents for dataset in datasets}) != 1:
        raise ValueError("all Inter-CSL agents must have the same document count")
    for dataset in datasets:
        for attr in ATTRIBUTES:
            values = dataset.flatten(attr)
            if values.ndim != 2 or not np.all(np.isfinite(values)):
                raise ValueError(f"observations[{attr!r}] must be a finite N x P matrix")
            if len(values) < config.hyperparameters.category_counts.as_mapping()[attr]:
                raise ValueError(f"observations[{attr!r}] are fewer than categories")
        if config.freeze.gmm and dataset.true_categories is None:
            raise ValueError("freeze.gmm requires full ground-truth categories")
    if config.mode == "single-csl":
        corpus = load_single_words(config.paths.data_root, datasets[0].num_documents)
        if (
            config.attribute_model.exclusivity == "distinct"
            and np.any(corpus.lengths > len(ATTRIBUTES))
        ):
            raise ValueError("distinct attributes require at most four words")


def accept_words(
    listener: CslModel,
    suggested_words: Sequence[npt.ArrayLike],
    interaction: str,
) -> tuple[IntArray, ...]:
    """Apply the token-wise legacy Inter-CSL MH acceptance kernel.

    token単位のlegacy Inter-CSL MH受理kernelを適用する。

    Args:
        listener: Listening agent.
            聞き手エージェント。
        suggested_words: Speaker proposal corpus.
            話者の提案コーパス。
        interaction: ``mh`` or ``all_accept``.
            ``mh`` または ``all_accept``。

    Returns:
        Accepted internal word corpus.
        受理後の内部単語コーパス。
    """
    suggested = listener._coerce_words(suggested_words)
    assert suggested is not None
    listener._validate_word_shapes(suggested)
    if interaction == "all_accept":
        return tuple(row.copy() for row in suggested)
    current = tuple(row.copy() for row in listener.state.words)
    theta = listener.word_theta()
    empirical = listener.config.word_model.rescaling == "empirical_unigram"
    current_counts = Counter(np.concatenate(current).tolist()) if empirical else None
    suggested_counts = Counter(np.concatenate(suggested).tolist()) if empirical else None
    for doc_id, attributes in enumerate(listener.state.attributes):
        for word_id, attr in enumerate(attributes):
            category = listener.global_category(
                attr, listener.attended_category(attr, doc_id)
            )
            old_word = int(current[doc_id][word_id])
            new_word = int(suggested[doc_id][word_id])
            old_probability = theta[category, old_word]
            new_probability = theta[category, new_word]
            if empirical:
                assert current_counts is not None and suggested_counts is not None
                old_probability /= current_counts[old_word]
                new_probability /= suggested_counts[new_word]
            ratio = min(1.0, float(new_probability / old_probability))
            if listener.rng.random() <= ratio:
                current[doc_id][word_id] = new_word
    return current


def _learn_single(
    config: Config, directory: Path, seed: np.random.SeedSequence
) -> None:
    dataset = load_dataset(config.paths.data_root, "U")
    corpus = load_single_words(config.paths.data_root, dataset.num_documents)
    model = CslModel(
        dataset,
        config,
        rng=np.random.default_rng(seed),
        initial_words=corpus.encoded,
        vocabulary=corpus.vocabulary,
    )
    history: list[dict[str, object]] = []
    for iteration in range(config.run.iterations):
        model.step(corpus.encoded)
        row: dict[str, object] = {"iteration": iteration + 1}
        row.update(category_metrics(model.state, dataset))
        history.append(row)
    save_state(model.state, directory, "U")
    write_analysis(directory, model.state, dataset, history=history)


def _learn_inter(
    config: Config, directory: Path, seed: np.random.SeedSequence
) -> None:
    dataset_a = load_dataset(config.paths.data_root, "A")
    dataset_b = load_dataset(config.paths.data_root, "B")
    rng_a, rng_b = [np.random.default_rng(child) for child in seed.spawn(2)]
    model_a = CslModel(dataset_a, config, rng=rng_a)
    model_b = CslModel(dataset_b, config, rng=rng_b)
    # Legacy generate() uses the same first draw as internal and proposal words.
    proposal_a = tuple(row.copy() for row in model_a.state.words)
    proposal_b = tuple(row.copy() for row in model_b.state.words)
    history: list[dict[str, object]] = []
    for iteration in range(config.run.iterations):
        source_a = proposal_a if config.interaction == "no_communication" else proposal_b
        accepted_a = accept_words(model_a, source_a, config.interaction)
        model_a.step(accepted_a)
        proposal_a = model_a.draw_words()
        source_b = proposal_b if config.interaction == "no_communication" else proposal_a
        accepted_b = accept_words(model_b, source_b, config.interaction)
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


def _pack_words(words: tuple[IntArray, ...]) -> tuple[IntArray, IntArray]:
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum([len(row) for row in words])]
    )
    return np.concatenate(words), offsets
