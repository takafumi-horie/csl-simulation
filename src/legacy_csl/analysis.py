"""Standard metrics and compact plots for standalone CSL runs.

スタンドアローン CSL 実行の標準指標と基本図。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

from .data import ATTRIBUTES, ObservationDataset
from .model import CslState
from .state_io import write_metric_rows


def category_metrics(
    state: CslState,
    dataset: ObservationDataset,
    *,
    other_state: CslState | None = None,
) -> dict[str, float]:
    """Calculate ARI against truth or another agent and optional kappa.

    正解または別エージェントとの ARI と任意の kappa を計算する。

    Args:
        state: Primary agent state.
            主エージェント状態。
        dataset: Primary dataset and optional truth.
            主データセットと任意の正解。
        other_state: Optional second agent state.
            任意の第2エージェント状態。

    Returns:
        Flat metric dictionary.
        フラットな指標辞書。
    """
    metrics: dict[str, float] = {}
    for attr in ATTRIBUTES:
        primary = _attended_state_categories(state, dataset, attr)
        truth = dataset.attended_categories(attr)
        if truth is not None:
            metrics[f"ari_truth_{attr}"] = float(
                adjusted_rand_score(truth, primary)
            )
        if other_state is not None:
            secondary = _attended_state_categories(other_state, dataset, attr)
            metrics[f"ari_agents_{attr}"] = float(
                adjusted_rand_score(primary, secondary)
            )
            metrics[f"kappa_agents_{attr}"] = float(
                cohen_kappa_score(primary, secondary)
            )
    if other_state is not None:
        comparisons = [
            left == right
            for left_doc, right_doc in zip(state.words, other_state.words)
            for left, right in zip(left_doc, right_doc)
        ]
        metrics["word_exact_agreement"] = float(
            np.mean(comparisons)
        )
        metrics["utterance_exact_agreement"] = float(
            np.mean(
                [
                    np.array_equal(left, right)
                    for left, right in zip(state.words, other_state.words)
                ]
            )
        )
    return metrics


def prediction_metrics(
    predicted: Mapping[str, npt.NDArray[np.float64]],
    dataset: ObservationDataset,
) -> dict[str, float]:
    """Calculate modality-wise mean squared prediction errors.

    モダリティごとの予測平均二乗誤差を計算する。

    Args:
        predicted: Predicted observation matrices by attribute.
            属性別の予測観測行列。
        dataset: Target observations.
            対象観測。

    Returns:
        Attribute-wise MSE dictionary.
        属性別 MSE 辞書。
    """
    return {
        f"mse_{attr}": float(
            np.mean(np.square(predicted[attr] - dataset.attended(attr)))
        )
        for attr in ATTRIBUTES
    }


def write_analysis(
    directory: Path,
    state: CslState,
    dataset: ObservationDataset,
    *,
    other_state: CslState | None = None,
    history: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Write metrics, learning curves, and category-word heatmaps.

    指標、学習曲線、カテゴリ×単語ヒートマップを保存する。

    Args:
        directory: Experiment output directory.
            実験出力ディレクトリ。
        state: Primary agent state.
            主エージェント状態。
        dataset: Primary dataset.
            主データセット。
        other_state: Optional second agent state.
            任意の第2エージェント状態。
        history: Optional per-iteration metrics.
            任意の反復ごとの指標。

    Returns:
        Final metric dictionary.
        最終指標辞書。
    """
    analysis_dir = directory / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics = category_metrics(state, dataset, other_state=other_state)
    (analysis_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_metric_rows(analysis_dir / "metrics.csv", [metrics])
    if history:
        write_metric_rows(analysis_dir / "history.csv", history)
        _plot_history(history, analysis_dir / "learning_curves.png")
    _plot_theta(state.theta, analysis_dir / "category_word_heatmap.png")
    return metrics


def _plot_history(history: list[dict[str, object]], path: Path) -> None:
    numeric_keys = [
        key
        for key, value in history[0].items()
        if key != "iteration" and isinstance(value, (float, int, np.number))
    ]
    if not numeric_keys:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    iterations = [int(row["iteration"]) for row in history]
    for key in numeric_keys:
        axis.plot(iterations, [float(row[key]) for row in history], label=key)
    axis.set_xlabel("iteration")
    axis.set_ylabel("metric")
    axis.legend(fontsize="x-small", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_theta(theta: npt.NDArray[np.float64], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, max(3, theta.shape[0] * 0.15)))
    image = axis.imshow(theta, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("word")
    axis.set_ylabel("global category z")
    figure.colorbar(image, ax=axis, label="theta")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _attended_state_categories(
    state: CslState, dataset: ObservationDataset, attribute: str
) -> npt.NDArray[np.int64]:
    assignments = state.z[attribute]
    if attribute == "a":
        assert isinstance(assignments, np.ndarray)
        return assignments
    assert isinstance(assignments, tuple)
    return np.asarray(
        [values[int(attention)] for values, attention in zip(assignments, dataset.attentions)],
        dtype=np.int64,
    )
