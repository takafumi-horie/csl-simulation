"""Standard metrics and compact plots for standalone CSL runs.

スタンドアローン CSL 実行の標準指標と基本図。
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _ThetaHeatmap:
    """Store filtered and attribute-safe heatmap axes.

    filter済みで属性区分を保持したheatmap軸を保持する。
    """

    values: npt.NDArray[np.float64]
    category_labels: tuple[str, ...]
    word_labels: tuple[str, ...]
    category_boundaries: tuple[int, ...]
    word_boundaries: tuple[int, ...]


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
    _plot_theta(state, dataset, analysis_dir / "category_word_heatmap.png")
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


def _plot_theta(
    state: CslState, dataset: ObservationDataset, path: Path
) -> None:
    """Plot used categories against vocabulary strings in diagonal-like order.

    使用カテゴリと語彙文字列を対角線状の順序で描画する。

    Args:
        state: Inferred state containing theta, assignments, and vocabulary.
            theta、割当て、語彙を含む推論状態。
        dataset: Dataset used to select attended categories.
            注視カテゴリの選択に使うデータセット。
        path: Destination image path.
            画像の出力先。
    """
    heatmap = _prepare_theta_heatmap(state, dataset)
    width = max(8.0, len(heatmap.word_labels) * 0.45)
    height = max(3.0, len(heatmap.category_labels) * 0.28)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        heatmap.values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axis.set_xticks(
        np.arange(len(heatmap.word_labels)),
        labels=heatmap.word_labels,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(heatmap.category_labels)),
        labels=heatmap.category_labels,
    )
    axis.set_xlabel("word")
    axis.set_ylabel("attribute-category")
    for boundary in heatmap.category_boundaries:
        axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for boundary in heatmap.word_boundaries:
        axis.axvline(boundary - 0.5, color="white", linewidth=1.5)
    figure.colorbar(image, ax=axis, label="theta")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _prepare_theta_heatmap(
    state: CslState, dataset: ObservationDataset
) -> _ThetaHeatmap:
    """Filter and order theta while preserving strict attribute blocks.

    厳密な属性blockを保持しながらthetaをfilter・整列する。

    A category is considered used when it attends at least one document, which
    matches the categories that can contribute to word likelihoods. Columns
    are stably sorted by their strongest used category. Rows are then sorted
    only within each attribute block by their peak-word position; rows never
    cross an ``a/p/o/c`` boundary. Category labels retain the zero-based local
    IDs used by legacy outputs, for example ``p4``.

    単語尤度に寄与し得るカテゴリに合わせ、1文書以上で注視されたカテゴリを
    使用済みとする。列は最大の使用カテゴリで安定sortし、その後、行を各属性
    block内だけでpeak単語位置によりsortする。行が ``a/p/o/c`` 境界を跨ぐことは
    ない。カテゴリlabelはlegacy出力と同じ0始まりの局所ID（例: ``p4``）を保つ。

    Args:
        state: Inferred state containing theta, assignments, pi, and vocabulary.
            theta、割当て、pi、語彙を含む推論状態。
        dataset: Dataset containing attention indexes.
            注視indexを含むデータセット。

    Returns:
        Filtered values, display labels, and attribute boundary positions.
            filter済み値、表示label、属性境界位置。

    Raises:
        ValueError: If theta dimensions do not match pi or vocabulary.
            theta次元がpiまたは語彙と一致しない場合。
    """
    category_counts = {attr: len(state.pi[attr]) for attr in ATTRIBUTES}
    if state.theta.shape != (
        sum(category_counts.values()),
        len(state.vocabulary),
    ):
        raise ValueError("theta shape must match pi category counts and vocabulary")

    row_indexes: list[int] = []
    row_labels: list[str] = []
    block_sizes: list[int] = []
    offset = 0
    for attr in ATTRIBUTES:
        used_categories = np.unique(
            _attended_state_categories(state, dataset, attr)
        )
        if np.any(used_categories < 0) or np.any(
            used_categories >= category_counts[attr]
        ):
            raise ValueError(f"z[{attr!r}] contains a category outside pi")
        local_categories = [int(category) for category in used_categories]
        row_indexes.extend(offset + category for category in local_categories)
        row_labels.extend(f"{attr}{category}" for category in local_categories)
        block_sizes.append(len(local_categories))
        offset += category_counts[attr]

    values = state.theta[np.asarray(row_indexes, dtype=np.int64)].copy()
    dominant_rows = np.argmax(values, axis=0)
    column_order = np.argsort(dominant_rows, kind="stable")
    values = values[:, column_order]
    word_labels = [state.vocabulary[index] for index in column_order]

    ordered_values: list[npt.NDArray[np.float64]] = []
    ordered_labels: list[str] = []
    start = 0
    for block_size in block_sizes:
        stop = start + block_size
        block = values[start:stop]
        row_order = np.argsort(np.argmax(block, axis=1), kind="stable")
        ordered_values.append(block[row_order])
        ordered_labels.extend(row_labels[start + int(index)] for index in row_order)
        start = stop
    values = np.concatenate(ordered_values, axis=0)

    category_boundaries = tuple(
        int(boundary) for boundary in np.cumsum(block_sizes)[:-1]
    )
    dominant_rows = np.argmax(values, axis=0)
    dominant_attributes = np.searchsorted(
        np.asarray(category_boundaries, dtype=np.int64),
        dominant_rows,
        side="right",
    )
    word_boundaries = tuple(
        int(index + 1)
        for index in np.flatnonzero(np.diff(dominant_attributes))
    )
    return _ThetaHeatmap(
        values=values,
        category_labels=tuple(ordered_labels),
        word_labels=tuple(word_labels),
        category_boundaries=category_boundaries,
        word_boundaries=word_boundaries,
    )


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
