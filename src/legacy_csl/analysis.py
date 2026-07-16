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
class _CategoryWordHeatmap:
    """Store values with filtered, attribute-safe category and word axes.

    filter済みで属性区分を保持したカテゴリ・単語heatmapを保持する。
    """

    values: npt.NDArray[np.float64]
    category_labels: tuple[str, ...]
    word_labels: tuple[str, ...]
    category_boundaries: tuple[int, ...]
    word_boundaries: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _GroundTruthWordCounts:
    """Store raw ground-truth-category and accepted-word token counts.

    ground-truthカテゴリと受理済み単語tokenの生の共起数を保持する。
    """

    values: npt.NDArray[np.float64]
    category_keys: tuple[tuple[str, int], ...]
    vocabulary: tuple[str, ...]


def category_metrics(
    state: CslState,
    dataset: ObservationDataset,
    *,
    other_state: CslState | None = None,
    other_dataset: ObservationDataset | None = None,
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
        other_dataset: Dataset belonging to the second agent.
            第2エージェントに属するデータセット。

    Returns:
        Flat metric dictionary.
        フラットな指標辞書。
    """
    if (other_state is None) != (other_dataset is None):
        raise ValueError("other_state and other_dataset must be provided together")

    metrics: dict[str, float] = {}
    for attr in ATTRIBUTES:
        primary = _attended_state_categories(state, dataset, attr)
        truth = dataset.attended_categories(attr)
        if truth is not None:
            metrics[f"ari_truth_{attr}"] = float(
                adjusted_rand_score(truth, primary)
            )
        if other_state is not None:
            assert other_dataset is not None
            if len(primary) != other_dataset.num_documents:
                raise ValueError("agent datasets must have equal document counts")
            secondary = _attended_state_categories(
                other_state, other_dataset, attr
            )
            metrics[f"ari_agents_{attr}"] = float(
                adjusted_rand_score(primary, secondary)
            )
            metrics[f"kappa_agents_{attr}"] = float(
                cohen_kappa_score(primary, secondary)
            )
            if truth is not None:
                metrics[f"ari_truth_A_{attr}"] = metrics[f"ari_truth_{attr}"]
            other_truth = other_dataset.attended_categories(attr)
            if other_truth is not None:
                metrics[f"ari_truth_B_{attr}"] = float(
                    adjusted_rand_score(other_truth, secondary)
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
    other_dataset: ObservationDataset | None = None,
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
        other_dataset: Optional second agent dataset.
            任意の第2エージェントデータセット。
        history: Optional per-iteration metrics.
            任意の反復ごとの指標。

    Returns:
        Final metric dictionary.
        最終指標辞書。
    """
    analysis_dir = directory / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics = category_metrics(
        state,
        dataset,
        other_state=other_state,
        other_dataset=other_dataset,
    )
    (analysis_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_metric_rows(analysis_dir / "metrics.csv", [metrics])
    if history:
        write_metric_rows(analysis_dir / "history.csv", history)
        _plot_history(history, analysis_dir / "learning_curves.png")
    if other_state is None:
        _plot_theta(state, dataset, analysis_dir / "category_word_heatmap.png")
    else:
        assert other_dataset is not None
        reference_heatmap = _plot_theta(
            state,
            dataset,
            analysis_dir / "category_word_heatmap_A.png",
        )
        _plot_theta(
            other_state,
            other_dataset,
            analysis_dir / "category_word_heatmap_B.png",
            reference=reference_heatmap,
        )
        truth_heatmap_a, truth_heatmap_b = _prepare_ground_truth_heatmaps(
            state,
            dataset,
            other_state,
            other_dataset,
        )
        truth_heatmaps = tuple(
            heatmap
            for heatmap in (truth_heatmap_a, truth_heatmap_b)
            if heatmap is not None
        )
        if truth_heatmaps:
            value_max = max(
                1.0,
                max(float(np.max(heatmap.values)) for heatmap in truth_heatmaps),
            )
            if truth_heatmap_a is not None:
                _plot_category_word_heatmap(
                    truth_heatmap_a,
                    analysis_dir
                    / "ground_truth_category_word_heatmap_A.png",
                    colorbar_label="co-occurrence count",
                    value_max=value_max,
                    y_axis_label="attended ground-truth attribute-category",
                )
            if truth_heatmap_b is not None:
                _plot_category_word_heatmap(
                    truth_heatmap_b,
                    analysis_dir
                    / "ground_truth_category_word_heatmap_B.png",
                    colorbar_label="co-occurrence count",
                    value_max=value_max,
                    y_axis_label="attended ground-truth attribute-category",
                )
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
    state: CslState,
    dataset: ObservationDataset,
    path: Path,
    *,
    reference: _CategoryWordHeatmap | None = None,
) -> _CategoryWordHeatmap:
    """Plot used categories against vocabulary strings in diagonal-like order.

    使用カテゴリと語彙文字列を対角線状の順序で描画する。

    Args:
        state: Inferred state containing theta, assignments, and vocabulary.
            theta、割当て、語彙を含む推論状態。
        dataset: Dataset used to select attended categories.
            注視カテゴリの選択に使うデータセット。
        path: Destination image path.
            画像の出力先。
        reference: Optional heatmap whose word axis is reused exactly.
            単語軸を完全に共有する任意の参照heatmap。

    Returns:
        Prepared heatmap, which can be used as another agent's reference.
            別agentの参照に使える整列済みheatmap。
    """
    heatmap = _prepare_theta_heatmap(state, dataset, reference=reference)
    _plot_category_word_heatmap(heatmap, path, colorbar_label="theta")
    return heatmap


def _plot_category_word_heatmap(
    heatmap: _CategoryWordHeatmap,
    path: Path,
    *,
    colorbar_label: str,
    value_max: float | None = None,
    y_axis_label: str = "attribute-category",
) -> None:
    """Render a prepared category-word heatmap.

    整列済みのカテゴリ×単語heatmapを描画する。

    Args:
        heatmap: Prepared values and display axes.
            整列済みの値と表示軸。
        path: Destination image path.
            画像の出力先。
        colorbar_label: Label describing the displayed values.
            表示値を説明するcolorbar label。
        value_max: Optional shared upper color limit.
            任意の共通色上限。
        y_axis_label: Label describing category rows.
            カテゴリ行を説明するy軸label。
    """
    width = max(8.0, len(heatmap.word_labels) * 0.45)
    height = max(3.0, len(heatmap.category_labels) * 0.28)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        heatmap.values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0 if value_max is not None else None,
        vmax=value_max,
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
    axis.set_ylabel(y_axis_label)
    for boundary in heatmap.category_boundaries:
        axis.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for boundary in heatmap.word_boundaries:
        axis.axvline(boundary - 0.5, color="white", linewidth=1.5)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _prepare_theta_heatmap(
    state: CslState,
    dataset: ObservationDataset,
    *,
    reference: _CategoryWordHeatmap | None = None,
) -> _CategoryWordHeatmap:
    """Filter and order theta while preserving strict attribute blocks.

    厳密な属性blockを保持しながらthetaをfilter・整列する。

    A category is considered used when it attends at least one document, which
    matches the categories that can contribute to word likelihoods. Without a
    reference, columns are stably sorted by their strongest used category.
    With a reference, its word order and word boundaries are reused exactly.
    Rows are then sorted only within each attribute block by their peak-word
    position; rows never cross an ``a/p/o/c`` boundary. Category labels retain
    the zero-based local IDs used by legacy outputs, for example ``p4``.

    単語尤度に寄与し得るカテゴリに合わせ、1文書以上で注視されたカテゴリを
    使用済みとする。参照がなければ列は最大の使用カテゴリで安定sortし、参照が
    あればその単語順と単語境界を完全に再利用する。その後、行を各属性block内
    だけでpeak単語位置によりsortする。行が ``a/p/o/c`` 境界を跨ぐことは
    ない。カテゴリlabelはlegacy出力と同じ0始まりの局所ID（例: ``p4``）を保つ。

    Args:
        state: Inferred state containing theta, assignments, pi, and vocabulary.
            theta、割当て、pi、語彙を含む推論状態。
        dataset: Dataset containing attention indexes.
            注視indexを含むデータセット。
        reference: Optional prepared heatmap defining the shared word axis.
            共通単語軸を定める任意の整列済みheatmap。

    Returns:
        Filtered values, display labels, and attribute boundary positions.
            filter済み値、表示label、属性境界位置。

    Raises:
        ValueError: If dimensions are invalid or the reference vocabulary
            differs from the state's vocabulary.
            次元が不正な場合、または参照とstateの語彙が異なる場合。
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
    if reference is None:
        dominant_rows = np.argmax(values, axis=0)
        column_order = np.argsort(dominant_rows, kind="stable")
    else:
        vocabulary_indexes = {
            word: index for index, word in enumerate(state.vocabulary)
        }
        if set(reference.word_labels) != set(vocabulary_indexes) or len(
            reference.word_labels
        ) != len(vocabulary_indexes):
            raise ValueError("reference heatmap must contain the same vocabulary")
        column_order = np.asarray(
            [vocabulary_indexes[word] for word in reference.word_labels],
            dtype=np.int64,
        )
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
    if reference is None:
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
    else:
        word_boundaries = reference.word_boundaries
    return _CategoryWordHeatmap(
        values=values,
        category_labels=tuple(ordered_labels),
        word_labels=tuple(word_labels),
        category_boundaries=category_boundaries,
        word_boundaries=word_boundaries,
    )


def _prepare_ground_truth_heatmaps(
    state: CslState,
    dataset: ObservationDataset,
    other_state: CslState,
    other_dataset: ObservationDataset,
) -> tuple[_CategoryWordHeatmap | None, _CategoryWordHeatmap | None]:
    """Prepare Inter heatmaps of attended truth categories versus words.

    注視対象の正解カテゴリと単語のInter heatmapを整列する。

    Every accepted word token is paired only with the attended truth category
    selected by that token's inferred attribute ``F``. The two agents use the
    union of their truth-category rows and observed words. Their shared row and
    column orders are determined from the summed co-occurrence counts, which is
    valid because ground-truth IDs have the same external meaning and avoids
    privileging either agent.

    各受理済み単語tokenは、そのtokenの推論属性 ``F`` が指す注視対象の
    正解カテゴリとのみ対応付ける。両agentの正解カテゴリ行と観測単語の
    和集合を使い、共起数の和から共通の行列順を決める。

    Returns:
        Prepared A and B heatmaps. An entry is ``None`` when that agent has no
        ground truth.
            整列済みA・B heatmap。該当agentに正解がなければ ``None``。
    """
    counts_a = _ground_truth_word_counts(state, dataset)
    counts_b = _ground_truth_word_counts(other_state, other_dataset)
    available = tuple(counts for counts in (counts_a, counts_b) if counts is not None)
    if not available:
        return None, None

    reference_vocabulary = available[0].vocabulary
    for counts in available[1:]:
        if set(counts.vocabulary) != set(reference_vocabulary):
            raise ValueError("Inter ground-truth heatmaps require matching vocabularies")

    category_keys = tuple(
        (attr, category)
        for attr in ATTRIBUTES
        for category in sorted(
            {
                key_category
                for counts in available
                for key_attr, key_category in counts.category_keys
                if key_attr == attr
            }
        )
    )
    used_words = {
        word
        for counts in available
        for word, total in zip(counts.vocabulary, np.sum(counts.values, axis=0))
        if total > 0
    }
    word_labels = tuple(word for word in reference_vocabulary if word in used_words)
    if not category_keys or not word_labels:
        return None, None

    values_a = _align_ground_truth_counts(counts_a, category_keys, word_labels)
    values_b = _align_ground_truth_counts(counts_b, category_keys, word_labels)
    aggregate = sum(
        values for values in (values_a, values_b) if values is not None
    )

    dominant_rows = np.argmax(aggregate, axis=0)
    column_order = np.argsort(dominant_rows, kind="stable")
    word_labels = tuple(word_labels[int(index)] for index in column_order)
    aggregate = aggregate[:, column_order]
    if values_a is not None:
        values_a = values_a[:, column_order]
    if values_b is not None:
        values_b = values_b[:, column_order]

    block_sizes = [
        sum(key_attr == attr for key_attr, _ in category_keys) for attr in ATTRIBUTES
    ]
    row_order: list[int] = []
    start = 0
    for block_size in block_sizes:
        stop = start + block_size
        block = aggregate[start:stop]
        block_order = sorted(
            range(block_size),
            key=lambda index: (
                float(np.sum(block[index])) == 0.0,
                int(np.argmax(block[index])),
            ),
        )
        row_order.extend(start + index for index in block_order)
        start = stop
    row_indexes = np.asarray(row_order, dtype=np.int64)
    category_keys = tuple(category_keys[index] for index in row_order)
    aggregate = aggregate[row_indexes]
    if values_a is not None:
        values_a = values_a[row_indexes]
    if values_b is not None:
        values_b = values_b[row_indexes]

    nonempty_block_sizes = [size for size in block_sizes if size]
    category_boundaries = tuple(
        int(boundary) for boundary in np.cumsum(nonempty_block_sizes)[:-1]
    )
    dominant_rows = np.argmax(aggregate, axis=0)
    dominant_attributes = np.searchsorted(
        np.asarray(category_boundaries, dtype=np.int64),
        dominant_rows,
        side="right",
    )
    word_boundaries = tuple(
        int(index + 1)
        for index in np.flatnonzero(np.diff(dominant_attributes))
    )
    category_labels = tuple(f"{attr}{category}" for attr, category in category_keys)

    def prepare(values: npt.NDArray[np.float64] | None) -> _CategoryWordHeatmap | None:
        if values is None:
            return None
        return _CategoryWordHeatmap(
            values=values,
            category_labels=category_labels,
            word_labels=word_labels,
            category_boundaries=category_boundaries,
            word_boundaries=word_boundaries,
        )

    return prepare(values_a), prepare(values_b)


def _ground_truth_word_counts(
    state: CslState,
    dataset: ObservationDataset,
) -> _GroundTruthWordCounts | None:
    """Count accepted words against attended truth categories using inferred F.

    推論済みFで属性を選び、受理済み単語と注視正解カテゴリを数える。
    """
    truth = {attr: dataset.attended_categories(attr) for attr in ATTRIBUTES}
    if any(values is None for values in truth.values()):
        return None
    attended_truth = {
        attr: np.asarray(values, dtype=np.int64)
        for attr, values in truth.items()
        if values is not None
    }
    category_keys = tuple(
        (attr, int(category))
        for attr in ATTRIBUTES
        for category in np.unique(attended_truth[attr])
    )
    row_indexes = {key: index for index, key in enumerate(category_keys)}
    values = np.zeros((len(category_keys), len(state.vocabulary)), dtype=np.float64)
    if len(state.words) != dataset.num_documents or len(state.attributes) != len(
        state.words
    ):
        raise ValueError("state words and attributes must match dataset documents")
    for document, (words, attributes) in enumerate(
        zip(state.words, state.attributes)
    ):
        if len(words) != len(attributes):
            raise ValueError("each document must have one inferred F per word")
        for word, attr in zip(words, attributes):
            if attr not in attended_truth:
                raise ValueError(f"unknown inferred attribute {attr!r}")
            word_index = int(word)
            if word_index < 0 or word_index >= len(state.vocabulary):
                raise ValueError("word index is outside the state vocabulary")
            category = int(attended_truth[attr][document])
            values[row_indexes[(attr, category)], word_index] += 1.0
    return _GroundTruthWordCounts(values, category_keys, state.vocabulary)


def _align_ground_truth_counts(
    counts: _GroundTruthWordCounts | None,
    category_keys: tuple[tuple[str, int], ...],
    word_labels: tuple[str, ...],
) -> npt.NDArray[np.float64] | None:
    """Align one agent's counts to shared semantic row and word axes."""
    if counts is None:
        return None
    category_indexes = {key: index for index, key in enumerate(category_keys)}
    word_indexes = {word: index for index, word in enumerate(word_labels)}
    aligned = np.zeros((len(category_keys), len(word_labels)), dtype=np.float64)
    for source_row, key in enumerate(counts.category_keys):
        target_row = category_indexes[key]
        for source_column, word in enumerate(counts.vocabulary):
            target_column = word_indexes.get(word)
            if target_column is not None:
                aligned[target_row, target_column] = counts.values[
                    source_row, source_column
                ]
    return aligned


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
