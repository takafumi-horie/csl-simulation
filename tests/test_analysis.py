"""Tests for analysis labels and attribute-safe heatmap ordering.

解析labelと属性を跨がないheatmap整列のテスト。
"""

from __future__ import annotations

import numpy as np

from legacy_csl.analysis import (
    _ground_truth_word_counts,
    _plot_theta,
    _prepare_ground_truth_heatmaps,
    _prepare_theta_heatmap,
    category_metrics,
)
from legacy_csl.data import ObservationDataset
from legacy_csl.model import CslState


def _state_and_dataset() -> tuple[CslState, ObservationDataset]:
    observations = {
        "a": np.ones((2, 1)),
        "p": (np.ones((2, 1)), np.ones((1, 1))),
        "o": (np.ones((2, 1)), np.ones((1, 1))),
        "c": (np.ones((2, 1)), np.ones((1, 1))),
    }
    dataset = ObservationDataset(observations, np.array([0, 0]))
    z = {
        "a": np.array([0, 0]),
        # p1 occurs only on a distractor and must not appear in the heatmap.
        "p": (np.array([4, 1]), np.array([4])),
        "o": (np.array([0, 1]), np.array([0])),
        "c": (np.array([1, 0]), np.array([1])),
    }
    counts = {"a": 2, "p": 5, "o": 2, "c": 2}
    theta = np.full((sum(counts.values()), 4), 0.25)
    offsets = {"a": 0, "p": 2, "o": 7, "c": 9}
    peaks = {
        ("a", 0): 2,
        ("p", 4): 1,
        ("o", 0): 3,
        ("c", 1): 0,
    }
    for (attr, category), word in peaks.items():
        theta[offsets[attr] + category] = 0.05
        theta[offsets[attr] + category, word] = 0.85
    pi = {attr: np.full(count, 1.0 / count) for attr, count in counts.items()}
    means = {attr: np.zeros((count, 1)) for attr, count in counts.items()}
    covariances = {
        attr: np.ones((count, 1, 1)) for attr, count in counts.items()
    }
    state = CslState(
        z=z,
        pi=pi,
        means=means,
        covariances=covariances,
        theta=theta,
        attributes=[("a", "p", "o", "c")] * 2,
        words=(np.arange(4), np.arange(4)),
        vocabulary=("red", "left", "push", "round"),
    )
    return state, dataset


def test_theta_heatmap_filters_labels_and_orders_attribute_blocks() -> None:
    state, dataset = _state_and_dataset()

    heatmap = _prepare_theta_heatmap(state, dataset)

    assert heatmap.category_labels == ("a0", "p4", "o0", "c1")
    assert "p1" not in heatmap.category_labels
    assert heatmap.word_labels == ("push", "left", "round", "red")
    assert heatmap.category_boundaries == (1, 2, 3)
    assert heatmap.word_boundaries == (1, 2, 3)
    np.testing.assert_allclose(np.diag(heatmap.values), 0.85)


def test_theta_heatmap_plot_is_written(tmp_path) -> None:
    state, dataset = _state_and_dataset()
    path = tmp_path / "heatmap.png"

    _plot_theta(state, dataset, path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_inter_heatmaps_share_agent_a_word_axis_and_reorder_b_rows() -> None:
    state_a, dataset_a = _paired_heatmap_state(reverse_peaks=False)
    state_b, dataset_b = _paired_heatmap_state(reverse_peaks=True)

    heatmap_a = _prepare_theta_heatmap(state_a, dataset_a)
    independent_b = _prepare_theta_heatmap(state_b, dataset_b)
    heatmap_b = _prepare_theta_heatmap(
        state_b,
        dataset_b,
        reference=heatmap_a,
    )

    assert heatmap_a.word_labels == tuple(f"w{index}" for index in range(8))
    assert independent_b.word_labels != heatmap_a.word_labels
    assert heatmap_b.word_labels == heatmap_a.word_labels
    assert heatmap_b.word_boundaries == heatmap_a.word_boundaries == (2, 4, 6)
    assert heatmap_b.category_labels == (
        "a1",
        "a0",
        "p1",
        "p0",
        "o1",
        "o0",
        "c1",
        "c0",
    )
    assert heatmap_b.category_boundaries == (2, 4, 6)
    np.testing.assert_allclose(np.diag(heatmap_b.values), 0.85)


def test_ground_truth_word_counts_use_attention_and_inferred_attributes() -> None:
    state, dataset = _ground_truth_heatmap_state_and_dataset()

    counts = _ground_truth_word_counts(state, dataset)

    assert counts is not None
    labels = tuple(f"{attr}{category}" for attr, category in counts.category_keys)
    nonzero = {
        (labels[row], counts.vocabulary[column]): int(counts.values[row, column])
        for row, column in zip(*np.nonzero(counts.values))
    }
    assert nonzero == {
        ("p20", "w0"): 1,
        ("a10", "w1"): 1,
        ("c40", "w2"): 1,
        ("o30", "w3"): 1,
        ("o31", "w0"): 1,
        ("c41", "w1"): 1,
        ("a11", "w2"): 1,
        ("p21", "w3"): 1,
    }
    assert int(np.sum(counts.values)) == 8
    assert not any("9" in label for label in labels)


def test_ground_truth_inter_heatmaps_share_joint_semantic_axes() -> None:
    state_a, dataset_a = _ground_truth_heatmap_state_and_dataset()
    state_b, dataset_b = _ground_truth_heatmap_state_and_dataset()
    state_b.words = tuple(words[::-1].copy() for words in state_b.words)

    heatmap_a, heatmap_b = _prepare_ground_truth_heatmaps(
        state_a,
        dataset_a,
        state_b,
        dataset_b,
    )

    assert heatmap_a is not None
    assert heatmap_b is not None
    assert heatmap_a.category_labels == heatmap_b.category_labels
    assert heatmap_a.word_labels == heatmap_b.word_labels
    assert heatmap_a.category_boundaries == heatmap_b.category_boundaries == (
        2,
        4,
        6,
    )
    assert heatmap_a.word_boundaries == heatmap_b.word_boundaries
    assert int(np.sum(heatmap_a.values)) == 8
    assert int(np.sum(heatmap_b.values)) == 8


def test_ground_truth_heatmap_is_omitted_only_for_agent_without_truth() -> None:
    state_a, dataset_a = _ground_truth_heatmap_state_and_dataset()
    state_b, dataset_b = _ground_truth_heatmap_state_and_dataset()
    dataset_a = ObservationDataset(dataset_a.observations, dataset_a.attentions)

    heatmap_a, heatmap_b = _prepare_ground_truth_heatmaps(
        state_a,
        dataset_a,
        state_b,
        dataset_b,
    )

    assert heatmap_a is None
    assert heatmap_b is not None


def _ground_truth_heatmap_state_and_dataset() -> tuple[
    CslState, ObservationDataset
]:
    observations = {
        "a": np.ones((2, 1)),
        **{
            attr: (np.ones((2, 1)), np.ones((2, 1)))
            for attr in ("p", "o", "c")
        },
    }
    truth = {
        "a": np.array([10, 11]),
        "p": (np.array([20, 920]), np.array([921, 21])),
        "o": (np.array([30, 930]), np.array([931, 31])),
        "c": (np.array([40, 940]), np.array([941, 41])),
    }
    dataset = ObservationDataset(observations, np.array([0, 1]), truth)
    attributes = ("a", "p", "o", "c")
    state = CslState(
        z={
            "a": np.array([0, 0]),
            **{
                attr: (np.array([0, 0]), np.array([0, 0]))
                for attr in ("p", "o", "c")
            },
        },
        pi={attr: np.ones(1) for attr in attributes},
        means={attr: np.zeros((1, 1)) for attr in attributes},
        covariances={attr: np.ones((1, 1, 1)) for attr in attributes},
        theta=np.full((4, 4), 0.25),
        attributes=[("p", "a", "c", "o"), ("o", "c", "a", "p")],
        words=(np.arange(4), np.arange(4)),
        vocabulary=("w0", "w1", "w2", "w3"),
    )
    return state, dataset


def _paired_heatmap_state(
    *, reverse_peaks: bool
) -> tuple[CslState, ObservationDataset]:
    observations = {
        "a": np.ones((2, 1)),
        **{
            attr: (np.ones((1, 1)), np.ones((1, 1)))
            for attr in ("p", "o", "c")
        },
    }
    dataset = ObservationDataset(observations, np.array([0, 0]))
    categories = np.array([0, 1])
    z = {
        "a": categories.copy(),
        **{
            attr: (np.array([0]), np.array([1]))
            for attr in ("p", "o", "c")
        },
    }
    theta = np.full((8, 8), 0.05)
    for row in range(8):
        peak = row ^ 1 if reverse_peaks else row
        theta[row, peak] = 0.85
    pi = {attr: np.full(2, 0.5) for attr in ("a", "p", "o", "c")}
    means = {attr: np.zeros((2, 1)) for attr in pi}
    covariances = {attr: np.ones((2, 1, 1)) for attr in pi}
    state = CslState(
        z=z,
        pi=pi,
        means=means,
        covariances=covariances,
        theta=theta,
        attributes=[("a", "p", "o", "c")] * 2,
        words=(np.arange(4), np.arange(4, 8)),
        vocabulary=tuple(f"w{index}" for index in range(8)),
    )
    return state, dataset


def test_inter_metrics_use_each_agents_own_attention_and_truth() -> None:
    state_a, dataset_a = _inter_metric_state_and_dataset(
        attention=0,
        distractor_categories=np.array([0, 1, 0, 1]),
    )
    state_b, dataset_b = _inter_metric_state_and_dataset(
        attention=1,
        distractor_categories=np.array([1, 0, 1, 0]),
    )

    metrics = category_metrics(
        state_a,
        dataset_a,
        other_state=state_b,
        other_dataset=dataset_b,
    )

    assert metrics["ari_agents_p"] == 1.0
    assert metrics["ari_truth_A_p"] == 1.0
    assert metrics["ari_truth_B_p"] == 1.0
    assert metrics["ari_truth_p"] == metrics["ari_truth_A_p"]


def _inter_metric_state_and_dataset(
    *, attention: int, distractor_categories: np.ndarray
) -> tuple[CslState, ObservationDataset]:
    attended_categories = np.array([0, 0, 1, 1])
    assignments = []
    for attended, distractor in zip(attended_categories, distractor_categories):
        values = np.empty(2, dtype=np.int64)
        values[attention] = attended
        values[1 - attention] = distractor
        assignments.append(values)
    object_assignments = tuple(assignments)
    observations = {
        "a": np.ones((4, 1)),
        **{
            attr: tuple(np.ones((2, 1)) for _ in range(4))
            for attr in ("p", "o", "c")
        },
    }
    truth = {
        "a": attended_categories,
        "p": object_assignments,
        "o": object_assignments,
        "c": object_assignments,
    }
    dataset = ObservationDataset(
        observations,
        np.full(4, attention, dtype=np.int64),
        truth,
    )
    z = {
        "a": attended_categories.copy(),
        "p": tuple(values.copy() for values in object_assignments),
        "o": tuple(values.copy() for values in object_assignments),
        "c": tuple(values.copy() for values in object_assignments),
    }
    attributes = ("a", "p", "o", "c")
    state = CslState(
        z=z,
        pi={attr: np.array([0.5, 0.5]) for attr in attributes},
        means={attr: np.zeros((2, 1)) for attr in attributes},
        covariances={attr: np.ones((2, 1, 1)) for attr in attributes},
        theta=np.full((8, 4), 0.25),
        attributes=[attributes] * 4,
        words=tuple(np.arange(4) for _ in range(4)),
        vocabulary=("w0", "w1", "w2", "w3"),
    )
    return state, dataset
