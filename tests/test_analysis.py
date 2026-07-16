"""Tests for analysis labels and attribute-safe heatmap ordering.

解析labelと属性を跨がないheatmap整列のテスト。
"""

from __future__ import annotations

import numpy as np

from legacy_csl.analysis import (
    _plot_theta,
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
