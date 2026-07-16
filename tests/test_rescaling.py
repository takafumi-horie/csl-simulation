"""Tests for word rescaling and theta sampling."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from legacy_csl.rescaling import (
    empirical_unigram_rescaling,
    legacy_rescaling_term,
    legacy_theta_sum_rescaling,
    rescale_theta,
    sample_dirichlet_theta,
    sample_legacy_importance_theta,
    sample_theta,
)


def test_legacy_theta_sum_matches_hand_calculation() -> None:
    theta = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)

    actual = legacy_theta_sum_rescaling(theta)

    unnormalized = np.array([[0.8 / 1.2, 0.2 / 0.8], [0.4 / 1.2, 0.6 / 0.8]])
    expected = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert actual.dtype == np.float64


def test_legacy_rescaling_term_is_category_sum() -> None:
    theta = np.array([[0.8, 0.2], [0.4, 0.6]])

    actual = legacy_rescaling_term(theta)

    np.testing.assert_allclose(actual, [1.2, 0.8])
    assert actual.dtype == np.float64


def test_rescaling_does_not_mutate_input() -> None:
    theta = np.array([[0.75, 0.25], [0.2, 0.8]], dtype=np.float32)
    original = theta.copy()

    legacy_theta_sum_rescaling(theta)
    empirical_unigram_rescaling(theta, [3.0, 1.0])
    rescale_theta(theta, "none")

    np.testing.assert_array_equal(theta, original)


def test_empirical_unigram_rescaling_accepts_counts() -> None:
    theta = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    counts = np.array([6.0, 3.0, 1.0])

    actual = empirical_unigram_rescaling(theta, counts)

    expected_weights = theta / counts
    expected = expected_weights / expected_weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "theta",
    [
        [[1.0, -0.1], [0.2, 0.8]],
        [[1.0, np.nan], [0.2, 0.8]],
        [[0.0, 0.0], [0.2, 0.8]],
        [1.0, 2.0],
        [],
    ],
)
def test_rescaling_rejects_invalid_theta(theta: object) -> None:
    with pytest.raises(ValueError):
        rescale_theta(theta, "none")


def test_legacy_rescaling_rejects_zero_mass_column() -> None:
    with pytest.raises(ValueError, match="column"):
        legacy_theta_sum_rescaling([[1.0, 0.0], [1.0, 0.0]])


@pytest.mark.parametrize("unigram", [[1.0], [1.0, 0.0], [1.0, np.inf]])
def test_empirical_rescaling_rejects_invalid_unigram(unigram: object) -> None:
    with pytest.raises(ValueError):
        empirical_unigram_rescaling([[0.5, 0.5], [0.4, 0.6]], unigram)


def test_rescaling_dispatch_validates_mode_arguments() -> None:
    theta = [[0.5, 0.5], [0.4, 0.6]]
    with pytest.raises(ValueError, match="required"):
        rescale_theta(theta, "empirical_unigram")
    with pytest.raises(ValueError, match="unknown"):
        rescale_theta(theta, "other")  # type: ignore[arg-type]


def test_none_rescaling_returns_equal_independent_float64_copy() -> None:
    theta = np.array([[0.75, 0.25], [0.2, 0.8]], dtype=np.float32)

    actual = rescale_theta(theta, "none")

    np.testing.assert_allclose(actual, theta)
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, theta)


def test_dirichlet_sampler_is_seed_reproducible_and_normalized() -> None:
    alpha = np.array([[2.0, 3.0, 4.0], [4.0, 2.0, 1.0]])

    first = sample_dirichlet_theta(alpha, rng=np.random.default_rng(12))
    second = sample_dirichlet_theta(alpha, rng=np.random.default_rng(12))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_legacy_importance_sampler_is_seed_reproducible() -> None:
    alpha = np.array([[2.0, 1.0, 3.0], [1.0, 4.0, 2.0]])
    counts = np.array([4, 1, 3])

    first = sample_legacy_importance_theta(
        alpha, counts, rng=np.random.default_rng(32), n_candidates=40
    )
    second = sample_legacy_importance_theta(
        alpha, counts, rng=np.random.default_rng(32), n_candidates=40
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_legacy_importance_default_candidate_count_is_original_value() -> None:
    default = inspect.signature(sample_legacy_importance_theta).parameters[
        "n_candidates"
    ].default

    assert default == 10_000


def test_legacy_importance_uses_column_sum_logweights() -> None:
    alpha = np.array([[2.0, 1.0], [1.0, 3.0]])
    counts = np.array([5.0, 2.0])
    expected_rng = np.random.default_rng(91)
    candidates = np.empty((8, 2, 2))
    for category, row in enumerate(alpha):
        candidates[:, category, :] = expected_rng.dirichlet(row, size=8)
    log_weights = -(np.log(candidates.sum(axis=1)) @ counts)
    weights = np.exp(log_weights - log_weights.max())
    selected = expected_rng.choice(8, p=weights / weights.sum())

    actual = sample_legacy_importance_theta(
        alpha, counts, rng=np.random.default_rng(91), n_candidates=8
    )

    np.testing.assert_array_equal(actual, candidates[selected])


def test_theta_sampler_dispatch_switches_methods() -> None:
    alpha = np.array([[2.0, 3.0], [4.0, 1.0]])
    expected = sample_dirichlet_theta(alpha, rng=np.random.default_rng(5))

    actual = sample_theta(
        alpha,
        rng=np.random.default_rng(5),
        sampler="dirichlet",
        rescaling_mode="none",
    )

    np.testing.assert_array_equal(actual, expected)


def test_theta_sampler_rejects_invalid_legacy_combination() -> None:
    alpha = [[2.0, 3.0], [4.0, 1.0]]
    with pytest.raises(ValueError, match="legacy_theta_sum"):
        sample_theta(
            alpha,
            rng=np.random.default_rng(0),
            sampler="legacy_importance",
            word_counts=[2, 1],
            rescaling_mode="none",
            n_candidates=4,
        )


def test_theta_sampler_requires_counts_and_known_method() -> None:
    alpha = [[2.0, 3.0], [4.0, 1.0]]
    with pytest.raises(ValueError, match="word_counts"):
        sample_theta(alpha, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="unknown"):
        sample_theta(
            alpha,
            rng=np.random.default_rng(0),
            sampler="other",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("counts", "n_candidates"),
    [([1.0], 4), ([1.0, -1.0], 4), ([1.0, 2.0], 0), ([1.0, 2.0], True)],
)
def test_legacy_sampler_rejects_invalid_inputs(
    counts: object, n_candidates: object
) -> None:
    with pytest.raises(ValueError):
        sample_legacy_importance_theta(
            [[2.0, 3.0], [4.0, 1.0]],
            counts,
            rng=np.random.default_rng(0),
            n_candidates=n_candidates,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "alpha",
    [[[2.0, 0.0], [1.0, 3.0]], [[2.0, np.nan]], [1.0, 2.0], []],
)
def test_theta_samplers_reject_invalid_dirichlet_parameters(alpha: object) -> None:
    with pytest.raises(ValueError):
        sample_dirichlet_theta(alpha, rng=np.random.default_rng(0))
