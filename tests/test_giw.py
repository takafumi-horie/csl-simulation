"""Tests for Gaussian--Inverse-Wishart inference."""

from __future__ import annotations

import numpy as np
import pytest

from legacy_csl.giw import GiwPrior, posterior_parameters


def _prior() -> GiwPrior:
    return GiwPrior(
        m0=np.array([0.1, 0.2]),
        kappa0=0.5,
        V0=np.array([[1.0, 0.25], [0.25, 1.5]]),
        nu0=3.0,
    )


def _observations() -> np.ndarray:
    return np.array(
        [[1.1, 2.2], [1.3, 2.4], [1.5, 2.6], [1.7, 2.8], [1.9, 3.0]],
        dtype=np.float64,
    )


def test_posterior_parameters_match_legacy_formula() -> None:
    prior = _prior()
    observations = _observations()

    m_n, kappa_n, V_n, nu_n = posterior_parameters(prior, observations)

    expected_kappa = 5.5
    expected_mean = (0.5 * np.array([0.1, 0.2]) + observations.sum(axis=0)) / 5.5
    expected_scale = (
        prior.V0
        + observations.T @ observations
        + 0.5 * np.outer([0.1, 0.2], [0.1, 0.2])
        - expected_kappa * np.outer(expected_mean, expected_mean)
    )
    np.testing.assert_allclose(m_n, expected_mean)
    assert kappa_n == expected_kappa
    np.testing.assert_allclose(V_n, expected_scale)
    assert nu_n == 8.0


def test_empty_cluster_posterior_is_exactly_prior() -> None:
    prior = _prior()

    m_n, kappa_n, V_n, nu_n = prior.posterior_parameters(
        np.empty((0, prior.dimension))
    )

    np.testing.assert_array_equal(m_n, prior.m0)
    assert kappa_n == prior.kappa0
    np.testing.assert_array_equal(V_n, prior.V0)
    assert nu_n == prior.nu0


def test_sampling_is_reproducible_and_preserves_full_covariance() -> None:
    first_prior = _prior().sample_prior(rng=np.random.default_rng(17))
    second_prior = _prior().sample_prior(rng=np.random.default_rng(17))
    first_posterior = _prior().sample_posterior(
        _observations(), rng=np.random.default_rng(29)
    )
    second_posterior = _prior().sample_posterior(
        _observations(), rng=np.random.default_rng(29)
    )

    for first, second in zip(first_prior, second_prior, strict=True):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(first_posterior, second_posterior, strict=True):
        np.testing.assert_array_equal(first, second)

    mean, covariance = first_posterior
    assert mean.shape == (2,)
    assert covariance.shape == (2, 2)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert covariance[0, 1] != 0.0


def test_empty_cluster_sample_matches_prior_sample_with_same_seed() -> None:
    prior = _prior()
    prior_sample = prior.sample_prior(rng=np.random.default_rng(41))
    posterior_sample = prior.sample_posterior([], rng=np.random.default_rng(41))

    for expected, actual in zip(prior_sample, posterior_sample, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"m0": [], "kappa0": 1.0, "V0": [[1.0]], "nu0": 2.0}, "m0"),
        (
            {"m0": [0.0, 0.0], "kappa0": 0.0, "V0": np.eye(2), "nu0": 3.0},
            "kappa0",
        ),
        (
            {
                "m0": [0.0, 0.0],
                "kappa0": 1.0,
                "V0": [[1.0, 2.0], [0.0, 1.0]],
                "nu0": 3.0,
            },
            "symmetric",
        ),
        (
            {
                "m0": [0.0, 0.0],
                "kappa0": 1.0,
                "V0": [[1.0, 2.0], [2.0, 1.0]],
                "nu0": 3.0,
            },
            "positive definite",
        ),
        (
            {"m0": [0.0, 0.0], "kappa0": 1.0, "V0": np.eye(2), "nu0": 1.0},
            "dimension - 1",
        ),
    ],
)
def test_prior_rejects_invalid_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GiwPrior(**kwargs)  # type: ignore[arg-type]


def test_observations_are_validated() -> None:
    prior = _prior()

    with pytest.raises(ValueError, match="shape"):
        prior.posterior_parameters([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="finite"):
        prior.posterior_parameters([[1.0, np.nan]])


def test_prior_does_not_expose_mutable_parameter_arrays() -> None:
    source_mean = np.array([0.0, 1.0])
    source_scale = np.array([[1.0, 0.2], [0.2, 1.0]])
    prior = GiwPrior(source_mean, 1.0, source_scale, 3.0)
    source_mean[0] = 99.0
    source_scale[0, 0] = 99.0

    exposed_mean = prior.m0
    exposed_scale = prior.V0
    exposed_mean[0] = 42.0
    exposed_scale[0, 0] = 42.0

    np.testing.assert_array_equal(prior.m0, [0.0, 1.0])
    np.testing.assert_array_equal(prior.V0, [[1.0, 0.2], [0.2, 1.0]])
