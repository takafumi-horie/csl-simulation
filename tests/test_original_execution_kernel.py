"""Small independent checks of the original execution contract."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import invwishart, multivariate_normal

from legacy_csl.config import (
    AttributeCategoryCounts,
    Config,
    CoreHyperparameters,
    OriginalHeuristics,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import ATTRIBUTES, ObservationDataset
from legacy_csl.model import CslModel
from legacy_csl.original import initialize_original_models


def _inputs(tmp_path, *, dimension=2, vocabulary_size=14):
    values = np.arange(8 * dimension, dtype=float).reshape(8, dimension) / 20
    dataset = ObservationDataset(
        observations={
            "a": values,
            **{attr: tuple(row[None, :] for row in values) for attr in ("p", "o", "c")},
        },
        attentions=np.zeros(8, dtype=np.int64),
    )
    config = Config(
        mode="single-csl",
        run=Run("original", iterations=2, experiments=1, seed=2),
        paths=Paths(tmp_path / "data", tmp_path / "out"),
        word_model=WordModel(rescaling="legacy_theta_sum", theta_sampler="dirichlet"),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=vocabulary_size,
        ),
        original_heuristics=OriginalHeuristics(
            execution_profile="original",
            theta_mean_samples=100,
            inference_theta="raw",
            gaussian_mean_samples=100,
            empty_component_mean="original_uniform",
        ),
    )
    words = tuple(row for row in (np.arange(32) % vocabulary_size).reshape(8, 4))
    return dataset, config, words


def _assert_rng_equal(actual, expected):
    actual_state, expected_state = actual.get_state(), expected.get_state()
    assert actual_state[0] == expected_state[0]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]


def test_constructor_rejection_consumes_no_rng_and_prediction_loads(tmp_path):
    dataset, config, words = _inputs(tmp_path)
    rng = np.random.RandomState(2)
    before = deepcopy(rng)
    with pytest.raises(ValueError, match="joint initialize_original_models"):
        CslModel(dataset, config, rng=rng, initial_words=words)
    _assert_rng_equal(rng, before)
    (model,) = initialize_original_models(
        [dataset], config, rng=rng, initial_words=words
    )
    loaded = CslModel.from_state(
        dataset, config, model.state, rng=np.random.default_rng(3)
    )
    assert type(loaded) is CslModel
    np.testing.assert_array_equal(loaded.state.theta, model.state.theta)
    before_theta = loaded.state.theta.copy()
    before_rng = deepcopy(loaded.rng.bit_generator.state)
    with pytest.raises(ValueError, match="not a learning resume"):
        loaded.step()
    np.testing.assert_array_equal(loaded.state.theta, before_theta)
    assert loaded.rng.bit_generator.state == before_rng


@pytest.mark.parametrize("dimension", [1, 2])
def test_single_keeps_observed_words_and_local_rng(tmp_path, dimension):
    dataset, config, words = _inputs(tmp_path, dimension=dimension)
    global_before = np.random.get_state()
    (model,) = initialize_original_models(
        [dataset], config, rng=np.random.RandomState(2), initial_words=words
    )
    assert len(model.vocabulary) == 14
    for _ in range(2):
        model.step()
        for actual, expected in zip(model.state.words, words):
            np.testing.assert_array_equal(actual, expected)
        for attr in ATTRIBUTES:
            categories = model.state.z[attr]
            flat = categories if attr == "a" else np.concatenate(categories)
            np.testing.assert_array_equal(
                model.pi_posterior_parameters(attr),
                np.bincount(flat, minlength=2) + 1.0,
            )
    global_after = np.random.get_state()
    np.testing.assert_array_equal(global_after[1], global_before[1])
    assert global_after[2:] == global_before[2:]


def test_gaussian_draws_match_centered_source_formula_and_empty_order(tmp_path):
    dataset, config, words = _inputs(tmp_path)
    (model,) = initialize_original_models(
        [dataset], config, rng=np.random.RandomState(2), initial_words=words
    )
    for attr in ATTRIBUTES:
        if attr == "a":
            model.state.z[attr][:] = 0
        else:
            for row in model.state.z[attr]:
                row[:] = 0
    oracle = deepcopy(model.rng)
    expected = {}
    for attr in ATTRIBUTES:
        samples = [np.array(row) for row in dataset.flatten(attr)]
        prior = model._giw_prior(attr, 2)
        n = len(samples)
        xbar = sum(samples) / float(n)
        k = prior.kappa0 + n
        center = (prior.kappa0 * prior.m0 + n * xbar) / k
        scale = (
            prior.V0
            + sum(np.dot(np.array([x - xbar]).T, np.array([x - xbar])) for x in samples)
            + (prior.kappa0 * n / k)
            * np.dot(np.array([xbar - prior.m0]).T, np.array([xbar - prior.m0]))
        )
        covariance = invwishart.rvs(df=prior.nu0 + n, scale=scale, random_state=oracle)
        mean = np.mean(
            multivariate_normal.rvs(
                mean=center, cov=covariance / k, size=100, random_state=oracle
            ),
            0,
        )
        limits = (-0.5, 0.5) if attr == "p" else (0, 1)
        empty_mean = np.array([oracle.uniform(*limits) for _ in range(2)])
        empty_covariance = invwishart.rvs(
            df=prior.nu0, scale=prior.V0, random_state=oracle
        )
        expected[attr] = (
            np.asarray([mean, empty_mean]),
            np.asarray([covariance, empty_covariance]),
        )
    model._sample_gaussians()
    for attr in ATTRIBUTES:
        np.testing.assert_array_equal(model.state.means[attr], expected[attr][0])
        np.testing.assert_array_equal(model.state.covariances[attr], expected[attr][1])
    _assert_rng_equal(model.rng, oracle)


def test_input_validation_and_underflow_provide_context(tmp_path):
    dataset, config, words = _inputs(tmp_path)
    rng = np.random.RandomState(2)
    before = deepcopy(rng)
    with pytest.raises(ValueError, match="effective vocabulary"):
        initialize_original_models(
            [dataset], config, rng=rng, initial_words=np.zeros((8, 4))
        )
    _assert_rng_equal(rng, before)
    (model,) = initialize_original_models(
        [dataset], config, rng=rng, initial_words=words, experiment_id=0
    )
    model.state.means["a"][:] = 1e10
    with pytest.raises(
        ValueError,
        match=r"experiment=0, seed=2, iteration=1, agent=U.*attribute=a.*nonpositive",
    ):
        model.step()


def test_inter_vocabulary_omission_is_explicit_and_does_not_retry(tmp_path):
    dataset, config, _ = _inputs(tmp_path)
    config = replace(
        config,
        mode="inter-csl",
        hyperparameters=replace(config.hyperparameters, vocabulary_size=100),
    )
    with pytest.raises(
        ValueError, match=r"seed=2.*agent=A.*initial internal words omit vocabulary IDs"
    ):
        initialize_original_models(
            [dataset, dataset], config, rng=np.random.RandomState(2), experiment_id=0
        )
