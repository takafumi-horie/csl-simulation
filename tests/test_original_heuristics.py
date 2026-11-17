"""Sampling oracles and boundaries for the four original heuristics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest
from scipy.stats import invwishart, multivariate_normal

from legacy_csl.config import (
    AttributeCategoryCounts,
    Config,
    CoreHyperparameters,
    Freeze,
    OriginalHeuristics,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import ATTRIBUTES, ObservationDataset
from legacy_csl.model import CslModel
from legacy_csl.runner import accept_words


def _model(tmp_path, *, dimension=2, freeze=None, sampler="dirichlet", **heuristics):
    # Every attribute has an occupied category and an empty one, including
    # one-dimensional observations where scipy squeezes its return values.
    values = np.arange(6 * dimension, dtype=float).reshape(6, dimension) / 20
    dataset = ObservationDataset(
        observations={
            "a": values,
            **{attr: tuple(row[None, :] for row in values) for attr in ("p", "o", "c")},
        },
        attentions=np.zeros(6, dtype=np.int64),
        true_categories={
            "a": np.zeros(6, dtype=np.int64),
            **{
                attr: tuple(np.zeros(1, dtype=np.int64) for _ in range(6))
                for attr in ("p", "o", "c")
            },
        },
    )
    config = Config(
        mode="inter-csl",
        run=Run("heuristics", iterations=1, experiments=1, seed=0),
        paths=Paths(tmp_path / "data", tmp_path / "out"),
        word_model=WordModel(theta_sampler=sampler, importance_candidates=32),
        freeze=freeze or Freeze(),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=8,
        ),
        original_heuristics=OriginalHeuristics(**heuristics),
    )
    model = CslModel(dataset, config, rng=np.random.default_rng(17))
    for attr in ATTRIBUTES:
        if attr == "a":
            model.state.z[attr][:] = 0
        else:
            for row in model.state.z[attr]:
                row[:] = 0
    return model


@pytest.mark.parametrize("count", [2, 100])
def test_theta_is_average_of_actual_dirichlet_draws(tmp_path, count):
    model = _model(tmp_path, theta_mean_samples=count)
    oracle = deepcopy(model.rng)
    expected = np.asarray(
        [
            oracle.dirichlet(row, size=count).mean(axis=0)
            for row in model.theta_posterior_parameters()
        ]
    )
    model._sample_theta()
    np.testing.assert_array_equal(model.state.theta, expected)
    assert model.rng.bit_generator.state == oracle.bit_generator.state


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("empty", ["prior", "original_uniform"])
@pytest.mark.parametrize("count", [1, 100])
def test_gaussian_draws_and_empty_components(tmp_path, dimension, empty, count):
    model = _model(
        tmp_path,
        dimension=dimension,
        gaussian_mean_samples=count,
        empty_component_mean=empty,
    )
    oracle = deepcopy(model.rng)
    expected = {}
    for attr in ATTRIBUTES:
        observations = model.dataset.flatten(attr)
        prior = model._giw_prior(attr, dimension)
        means, covariances = [], []
        for selected in (observations, observations[:0]):
            if len(selected) == 0 and empty == "original_uniform":
                low, high = (-0.5, 0.5) if attr == "p" else (0, 1)
                mean = oracle.uniform(low, high, size=dimension)
                covariance = np.asarray(
                    invwishart.rvs(df=prior.nu0, scale=prior.V0, random_state=oracle)
                ).reshape(dimension, dimension)
                assert np.all((mean >= low) & (mean < high))
            elif len(selected) and count > 1:
                center, precision, scale, degrees = prior.posterior_parameters(selected)
                covariance = np.asarray(
                    invwishart.rvs(df=degrees, scale=scale, random_state=oracle)
                ).reshape(dimension, dimension)
                mean = np.asarray(
                    multivariate_normal.rvs(
                        mean=center,
                        cov=covariance / precision,
                        size=count,
                        random_state=oracle,
                    )
                ).reshape(count, dimension).mean(axis=0)
            else:
                mean, covariance = prior.sample_posterior(selected, rng=oracle)
            means.append(mean)
            covariances.append(covariance)
        expected[attr] = np.asarray(means), np.asarray(covariances)

    model._sample_gaussians()
    for attr in ATTRIBUTES:
        np.testing.assert_array_equal(model.state.means[attr], expected[attr][0])
        np.testing.assert_array_equal(model.state.covariances[attr], expected[attr][1])
    assert model.rng.bit_generator.state == oracle.bit_generator.state


@pytest.mark.parametrize(
    ("sampler", "count"),
    [("dirichlet", 1), ("dirichlet", 100), ("legacy_importance", 1)],
)
@pytest.mark.parametrize("consumer", ["raw", "rescaled"])
def test_inference_theta_selection_and_scan_order(
    tmp_path, monkeypatch, sampler, count, consumer
):
    model = _model(
        tmp_path,
        sampler=sampler,
        theta_mean_samples=count,
        inference_theta=consumer,
    )
    old_raw, old_scaled = model.state.theta.copy(), model.word_theta()
    new_raw = np.roll(old_raw, 1, axis=1)
    calls = []
    received = {}

    def capture(name, theta):
        calls.append(name)
        received[name] = theta.copy()

    def update_theta():
        calls.append("theta")
        model.state.theta = new_raw.copy()

    monkeypatch.setattr(model, "_sample_z", lambda theta: capture("z", theta))
    monkeypatch.setattr(model, "_sample_pi", lambda: calls.append("pi"))
    monkeypatch.setattr(model, "_sample_gaussians", lambda: calls.append("gaussians"))
    monkeypatch.setattr(model, "_sample_attributes", lambda theta: capture("F", theta))
    monkeypatch.setattr(model, "_sample_theta", update_theta)
    model.step()

    np.testing.assert_array_equal(
        received["z"], old_raw if consumer == "raw" else old_scaled
    )
    if count > 1 or sampler == "legacy_importance":
        assert calls == ["z", "pi", "gaussians", "F", "theta"]
        np.testing.assert_array_equal(received["F"], received["z"])
    else:
        assert calls == ["z", "pi", "gaussians", "theta", "F"]
        np.testing.assert_array_equal(
            received["F"], new_raw if consumer == "raw" else model.word_theta()
        )


def test_frozen_initialization_and_updates_ignore_learning_heuristics(tmp_path):
    model = _model(tmp_path, freeze=Freeze(gmm=True, theta=True, attributes=True))
    config = replace(
        model.config,
        original_heuristics=OriginalHeuristics(100, "raw", 100, "original_uniform"),
    )
    other = CslModel(model.dataset, config, rng=np.random.default_rng(17))
    for attr in ATTRIBUTES:
        np.testing.assert_array_equal(model.state.means[attr], other.state.means[attr])
        np.testing.assert_array_equal(
            model.state.covariances[attr], other.state.covariances[attr]
        )
    assert model.rng.bit_generator.state == other.rng.bit_generator.state
    before = other.state.copy()
    rng_before = deepcopy(other.rng.bit_generator.state)
    other.step()
    for attr in ATTRIBUTES:
        np.testing.assert_array_equal(other.state.means[attr], before.means[attr])
        np.testing.assert_array_equal(other.state.pi[attr], before.pi[attr])
    np.testing.assert_array_equal(other.state.theta, before.theta)
    assert other.state.attributes == before.attributes
    assert other.rng.bit_generator.state == rng_before


def test_frozen_gmm_initialization_matches_defaults_and_step_updates_F_theta(
    tmp_path, monkeypatch
):
    defaults = _model(tmp_path, freeze=Freeze(gmm=True))
    enabled = replace(
        defaults.config,
        original_heuristics=OriginalHeuristics(100, "raw", 100, "original_uniform"),
    )
    model = CslModel(defaults.dataset, enabled, rng=np.random.default_rng(17))
    before = model.state.copy()
    for attr in ATTRIBUTES:
        for field in ("pi", "means", "covariances"):
            np.testing.assert_array_equal(
                getattr(before, field)[attr], getattr(defaults.state, field)[attr]
            )
        if attr == "a":
            np.testing.assert_array_equal(before.z[attr], defaults.state.z[attr])
        else:
            for actual, expected in zip(before.z[attr], defaults.state.z[attr], strict=True):
                np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(before.theta, defaults.state.theta)
    assert before.attributes == defaults.state.attributes
    for actual, expected in zip(before.words, defaults.state.words, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert model.rng.bit_generator.state == defaults.rng.bit_generator.state

    # Enumerate raw-theta likelihood products independently of AttributeSampler.
    # After those F draws, posterior theta counts must use the newly drawn F.
    oracle = deepcopy(model.rng)
    expected_attributes = []
    alpha = np.full_like(before.theta, model.config.hyperparameters.gamma)
    for doc_id, words in enumerate(before.words):
        candidates = tuple(permutations(ATTRIBUTES, len(words)))
        likelihoods = []
        for candidate in candidates:
            factors = []
            for word, attr in zip(words, candidate, strict=True):
                category = (
                    before.z[attr][doc_id]
                    if attr == "a"
                    else before.z[attr][doc_id][model.dataset.attentions[doc_id]]
                )
                factors.append(before.theta[model.offsets[attr] + category, word])
            likelihoods.append(np.prod(factors))
        probabilities = np.asarray(likelihoods) / np.sum(likelihoods)
        sampled = candidates[oracle.choice(len(candidates), p=probabilities)]
        expected_attributes.append(sampled)
        for word, attr in zip(words, sampled, strict=True):
            category = (
                before.z[attr][doc_id]
                if attr == "a"
                else before.z[attr][doc_id][model.dataset.attentions[doc_id]]
            )
            alpha[model.offsets[attr] + category, word] += 1
    expected_theta = np.asarray(
        [oracle.dirichlet(row, size=100).mean(axis=0) for row in alpha]
    )
    calls = []
    sample_attributes, sample_theta = model._sample_attributes, model._sample_theta

    def forbidden_gmm_update(*args, **kwargs):
        pytest.fail("freeze.gmm must skip every GMM update")

    def update_attributes(theta):
        calls.append("F")
        np.testing.assert_array_equal(theta, before.theta)
        sample_attributes(theta)

    def update_theta():
        calls.append("theta")
        assert model.state.attributes == expected_attributes
        sample_theta()

    for method in ("_sample_z", "_sample_pi", "_sample_gaussians"):
        monkeypatch.setattr(model, method, forbidden_gmm_update)
    monkeypatch.setattr(model, "_sample_attributes", update_attributes)
    monkeypatch.setattr(model, "_sample_theta", update_theta)
    model.step()

    assert calls == ["F", "theta"]
    assert model.state.attributes == expected_attributes
    np.testing.assert_array_equal(model.state.theta, expected_theta)
    assert not np.array_equal(model.state.theta, before.theta)
    assert model.rng.bit_generator.state == oracle.bit_generator.state
    for attr in ATTRIBUTES:
        for field in ("pi", "means", "covariances"):
            np.testing.assert_array_equal(
                getattr(model.state, field)[attr], getattr(before, field)[attr]
            )
        if attr == "a":
            np.testing.assert_array_equal(model.state.z[attr], before.z[attr])
        else:
            for actual, expected in zip(model.state.z[attr], before.z[attr], strict=True):
                np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(model.state.words, before.words, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_learning_options_do_not_change_word_and_prediction_consumers(tmp_path):
    model = _model(tmp_path)
    enabled = replace(
        model.config,
        original_heuristics=OriginalHeuristics(100, "raw", 100, "original_uniform"),
    )
    other = CslModel.from_state(
        model.dataset, enabled, model.state, rng=deepcopy(model.rng)
    )
    np.testing.assert_array_equal(model.word_theta(), other.word_theta())
    for first, second in zip(model.draw_words(), other.draw_words(), strict=True):
        np.testing.assert_array_equal(first, second)
    proposals = tuple((row + 1) % len(model.vocabulary) for row in model.state.words)
    for first, second in zip(
        accept_words(model, proposals, "mh"),
        accept_words(other, proposals, "mh"),
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)
    predicted = model.sample_categories_without_observations(model.state.words)
    other_predicted = other.sample_categories_without_observations(other.state.words)
    for attr in ATTRIBUTES:
        if attr == "a":
            np.testing.assert_array_equal(predicted[attr], other_predicted[attr])
        else:
            for first, second in zip(predicted[attr], other_predicted[attr], strict=True):
                np.testing.assert_array_equal(first, second)
    assert model.rng.bit_generator.state == other.rng.bit_generator.state
