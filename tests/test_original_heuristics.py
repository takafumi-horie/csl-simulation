"""Sampling oracles and boundaries for the four original heuristics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

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


def _model(tmp_path, *, dimension=2, freeze=None, **heuristics):
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
        word_model=WordModel(theta_sampler="dirichlet"),
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


@pytest.mark.parametrize("count", [1, 100])
@pytest.mark.parametrize("consumer", ["raw", "rescaled"])
def test_inference_theta_selection_and_scan_order(tmp_path, monkeypatch, count, consumer):
    model = _model(tmp_path, theta_mean_samples=count, inference_theta=consumer)
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
    if count > 1:
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
