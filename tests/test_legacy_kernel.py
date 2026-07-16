"""Formula-level oracles for the legacy categorical inference kernel."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import multivariate_normal

from legacy_csl.config import (
    AttributeCategoryCounts,
    AttributeModel,
    Config,
    CoreHyperparameters,
    Freeze,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import ObservationDataset, load_dataset, write_compact_fixture
from legacy_csl.model import CslModel
from legacy_csl.runner import accept_words
from legacy_csl.state_io import load_state, save_state


def _config(tmp_path, *, sampler="dirichlet", rescaling="none", freeze=None):
    return Config(
        mode="inter-csl",
        run=Run(name="oracle", iterations=1, experiments=1, seed=2),
        paths=Paths(tmp_path / "data", tmp_path / "output"),
        word_model=WordModel(
            rescaling=rescaling,
            theta_sampler=sampler,
            importance_candidates=8,
        ),
        attribute_model=AttributeModel("distinct"),
        freeze=freeze or Freeze(),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=8,
            word_length=4,
            gamma=0.25,
        ),
    )


def _model(tmp_path, **kwargs):
    config = _config(tmp_path, **kwargs)
    write_compact_fixture(config.paths.data_root, num_documents=6)
    dataset = load_dataset(config.paths.data_root, "A")
    return CslModel(dataset, config, rng=np.random.default_rng(40))


def test_z_logweights_use_full_covariance_and_words_only_for_attention(tmp_path):
    model = _model(tmp_path)
    model.state.pi["p"] = np.array([0.35, 0.65])
    model.state.means["p"] = np.array([[0.0, 0.0], [1.0, -0.5]])
    model.state.covariances["p"] = np.array(
        [[[1.0, 0.6], [0.6, 2.0]], [[0.7, -0.2], [-0.2, 1.3]]]
    )
    model.state.attributes[0] = ("p", "a", "o", "c")
    model.state.words = (
        np.array([2, 0, 1, 3]),
        *model.state.words[1:],
    )
    theta = np.full_like(model.state.theta, 1.0 / model.state.theta.shape[1])
    offset = model.offsets["p"]
    theta[offset, 2] = 0.5
    theta[offset, :] /= theta[offset, :].sum()
    theta[offset + 1, 2] = 0.05
    theta[offset + 1, :] /= theta[offset + 1, :].sum()
    observation_docs = model.dataset.document_arrays("p")

    attended = model.z_logweights("p", 0, 0, theta)
    distractor = model.z_logweights("p", 0, 1, theta)
    expected_observation_attended = np.array(
        [
            multivariate_normal.logpdf(
                observation_docs[0][0],
                mean=model.state.means["p"][category],
                cov=model.state.covariances["p"][category],
            )
            for category in range(2)
        ]
    )
    expected_observation_distractor = np.array(
        [
            multivariate_normal.logpdf(
                observation_docs[0][1],
                mean=model.state.means["p"][category],
                cov=model.state.covariances["p"][category],
            )
            for category in range(2)
        ]
    )

    np.testing.assert_allclose(
        attended,
        np.log(model.state.pi["p"])
        + expected_observation_attended
        + np.log(theta[offset : offset + 2, 2]),
    )
    np.testing.assert_allclose(
        distractor,
        np.log(model.state.pi["p"]) + expected_observation_distractor,
    )


def test_observation_category_logweights_use_only_pi_and_full_gaussian(tmp_path):
    model = _model(tmp_path)
    model.state.pi["p"] = np.array([0.35, 0.65])
    model.state.means["p"] = np.array([[0.0, 0.0], [1.0, -0.5]])
    model.state.covariances["p"] = np.array(
        [[[1.0, 0.6], [0.6, 2.0]], [[0.7, -0.2], [-0.2, 1.3]]]
    )
    observation = np.array([0.4, -0.1])
    expected = np.log(model.state.pi["p"]) + np.asarray(
        [
            multivariate_normal.logpdf(
                observation,
                mean=model.state.means["p"][category],
                cov=model.state.covariances["p"][category],
            )
            for category in range(2)
        ]
    )

    before = model.observation_category_logweights("p", observation)
    model.state.words = tuple(np.full(4, 7, dtype=np.int64) for _ in model.state.words)
    model.state.theta[:] = 1e-12
    model.state.theta[:, 7] = 1.0
    after = model.observation_category_logweights("p", observation)

    np.testing.assert_allclose(before, expected)
    np.testing.assert_allclose(after, expected)


def test_observation_prediction_samples_action_and_every_ragged_object(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    object_documents = (
        np.array([[0.1], [9.9]]),
        np.array([[9.7], [0.3], [10.1]]),
    )
    dataset = ObservationDataset(
        {
            "a": np.array([[9.8], [0.2]]),
            "p": object_documents,
            "o": object_documents,
            "c": object_documents,
        },
        np.array([1, 0]),
    )
    model = CslModel(dataset, config, rng=np.random.default_rng(74))
    for attr in ("a", "p", "o", "c"):
        model.state.pi[attr] = np.array([0.5, 0.5])
        model.state.means[attr] = np.array([[0.0], [10.0]])
        model.state.covariances[attr] = np.array([[[0.05]], [[0.05]]])
    monkeypatch.setattr(
        "legacy_csl.model._sample_logweights",
        lambda logweights, rng: int(np.argmax(logweights)),
    )

    sampled = model.sample_categories_from_observations()

    np.testing.assert_array_equal(sampled["a"], [1, 0])
    for attr in ("p", "o", "c"):
        assert isinstance(sampled[attr], tuple)
        np.testing.assert_array_equal(sampled[attr][0], [0, 1])
        np.testing.assert_array_equal(sampled[attr][1], [1, 0, 1])


def test_pi_posterior_counts_every_object(tmp_path):
    model = _model(tmp_path)
    model.state.z["p"] = (
        np.array([0, 1]),
        np.array([1, 1, 0]),
        np.array([0, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1]),
        np.array([1, 1, 1]),
    )

    actual = model.pi_posterior_parameters("p")

    np.testing.assert_allclose(actual, np.array([7.0, 10.0]))


def test_theta_posterior_uses_attended_z_and_attribute_columns(tmp_path):
    model = _model(tmp_path)
    model.state.attributes[0] = ("p", "a", "o", "c")
    model.state.words = (np.array([2, 0, 1, 3]), *model.state.words[1:])
    expected = np.full_like(model.state.theta, 0.25)
    for doc_id, attributes in enumerate(model.state.attributes):
        for word, attr in zip(model.state.words[doc_id], attributes):
            row = model.global_category(attr, model.attended_category(attr, doc_id))
            expected[row, word] += 1.0

    np.testing.assert_allclose(model.theta_posterior_parameters(), expected)


def test_scan_order_matches_each_reference_sampler(tmp_path):
    model = _model(tmp_path)
    events: list[str] = []
    model._sample_z = lambda theta: events.append("z")  # type: ignore[method-assign]
    model._sample_pi = lambda: events.append("pi")  # type: ignore[method-assign]
    model._sample_gaussians = lambda: events.append("phi")  # type: ignore[method-assign]
    model._sample_theta = lambda: events.append("theta")  # type: ignore[method-assign]
    model._sample_attributes = lambda theta: events.append("F")  # type: ignore[method-assign]

    model.step()

    assert events == ["z", "pi", "phi", "theta", "F"]

    legacy = _model(
        tmp_path / "legacy",
        sampler="legacy_importance",
        rescaling="legacy_theta_sum",
    )
    legacy_events: list[str] = []
    legacy._sample_z = lambda theta: legacy_events.append("z")  # type: ignore[method-assign]
    legacy._sample_pi = lambda: legacy_events.append("pi")  # type: ignore[method-assign]
    legacy._sample_gaussians = lambda: legacy_events.append("phi")  # type: ignore[method-assign]
    legacy._sample_theta = lambda: legacy_events.append("theta")  # type: ignore[method-assign]
    legacy._sample_attributes = (  # type: ignore[method-assign]
        lambda theta: legacy_events.append("F")
    )

    legacy.step()

    assert legacy_events == ["z", "pi", "phi", "F", "theta"]


def test_empirical_mh_uses_raw_theta_divided_by_snapshot_counts(tmp_path):
    model = _model(tmp_path, rescaling="empirical_unigram")
    model.state.attributes = [("a", "p", "o", "c")] * 6
    model.state.theta[:] = 1.0 / 8.0
    current = tuple(np.array([0, 0, 1, 1]) for _ in range(6))
    suggested = tuple(np.array([2, 2, 2, 3]) for _ in range(6))
    model.state.words = current

    class DeterministicGenerator:
        def random(self):
            return 0.5

    model.rng = DeterministicGenerator()  # type: ignore[assignment]
    accepted = accept_words(model, suggested, "mh")

    # The snapshot counts are n(0)=n(1)=12, n(2)=18, and n(3)=6.
    # Thus the ratios are 12/18 for 0 -> 2 and 12/6 for 1 -> 3.
    assert all(np.array_equal(left, right) for left, right in zip(accepted, suggested))


def test_checkpoint_roundtrip_preserves_ragged_z_and_full_covariance(tmp_path):
    model = _model(tmp_path)
    directory = tmp_path / "checkpoint"

    save_state(model.state, directory, "A")
    loaded = load_state(directory, "A")

    for attr in ("p", "o", "c"):
        assert isinstance(loaded.z[attr], tuple)
        for actual, expected in zip(loaded.z[attr], model.state.z[attr]):
            np.testing.assert_array_equal(actual, expected)
    for attr in ("a", "p", "o", "c"):
        np.testing.assert_allclose(loaded.covariances[attr], model.state.covariances[attr])
        assert loaded.covariances[attr].ndim == 3
    assert loaded.vocabulary == model.state.vocabulary


def test_freeze_gmm_uses_truth_and_remains_fixed(tmp_path):
    config = _config(
        tmp_path,
        freeze=Freeze(gmm=True),
    )
    config = replace(
        config,
        hyperparameters=replace(
            config.hyperparameters,
            category_counts=AttributeCategoryCounts(a=3, p=3, o=3, c=3),
        ),
    )
    write_compact_fixture(config.paths.data_root, num_documents=6)
    dataset = load_dataset(config.paths.data_root, "A")
    model = CslModel(dataset, config, rng=np.random.default_rng(52))
    before = model.state.copy()

    model.step()

    for attr in ("a", "p", "o", "c"):
        truth = dataset.true_categories[attr]  # type: ignore[index]
        if isinstance(truth, tuple):
            for actual, expected in zip(model.state.z[attr], truth):
                np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(model.state.z[attr], truth)
        np.testing.assert_array_equal(model.state.pi[attr], before.pi[attr])
        np.testing.assert_array_equal(model.state.means[attr], before.means[attr])
        np.testing.assert_array_equal(
            model.state.covariances[attr], before.covariances[attr]
        )
