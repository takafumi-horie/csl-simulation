"""Independent synthetic INTER trajectory and targeted draw-contract tests."""

from copy import deepcopy

import numpy as np
import pytest

import _original_reference as reference
from legacy_csl.config import (
    AttributeCategoryCounts,
    AttributeScalars,
    Config,
    CoreHyperparameters,
    GmmHyperparameters,
    OriginalHeuristics,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import ObservationDataset
from legacy_csl.original import initialize_original_models
from legacy_csl.runner import accept_words


class RecordingRandomState(np.random.RandomState):
    """Record explicit distribution calls while preserving the NumPy stream."""

    def __init__(self, seed):
        super().__init__(seed)
        self.calls = []

    def uniform(self, low=0.0, high=1.0, size=None):
        self.calls.append(("uniform", low, high, size))
        return super().uniform(low, high, size)

    def dirichlet(self, alpha, size=None):
        self.calls.append(("dirichlet", np.array(alpha), size))
        return super().dirichlet(alpha, size)

    def beta(self, a, b, size=None):
        self.calls.append(("beta", a, b, size))
        return super().beta(a, b, size)


def inputs_and_config(tmp_path, seed):
    inputs = reference.synthetic_inputs()
    settings = reference.SETTINGS
    datasets = [
        ObservationDataset(
            observations={
                "a": np.array([rows[0] for rows in observations["a"]]),
                **{attr: tuple(observations[attr]) for attr in ("p", "o", "c")},
            },
            attentions=attention,
        )
        for observations, attention in inputs
    ]
    gmm = GmmHyperparameters(
        mixture_concentration=AttributeScalars(**settings["concentration"]),
        mean_prior=AttributeScalars(**settings["mean"]),
        mean_precision=AttributeScalars(**settings["precision"]),
        covariance_scale=AttributeScalars(**settings["scale"]),
        covariance_df_offset=AttributeScalars(**settings["df_offset"]),
    )
    config = Config(
        mode="inter-csl",
        run=Run("independent-reference", iterations=1, experiments=1, seed=seed),
        paths=Paths(tmp_path / "data", tmp_path / "out"),
        word_model=WordModel(rescaling="legacy_theta_sum", theta_sampler="dirichlet"),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(**settings["categories"]),
            vocabulary_size=settings["vocabulary"],
            gamma=settings["gamma"],
            gmm=gmm,
        ),
        original_heuristics=OriginalHeuristics(
            execution_profile="original",
            theta_mean_samples=100,
            inference_theta="raw",
            gaussian_mean_samples=100,
            empty_component_mean="original_uniform",
        ),
    )
    return inputs, datasets, config


def assert_rng_equal(actual, expected):
    actual, expected = actual.get_state(), expected.get_state()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def assert_state_equal(model, expected):
    np.testing.assert_array_equal(model.state.theta, expected["theta"])
    assert model.state.attributes == expected["frames"]
    assert model.state.vocabulary == tuple(
        str(i) for i in range(reference.SETTINGS["vocabulary"])
    )
    np.testing.assert_array_equal(model.state.words, expected["words"])
    for attr in reference.MODALITIES:
        if attr == "a":
            np.testing.assert_array_equal(
                model.state.z[attr], [row[0] for row in expected["z"][attr]]
            )
        else:
            for actual_row, expected_row in zip(
                model.state.z[attr], expected["z"][attr]
            ):
                np.testing.assert_array_equal(actual_row, expected_row)
        for production_name, reference_name in (
            ("pi", "pi"),
            ("means", "means"),
            ("covariances", "covariances"),
        ):
            np.testing.assert_array_equal(
                getattr(model.state, production_name)[attr],
                expected[reference_name][attr],
            )


@pytest.mark.parametrize("seed", [0, 2, 10])
def test_joint_initialization_and_complete_mh_iteration_match_independent_reference(
    tmp_path, seed
):
    inputs, datasets, config = inputs_and_config(tmp_path, seed)
    rng, oracle_rng = np.random.RandomState(seed), np.random.RandomState(seed)
    models = initialize_original_models(datasets, config, rng=rng, experiment_id=0)
    expected = reference.initialize(inputs, oracle_rng)
    assert models[0].rng is rng and models[1].rng is rng
    for model, state in zip(models, expected):
        assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)
    # Each fresh proposal, acceptance and listener update is a separate boundary.
    for listener, speaker in ((0, 1), (1, 0)):
        suggested = models[speaker].draw_words()
        oracle_suggested = reference.proposal(
            expected[speaker], inputs[speaker][1], oracle_rng
        )
        np.testing.assert_array_equal(suggested, oracle_suggested)
        assert_rng_equal(rng, oracle_rng)
        accepted = accept_words(models[listener], suggested, "mh")
        oracle_accepted = reference.accept(
            expected[listener], inputs[listener][1], oracle_suggested, oracle_rng
        )
        np.testing.assert_array_equal(accepted, oracle_accepted)
        assert_rng_equal(rng, oracle_rng)
        models[listener].step(accepted)
        expected[listener]["words"] = oracle_accepted
        reference.step(expected[listener], *inputs[listener], oracle_rng)
        for model, state in zip(models, expected):
            assert_state_equal(model, state)
        assert_rng_equal(rng, oracle_rng)


def test_initialization_distribution_counts_and_last_frame_uniform(tmp_path):
    inputs, datasets, config = inputs_and_config(tmp_path, 2)
    rng = RecordingRandomState(2)
    models = initialize_original_models(datasets, config, rng=rng)
    # Before KMeans, only z/F use uniform: every fourth F draw is still drawn
    # from [0,1), even though only one modality remains to be selected.
    index = 0
    for observations, _ in inputs:
        for attr in reference.MODALITIES:
            for document in observations[attr]:
                for _ in document:
                    assert rng.calls[index] == (
                        "uniform",
                        0,
                        reference.SETTINGS["categories"][attr],
                        None,
                    )
                    index += 1
        for _ in observations["a"]:
            for upper in (4, 3, 2, 1):
                assert rng.calls[index] == ("uniform", 0, upper, None)
                index += 1
    theta_calls = [call for call in rng.calls if call[0] == "dirichlet"]
    assert len(theta_calls) == 2 * sum(reference.SETTINGS["categories"].values())
    for _, parameters, size in theta_calls:
        np.testing.assert_array_equal(parameters, [0.3] * 5)
        assert size == 100
    assert [call for call in rng.calls if call[0] == "beta"] == [
        ("beta", 1, 1, count)
        for _ in range(2)
        for count in reference.SETTINGS["categories"].values()
    ]
    for model in models:
        for attr in reference.MODALITIES:
            for covariance in model.state.covariances[attr]:
                np.testing.assert_array_equal(
                    covariance, np.eye(reference.SETTINGS["dimensions"][attr]) * 0.05
                )


def test_stagewise_posteriors_use_all_objects_attention_and_actual_draw_counts(
    tmp_path,
):
    inputs, datasets, config = inputs_and_config(tmp_path, 10)
    rng, oracle_rng = RecordingRandomState(10), np.random.RandomState(10)
    (model, _), (state, _) = (
        initialize_original_models(datasets, config, rng=rng),
        reference.initialize(inputs, oracle_rng),
    )
    observations, attention = inputs[0]
    # Check independent z probabilities/assignments, followed by inherited pi
    # parameters, GIW priors, attended row maps and theta posterior counts.
    model._sample_z(model.state.theta)
    reference.sample_z(state, observations, attention, oracle_rng)
    assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)
    rng.calls.clear()
    expected_pi = reference.mixture_parameters(state)
    model._sample_pi()
    reference.sample_pi(state, oracle_rng)
    calls = [call for call in rng.calls if call[0] == "dirichlet"]
    assert len(calls) == 4
    for attr, (_, parameters, size) in zip(reference.MODALITIES, calls):
        np.testing.assert_array_equal(parameters, expected_pi[attr])
        assert size is None  # One draw; never a 100-draw mixture average.
    assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)
    model._sample_gaussians()
    reference.sample_gaussians(state, observations, oracle_rng)
    assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)
    model._sample_attributes(model.state.theta)
    reference.sample_frames(state, attention, oracle_rng)
    assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)
    expected_theta = reference.theta_parameters(state, attention)
    rng.calls.clear()
    model._sample_theta()
    reference.sample_theta(state, attention, oracle_rng)
    calls = [call for call in rng.calls if call[0] == "dirichlet"]
    assert len(calls) == sum(reference.SETTINGS["categories"].values())
    for (_, parameters, size), expected_parameters in zip(calls, expected_theta):
        np.testing.assert_array_equal(parameters, expected_parameters)
        assert size == 100
    assert_state_equal(model, state)
    assert_rng_equal(rng, oracle_rng)


def test_mh_consumes_uniform_even_when_every_acceptance_ratio_is_one(tmp_path):
    inputs, datasets, config = inputs_and_config(tmp_path, 0)
    rng = np.random.RandomState(0)
    model = initialize_original_models(datasets, config, rng=rng)[0]
    expected_rng = deepcopy(rng)
    accepted = accept_words(model, model.state.words, "mh")
    for document in model.state.words:
        for _ in document:
            expected_rng.rand()
    np.testing.assert_array_equal(accepted, model.state.words)
    assert_rng_equal(rng, expected_rng)


def test_learning_passes_raw_theta_while_proposals_and_mh_use_corrected_theta(
    tmp_path, monkeypatch
):
    inputs, datasets, config = inputs_and_config(tmp_path, 2)
    rng = np.random.RandomState(2)
    model = initialize_original_models(datasets, config, rng=rng)[0]
    state = reference.initialize(inputs, np.random.RandomState(2))[0]
    # Deliberately unbalanced columns make the choice of theta observable.
    theta = np.arange(1, 71, dtype=float).reshape(14, 5) ** 2
    theta[:, 0] *= np.arange(1, 15)
    theta /= theta.sum(axis=1, keepdims=True)
    model.state.theta = theta.copy()
    state["theta"] = theta.copy()
    corrected = reference.corrected(theta)
    assert not np.array_equal(theta, corrected)
    oracle_rng = deepcopy(rng)
    before = model.state.theta.copy()
    np.testing.assert_array_equal(model.word_theta(), corrected)
    np.testing.assert_array_equal(model.state.theta, before)
    wrong_rng = deepcopy(rng)
    suggested = model.draw_words()
    expected_suggested = reference.proposal(state, inputs[0][1], oracle_rng)
    np.testing.assert_array_equal(suggested, expected_suggested)
    assert not np.array_equal(
        suggested, reference.proposal(state, inputs[0][1], wrong_rng, rescale=False)
    )
    assert_rng_equal(rng, oracle_rng)
    # Select dissimilar proposed words to exercise unequal MH ratios.
    suggested = [
        np.array([(int(word) + token + 1) % 5 for token, word in enumerate(row)])
        for row in state["words"]
    ]
    wrong_rng = deepcopy(rng)
    accepted = accept_words(model, suggested, "mh")
    expected = reference.accept(state, inputs[0][1], suggested, oracle_rng)
    assert not np.array_equal(
        expected,
        reference.accept(state, inputs[0][1], suggested, wrong_rng, rescale=False),
    )
    np.testing.assert_array_equal(accepted, expected)
    assert_rng_equal(rng, oracle_rng)
    seen = []
    monkeypatch.setattr(
        model, "_sample_z", lambda supplied: seen.append(("z", supplied.copy()))
    )
    monkeypatch.setattr(
        model,
        "_sample_attributes",
        lambda supplied: seen.append(("F", supplied.copy())),
    )
    for method in ("_sample_pi", "_sample_gaussians", "_sample_theta"):
        monkeypatch.setattr(model, method, lambda: None)
    model.step(accepted)
    assert [name for name, _ in seen] == ["z", "F"]
    for _, supplied in seen:
        np.testing.assert_array_equal(supplied, theta)
