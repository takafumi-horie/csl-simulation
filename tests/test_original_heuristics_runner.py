"""Saved workflows with the optional original learning heuristics enabled."""
from dataclasses import replace

import numpy as np
import pytest

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
    load_config,
)
from legacy_csl.data import write_compact_fixture
from legacy_csl.runner import analyze, learn, predict
from legacy_csl.state_io import load_state


@pytest.mark.parametrize("mode", ["single-csl", "inter-csl"])
def test_original_heuristics_saved_workflows(tmp_path, mode):
    config = Config(
        mode=mode,
        run=Run(name=mode, iterations=3, experiments=1, seed=21),
        paths=Paths(data_root=tmp_path / "data", output_root=tmp_path / "outputs"),
        word_model=WordModel(theta_sampler="dirichlet"),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=13,
            gmm=GmmHyperparameters(
                mixture_concentration=AttributeScalars(.1, .1, .1, .1)
            ),
        ),
        original_heuristics=OriginalHeuristics(
            theta_mean_samples=100,
            inference_theta="raw",
            gaussian_mean_samples=100,
            empty_component_mean="original_uniform",
        ),
    )
    write_compact_fixture(config.paths.data_root, num_documents=6)
    [directory] = learn(config)
    resolved = load_config(directory.parent / "resolved_config.yaml")
    assert resolved.original_heuristics == config.original_heuristics
    agents = ("U",) if mode == "single-csl" else ("A", "B")
    checkpoints = {}
    for agent in agents:
        state = load_state(directory, agent)
        assert np.isfinite(state.theta).all()
        np.testing.assert_allclose(state.theta.sum(axis=1), 1)
        assert len(state.words) == 6
        for attr in ("a", "p", "o", "c"):
            assert np.isfinite(state.means[attr]).all()
            assert np.isfinite(state.covariances[attr]).all()
        checkpoints[agent] = (directory / f"state_{agent}.npz").read_bytes()
        for filename in (f"theta_{agent}.csv", f"Mu_p_{agent}.csv"):
            assert (directory / "csv" / filename).read_bytes() == (
                directory / "iterations" / "iter0003" / "csv" / filename
            ).read_bytes()

    [analysis] = analyze(resolved)
    metric = "ari_truth_a" if mode == "single-csl" else "ari_agents_a"
    assert np.isfinite(analysis[metric])
    assert (directory / "analysis" / "learning_curves.png").is_file()

    if mode == "inter-csl":
        enabled_rows = predict(resolved)
        prediction_path = directory / "prediction" / "prediction.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            enabled = {name: archive[name].copy() for name in archive.files}
        # Learning-only switches must not change prediction from a saved state.
        disabled_rows = predict(
            replace(resolved, original_heuristics=OriginalHeuristics())
        )
        assert disabled_rows == enabled_rows
        with np.load(prediction_path, allow_pickle=False) as archive:
            for name, expected in enabled.items():
                np.testing.assert_array_equal(archive[name], expected)
    else:
        with pytest.raises(ValueError, match="only for inter-csl"):
            predict(resolved)

    for agent, checkpoint in checkpoints.items():
        assert (directory / f"state_{agent}.npz").read_bytes() == checkpoint
