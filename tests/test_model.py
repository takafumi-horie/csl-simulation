from __future__ import annotations

from dataclasses import replace

import numpy as np

from legacy_csl.config import (
    AttributeCategoryCounts,
    AttributeModel,
    Config,
    CoreHyperparameters,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import load_dataset, write_compact_fixture
from legacy_csl.model import CslModel
from legacy_csl.rescaling import legacy_theta_sum_rescaling


def _config(tmp_path, *, exclusivity="distinct", rescaling="legacy_theta_sum"):
    sampler = "legacy_importance" if rescaling == "legacy_theta_sum" else "dirichlet"
    return Config(
        mode="uni-csl",
        run=Run(name="test", iterations=1, experiments=1, seed=3),
        paths=Paths(data_root=tmp_path / "data", output_root=tmp_path / "out"),
        word_model=WordModel(
            rescaling=rescaling,
            theta_sampler=sampler,
            importance_candidates=8,
        ),
        attribute_model=AttributeModel(exclusivity=exclusivity),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=8,
            word_length=4,
            gamma=0.2,
        ),
    )


def test_effective_theta_is_single_legacy_transformation(tmp_path):
    config = _config(tmp_path)
    write_compact_fixture(config.paths.data_root, num_documents=6)
    model = CslModel(
        load_dataset(config.paths.data_root, "U"),
        config,
        rng=np.random.default_rng(10),
    )

    np.testing.assert_allclose(
        model.effective_theta(), legacy_theta_sum_rescaling(model.state.theta)
    )


def test_distinct_step_preserves_unique_attribute_columns(tmp_path):
    config = _config(tmp_path)
    write_compact_fixture(config.paths.data_root, num_documents=6)
    model = CslModel(
        load_dataset(config.paths.data_root, "U"),
        config,
        rng=np.random.default_rng(11),
    )

    model.step()

    assert all(len(sequence) == len(set(sequence)) for sequence in model.state.attributes)


def test_iid_mode_allows_word_length_larger_than_attribute_count(tmp_path):
    config = replace(
        _config(tmp_path, exclusivity="iid"),
        hyperparameters=replace(
            _config(tmp_path).hyperparameters,
            word_length=6,
        ),
    )
    write_compact_fixture(
        config.paths.data_root, num_documents=5, word_length=6
    )
    model = CslModel(
        load_dataset(config.paths.data_root, "U"),
        config,
        rng=np.random.default_rng(12),
    )

    assert model.state.words.shape == (5, 6)
    assert all(len(sequence) == 6 for sequence in model.state.attributes)


def test_readonly_z_does_not_expose_mutable_state(tmp_path):
    config = _config(tmp_path)
    write_compact_fixture(config.paths.data_root, num_documents=4)
    model = CslModel(
        load_dataset(config.paths.data_root, "U"),
        config,
        rng=np.random.default_rng(13),
    )
    view = model.readonly_z()

    view["a"][0] = 999

    assert model.state.z["a"][0] != 999

