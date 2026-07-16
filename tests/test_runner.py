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
from legacy_csl.runner import analyze, learn, predict, validate_data
from legacy_csl.state_io import load_state


def _state_csv_names(agent: str) -> set[str]:
    attributes = ("a", "p", "o", "c")
    names = {
        f"{prefix}_{attr}_{agent}.csv"
        for attr in attributes
        for prefix in ("pi", "Mu", "Sig")
    }
    names.update(f"z{attr}_{agent}.csv" for attr in attributes)
    names.update(
        {
            f"theta_{agent}.csv",
            f"uttered_words_{agent}.csv",
            f"F_{agent}.csv",
            f"W_list_{agent}.csv",
            f"word_length_{agent}.csv",
        }
    )
    return names


def _directory_names(directory) -> set[str]:
    return {path.name for path in directory.iterdir() if path.is_file()}


def _config(tmp_path, mode):
    return Config(
        mode=mode,
        run=Run(name=mode, iterations=2, experiments=1, seed=21),
        paths=Paths(data_root=tmp_path / "data", output_root=tmp_path / "outputs"),
        word_model=WordModel(importance_candidates=8),
        attribute_model=AttributeModel(exclusivity="distinct"),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(a=2, p=2, o=2, c=2),
            vocabulary_size=8,
            word_length=4,
            gamma=0.2,
        ),
    )


def test_single_end_to_end(tmp_path, monkeypatch):
    config = _config(tmp_path, "single-csl")
    write_compact_fixture(config.paths.data_root, num_documents=6)
    progress_calls = []

    def record_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return iterable

    monkeypatch.setattr("legacy_csl.runner.tqdm", record_progress)

    validate_data(config)
    [directory] = learn(config)
    rows = analyze(config)
    state = load_state(directory, "U")

    assert len(state.words) == 6
    assert all(word_d.shape == (4,) for word_d in state.words)
    expected_csvs = _state_csv_names("U")
    final_csv_directory = directory / "csv"
    assert _directory_names(final_csv_directory) == expected_csvs
    for iteration in (1, 2):
        iteration_csv_directory = (
            directory / "iterations" / f"iter{iteration:04d}" / "csv"
        )
        assert _directory_names(iteration_csv_directory) == expected_csvs
    assert not (directory / "iterations" / "iter0000").exists()
    final_iteration_csv = directory / "iterations" / "iter0002" / "csv"
    for filename in expected_csvs:
        assert (final_iteration_csv / filename).read_bytes() == (
            final_csv_directory / filename
        ).read_bytes()
    assert (directory / "analysis" / "learning_curves.png").is_file()
    assert "ari_truth_a" in rows[0]
    assert progress_calls == [
        {"desc": "Experiments", "unit": "trial", "position": 0},
        {
            "desc": "exp000 (1/1) iterations",
            "unit": "iteration",
            "position": 1,
            "leave": False,
        },
    ]


def test_inter_learning_prediction_and_analysis(tmp_path, monkeypatch):
    config = _config(tmp_path, "inter-csl")
    write_compact_fixture(config.paths.data_root, num_documents=6)

    [directory] = learn(config)
    observation_sampler = CslModel.sample_categories_from_observations
    sampled_datasets = []

    def record_observation_sampling(model):
        sampled_datasets.append(model.dataset)
        return observation_sampler(model)

    def reject_prior_sampling(model, pi):
        del model, pi
        raise AssertionError("prediction must condition speaker z on observations")

    monkeypatch.setattr(
        CslModel,
        "sample_categories_from_observations",
        record_observation_sampling,
    )
    monkeypatch.setattr(CslModel, "_sample_z_prior", reject_prior_sampling)
    prediction_rows = predict(config)
    analysis_rows = analyze(config)

    assert len(sampled_datasets) == 1
    assert (directory / "state_A.npz").is_file()
    assert (directory / "state_B.npz").is_file()
    expected_csvs = _state_csv_names("A") | _state_csv_names("B")
    final_csv_directory = directory / "csv"
    assert _directory_names(final_csv_directory) == expected_csvs
    for iteration in (1, 2):
        iteration_directory = directory / "iterations" / f"iter{iteration:04d}"
        iteration_csv_directory = iteration_directory / "csv"
        assert _directory_names(iteration_csv_directory) == expected_csvs
        assert not list(iteration_directory.glob("state_*.npz"))
        assert not list(iteration_directory.glob("metadata_*.json"))
    final_iteration_csv = directory / "iterations" / "iter0002" / "csv"
    for filename in expected_csvs:
        assert (final_iteration_csv / filename).read_bytes() == (
            final_csv_directory / filename
        ).read_bytes()
    assert (directory / "analysis" / "category_word_heatmap_A.png").is_file()
    assert (directory / "analysis" / "category_word_heatmap_B.png").is_file()
    assert (
        directory / "analysis" / "ground_truth_category_word_heatmap_A.png"
    ).is_file()
    assert (
        directory / "analysis" / "ground_truth_category_word_heatmap_B.png"
    ).is_file()
    assert (directory / "prediction" / "prediction.npz").is_file()
    with np.load(directory / "prediction" / "prediction.npz") as prediction:
        assert {
            "speaker_z_a",
            "speaker_z_p",
            "speaker_z_o",
            "speaker_z_c",
        } <= set(prediction.files)
    assert "mse_a" in prediction_rows[0]
    assert "ari_agents_a" in analysis_rows[0]
    assert "ari_truth_A_a" in analysis_rows[0]
    assert "ari_truth_B_a" in analysis_rows[0]


def test_validate_data_checks_each_legacy_agents_truth_layout(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, "inter-csl")
    write_compact_fixture(config.paths.data_root, num_documents=6)
    dataset = load_dataset(config.paths.data_root, "A")
    legacy_root = tmp_path / "legacy"
    config = replace(
        config,
        paths=replace(config.paths, data_root=legacy_root),
    )
    checked_agents = []

    monkeypatch.setattr(
        "legacy_csl.runner.validate_legacy_truth_layout",
        lambda data_root, agent: checked_agents.append((data_root, agent)),
    )
    monkeypatch.setattr("legacy_csl.runner.load_dataset", lambda root, agent: dataset)

    validate_data(config)

    assert checked_agents == [(legacy_root, "A"), (legacy_root, "B")]
