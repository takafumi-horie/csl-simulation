from __future__ import annotations

from legacy_csl.config import (
    AttributeCategoryCounts,
    AttributeModel,
    Config,
    CoreHyperparameters,
    Paths,
    Run,
    WordModel,
)
from legacy_csl.data import write_compact_fixture
from legacy_csl.runner import analyze, learn, predict, validate_data
from legacy_csl.state_io import load_state


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
    assert (directory / "csv" / "theta_U.csv").is_file()
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


def test_inter_learning_prediction_and_analysis(tmp_path):
    config = _config(tmp_path, "inter-csl")
    write_compact_fixture(config.paths.data_root, num_documents=6)

    [directory] = learn(config)
    prediction_rows = predict(config)
    analysis_rows = analyze(config)

    assert (directory / "state_A.npz").is_file()
    assert (directory / "state_B.npz").is_file()
    assert (directory / "analysis" / "category_word_heatmap_A.png").is_file()
    assert (directory / "analysis" / "category_word_heatmap_B.png").is_file()
    assert (directory / "prediction" / "prediction.npz").is_file()
    assert "mse_a" in prediction_rows[0]
    assert "ari_agents_a" in analysis_rows[0]
    assert "ari_truth_A_a" in analysis_rows[0]
    assert "ari_truth_B_a" in analysis_rows[0]
