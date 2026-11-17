"""Execution-profile validation, raw reader, and workflow integration contracts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import random

import numpy as np
import pytest
import yaml

from legacy_csl.config import (
    AttributeCategoryCounts,
    Config,
    ConfigError,
    CoreHyperparameters,
    OriginalHeuristics,
    Paths,
    Run,
    WordModel,
    dump_resolved_config,
    load_config,
)
from legacy_csl.data import (
    ATTRIBUTES,
    LEGACY_TRUTH_DIRECTORY,
    dataset_input_files,
    load_dataset,
    load_datasets,
    load_single_words,
    write_compact_fixture,
)
from legacy_csl.runner import analyze, learn, predict, validate_data
from legacy_csl.state_io import load_state


def _config(tmp_path, mode="inter-csl", *, vocabulary_size=4, seed=0):
    return Config(
        mode=mode,
        run=Run("original", iterations=2, experiments=1, seed=seed),
        paths=Paths(tmp_path / "data", tmp_path / "out"),
        word_model=WordModel(theta_sampler="dirichlet"),
        hyperparameters=CoreHyperparameters(
            category_counts=AttributeCategoryCounts(2, 2, 2, 2),
            vocabulary_size=vocabulary_size,
        ),
        original_heuristics=OriginalHeuristics(
            theta_mean_samples=100,
            inference_theta="raw",
            gaussian_mean_samples=100,
            empty_component_mean="original_uniform",
            execution_profile="original",
        ),
    )


@pytest.mark.parametrize("value", ["unknown", True, False, 0, 1.0, None, [], {}])
def test_profile_rejects_non_enum_values_directly_and_in_yaml(tmp_path, value):
    with pytest.raises(ConfigError, match="original_heuristics.execution_profile"):
        OriginalHeuristics(execution_profile=value)
    path = tmp_path / "config.yaml"
    dump_resolved_config(_config(tmp_path), path)
    values = yaml.safe_load(path.read_text())
    values["original_heuristics"]["execution_profile"] = value
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ConfigError, match="original_heuristics.execution_profile"):
        load_config(path)


def test_profile_round_trip_and_default(tmp_path):
    path = tmp_path / "config.yaml"
    original = _config(tmp_path)
    dump_resolved_config(original, path)
    assert load_config(path) == original
    current = replace(
        original,
        original_heuristics=replace(
            original.original_heuristics, execution_profile="current"
        ),
    )
    dump_resolved_config(current, path)
    values = yaml.safe_load(path.read_text())
    del values["original_heuristics"]["execution_profile"]
    path.write_text(yaml.safe_dump(values))
    assert load_config(path) == current


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("word_model", "rescaling", "none"),
        ("word_model", "theta_sampler", "legacy_importance"),
        ("original_heuristics", "theta_mean_samples", 1),
        ("original_heuristics", "inference_theta", "rescaled"),
        ("original_heuristics", "gaussian_mean_samples", 1),
        ("original_heuristics", "empty_component_mean", "prior"),
        ("attribute_model", "exclusivity", "iid"),
        ("freeze", "gmm", True),
        ("freeze", "theta", True),
        ("freeze", "attributes", True),
        ("hyperparameters", "word_length", 3),
    ],
)
def test_original_required_settings_are_rejected_without_rewriting(
    tmp_path, section, field, value
):
    config = _config(tmp_path)
    with pytest.raises(ConfigError, match=f"{section}.{field}"):
        replace(
            config, **{section: replace(getattr(config, section), **{field: value})}
        )


@pytest.mark.parametrize("interaction", ["all_accept", "no_communication"])
def test_original_rejects_unsupported_communication(tmp_path, interaction):
    with pytest.raises(ConfigError, match="interaction"):
        replace(_config(tmp_path), interaction=interaction)


def test_original_requires_ten_kmeans_starts(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ConfigError, match="hyperparameters.gmm.kmeans_n_init"):
        replace(
            config,
            hyperparameters=replace(
                config.hyperparameters,
                gmm=replace(config.hyperparameters.gmm, kmeans_n_init=1),
            ),
        )


@pytest.mark.parametrize(("seed", "experiments"), [(-1, 1), (2**32, 1), (2**32 - 1, 2)])
def test_original_seed_range_checked_before_execution(tmp_path, seed, experiments):
    config = _config(tmp_path)
    with pytest.raises(ConfigError, match="run.seed"):
        replace(config, run=replace(config.run, seed=seed, experiments=experiments))
    assert (
        replace(config, run=replace(config.run, seed=2**32 - 1)).run.seed == 2**32 - 1
    )


def _legacy_inputs(root, *, agents=("A", "B"), documents=4, constant_b=False):
    for agent in agents:
        for doc_id in range(documents):
            doc = root / f"agent{agent}" / f"testss{doc_id + 1:03}"
            (doc / "image").mkdir(parents=True)
            (doc / "object_center.txt").write_text("2\n0\n1,0.1,0.2\n2,0.3,0.4\n")
            scale = (1 if constant_b else 2 + 2 * doc_id) if agent == "B" else doc_id
            (doc / "target_object.txt").write_text(
                f"1\n0,0,0\n{scale},{2 * scale},{3 * scale}\n0\n0.25\n"
            )
            lines = []
            for name, values in {
                "head": np.zeros(6),
                "left_arm": np.zeros(16),
                "right_arm": np.zeros(16),
                "torso": np.zeros(3),
                "tactile": np.arange(108),
            }.items():
                lines.extend([f"{name},{len(values)}", ",".join(map(str, values)), ""])
            (doc / "action.csv").write_text("\n".join(lines))
            for object_id in (1, 2):
                (doc / "image" / f"object_CNN_PCA_40_{object_id:02}.csv").write_text(
                    f"{object_id + doc_id / 10},{object_id / 2}"
                )
                (doc / "image" / f"object_RGB_BoF_40_{object_id:02}.csv").write_text(
                    f"{object_id + doc_id},2"
                )
        truth_root = root / f"agent{agent}" / LEGACY_TRUTH_DIRECTORY
        truth_root.mkdir()
        (truth_root / "za_C.csv").write_text(
            ",".join(str(i % 2) for i in range(documents))
        )
        for attr in ("p", "o", "c"):
            (truth_root / f"z{attr}_C.csv").write_text("0,1\n" * documents)


def test_original_inter_uses_raw_b_range_and_prefers_legacy(tmp_path):
    _legacy_inputs(tmp_path)
    normal_a = load_dataset(tmp_path, "A")
    normal_b = load_dataset(tmp_path, "B")
    original_a, original_b = load_datasets(
        tmp_path, ("A", "B"), execution_profile="original"
    )
    np.testing.assert_array_equal(normal_a.observations["a"][:, 0], np.arange(4) / 3)
    np.testing.assert_array_equal(
        original_a.observations["a"][:, 0], (np.arange(4) - 2) / 6
    )
    np.testing.assert_array_equal(
        original_b.observations["a"], normal_b.observations["a"]
    )
    np.testing.assert_array_equal(
        original_a.observations["a"][:, 3:], normal_a.observations["a"][:, 3:]
    )
    for attr in ("p", "o", "c"):
        for actual, expected in zip(
            original_a.observations[attr], normal_a.observations[attr]
        ):
            np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original_a.attentions, normal_a.attentions)
    for agent in ("A", "B"):
        (tmp_path / f"agent{agent}.npz").write_bytes(b"ignored invalid compact")
    again = load_datasets(tmp_path, ("A", "B"), execution_profile="original")
    np.testing.assert_array_equal(
        again[0].observations["a"], original_a.observations["a"]
    )
    assert all(
        path.suffix != ".npz"
        for path in dataset_input_files(tmp_path, "A", execution_profile="original")
    )


def test_original_inter_rejects_compact_and_undefined_normalization(tmp_path):
    write_compact_fixture(tmp_path / "compact")
    with pytest.raises(ValueError, match="raw legacy.*compact NPZ"):
        load_datasets(tmp_path / "compact", ("A", "B"), execution_profile="original")
    _legacy_inputs(tmp_path / "constant", constant_b=True)
    with pytest.raises(ValueError, match="constant columns"):
        load_datasets(tmp_path / "constant", ("A", "B"), execution_profile="original")


def test_single_original_keeps_corpus_and_rejects_mismatched_vocabulary(tmp_path):
    config = _config(tmp_path, "single-csl")
    write_compact_fixture(config.paths.data_root, num_documents=4)
    (config.paths.data_root / "single_words.json").write_text(
        json.dumps([["a", "b", "c", "d"]] * 4)
    )
    validate_data(config)
    # Custom vocabulary is valid when it matches the actual observed corpus.
    assert len(load_single_words(config.paths.data_root, 4).vocabulary) == 4
    with pytest.raises(ValueError, match="effective observed corpus vocabulary"):
        validate_data(
            replace(
                config,
                hyperparameters=replace(config.hyperparameters, vocabulary_size=14),
            )
        )
    np_state, python_state = np.random.get_state(), random.getstate()
    [directory] = learn(config)
    after = np.random.get_state()
    assert np_state[0] == after[0] and np_state[2:] == after[2:]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert random.getstate() == python_state
    state = load_state(directory, "U")
    for row in state.words:
        np.testing.assert_array_equal(row, [0, 1, 2, 3])
    assert state.vocabulary == ("a", "b", "c", "d")
    assert analyze(config)
    provenance = json.loads((directory / "execution_provenance.json").read_text())
    assert "SINGLE original source is unavailable" in provenance["guarantee_scope"]
    assert provenance["inputs"][0]["format"] == "compact_npz"
    assert len((directory / "rng_boundaries.jsonl").read_text().splitlines()) == 3
    assert "numpy" in provenance["versions"] and "legacy-csl" in provenance["versions"]
    for item in provenance["inputs"]:
        for entry in item.get("files", [item]):
            assert (
                hashlib.sha256(
                    (config.paths.data_root / entry["path"]).read_bytes()
                ).hexdigest()
                == entry["sha256"]
            )


def test_original_inter_native_workflows_and_additive_trial_seed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _legacy_inputs(config.paths.data_root)
    config = replace(config, run=replace(config.run, experiments=2, iterations=1))
    directories = learn(config)
    assert len(directories) == 2
    assert analyze(config)
    datasets = []
    from legacy_csl.model import CslModel

    sampler = CslModel.sample_categories_from_observations

    def record(model):
        datasets.append(model.dataset)
        return sampler(model)

    monkeypatch.setattr(CslModel, "sample_categories_from_observations", record)
    assert predict(config)
    np.testing.assert_array_equal(
        datasets[0].observations["a"][:, 0], (np.arange(4) - 2) / 6
    )
    for i, directory in enumerate(directories):
        provenance = json.loads((directory / "execution_provenance.json").read_text())
        assert provenance["effective_seed"] == i
        assert all(item["format"] == "legacy_datadump" for item in provenance["inputs"])
        assert [
            record["boundary"]
            for record in map(
                json.loads,
                (directory / "rng_boundaries.jsonl").read_text().splitlines(),
            )
        ] == ["initialized", "after_A", "after_B"]
    second = load_state(directories[1], "A")
    # Running a later trial independently must produce the same trajectory.
    separate = replace(
        config, run=replace(config.run, name="separate", experiments=1, seed=1)
    )
    [directory] = learn(separate)
    state = load_state(directory, "A")
    np.testing.assert_array_equal(state.theta, second.theta)
    for attr in ATTRIBUTES:
        np.testing.assert_array_equal(state.means[attr], second.means[attr])
        np.testing.assert_array_equal(state.pi[attr], second.pi[attr])
    first_rng = (directories[1] / "rng_boundaries.jsonl").read_text()
    assert first_rng == (directory / "rng_boundaries.jsonl").read_text()


def test_single_original_legacy_keeps_tactile_fourteen_words_and_concentration(
    tmp_path,
):
    config = _config(tmp_path, "single-csl", vocabulary_size=14)
    _legacy_inputs(config.paths.data_root, agents=("U",))
    tokens = list(range(14)) + [0, 1]
    (config.paths.data_root / "single_words.json").write_text(
        json.dumps([tokens[index : index + 4] for index in range(0, 16, 4)])
    )
    current = load_dataset(config.paths.data_root, "U")
    [original] = load_datasets(
        config.paths.data_root, ("U",), execution_profile="original"
    )
    assert np.asarray(original.observations["a"]).shape == (4, 38)
    np.testing.assert_array_equal(original.observations["a"], current.observations["a"])
    np.testing.assert_array_equal(
        original.observations["a"][0, 29:],
        [np.mean(np.arange(index, index + 12)) / 255.0 for index in range(0, 108, 12)],
    )
    assert set(
        config.hyperparameters.gmm.mixture_concentration.as_mapping().values()
    ) == {1.0}
    [directory] = learn(config)
    state = load_state(directory, "U")
    corpus = load_single_words(config.paths.data_root, 4)
    assert len(state.vocabulary) == 14
    for observed, saved in zip(corpus.encoded, state.words):
        np.testing.assert_array_equal(saved, observed)
    assert state.means["a"].shape == (2, 38)
    assert not (directory / "B.npz").exists()
