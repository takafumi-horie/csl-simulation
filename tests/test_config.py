from pathlib import Path

import pytest
import yaml

from legacy_csl.config import ConfigError, dump_resolved_config, load_config


def _valid_config() -> dict:
    return {
        "mode": "inter-csl",
        "run": {"name": "test"},
        "paths": {"data_root": "data", "output_root": "outputs"},
    }


def _write_config(tmp_path: Path, values: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_loads_defaults_and_resolves_paths_relative_to_yaml(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, _valid_config()))

    assert config.mode == "inter-csl"
    assert config.run.iterations == 200
    assert config.word_model.importance_candidates == 10_000
    assert config.attribute_model.exclusivity == "distinct"
    assert config.interaction == "mh"
    assert config.hyperparameters.category_counts.as_mapping() == {
        "a": 10,
        "p": 10,
        "o": 10,
        "c": 10,
    }
    assert config.paths.data_root == (tmp_path / "data").resolve()
    assert config.paths.output_root == (tmp_path / "outputs").resolve()


def test_rejects_unknown_keys_at_every_level(tmp_path: Path) -> None:
    values = _valid_config()
    values["word_model"] = {"rescaling": "none", "typo": True}

    with pytest.raises(ConfigError, match=r"unknown key.*typo"):
        load_config(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "mode", "multi-csl", "mode must be one of"),
        (None, "interaction", "always_reject", "interaction must be one of"),
        (
            "word_model",
            "rescaling",
            "legacy",
            "word_model.rescaling must be one of",
        ),
        (
            "word_model",
            "theta_sampler",
            "importance",
            "word_model.theta_sampler must be one of",
        ),
        (
            "attribute_model",
            "exclusivity",
            "uniform",
            "attribute_model.exclusivity must be one of",
        ),
    ],
)
def test_rejects_unsupported_options(
    tmp_path: Path,
    section: str | None,
    key: str,
    value: str,
    message: str,
) -> None:
    values = _valid_config()
    target = values if section is None else values.setdefault(section, {})
    target[key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("rescaling", "theta_sampler"),
    [("none", "legacy_importance"), ("empirical_unigram", "legacy_importance")],
)
def test_legacy_importance_requires_legacy_rescaling(
    tmp_path: Path, rescaling: str, theta_sampler: str
) -> None:
    values = _valid_config()
    values["word_model"] = {
        "rescaling": rescaling,
        "theta_sampler": theta_sampler,
    }

    with pytest.raises(ConfigError, match="legacy_theta_sum"):
        load_config(_write_config(tmp_path, values))


def test_iid_attribute_mode_allows_more_than_four_words(tmp_path: Path) -> None:
    values = _valid_config()
    values["attribute_model"] = {"exclusivity": "iid"}
    values["hyperparameters"] = {"word_length": 8}

    config = load_config(_write_config(tmp_path, values))

    assert config.attribute_model.exclusivity == "iid"
    assert config.hyperparameters.word_length == 8


def test_distinct_attribute_mode_rejects_more_than_four_words(
    tmp_path: Path,
) -> None:
    values = _valid_config()
    values["attribute_model"] = {"exclusivity": "distinct"}
    values["hyperparameters"] = {"word_length": 5}

    with pytest.raises(ConfigError, match="at most 4"):
        load_config(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("run", "iterations"), True, "must be an integer"),
        (("freeze", "theta"), 1, "must be a boolean"),
        (("hyperparameters", "gamma"), float("inf"), "finite positive"),
        (("hyperparameters", "vocabulary_size"), 0, "must be positive"),
    ],
)
def test_rejects_invalid_scalar_values(
    tmp_path: Path, path: tuple[str, str], value: object, message: str
) -> None:
    values = _valid_config()
    section, key = path
    values.setdefault(section, {})[key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, values))


def test_dump_writes_resolved_defaults_and_round_trips(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, _valid_config()))
    destination = tmp_path / "resolved.yaml"

    dump_resolved_config(config, destination)
    dumped = yaml.safe_load(destination.read_text(encoding="utf-8"))
    reloaded = load_config(destination)

    assert dumped["word_model"]["rescaling"] == "legacy_theta_sum"
    assert dumped["attribute_model"]["exclusivity"] == "distinct"
    assert dumped["interaction"] == "mh"
    assert reloaded == config


def test_gmm_defaults_match_legacy_config_hyper(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, _valid_config()))
    gmm = config.hyperparameters.gmm

    assert set(gmm.mixture_concentration.as_mapping().values()) == {1.0}
    assert set(gmm.frozen_mixture_concentration.as_mapping().values()) == {0.1}
    assert set(gmm.mean_prior.as_mapping().values()) == {0.0}
    assert set(gmm.mean_precision.as_mapping().values()) == {0.001}
    assert set(gmm.covariance_scale.as_mapping().values()) == {0.01}
    assert set(gmm.covariance_df_offset.as_mapping().values()) == {2.0}
    assert gmm.kmeans_n_init == 10


def test_gmm_accepts_attribute_specific_values(tmp_path: Path) -> None:
    values = _valid_config()
    values["hyperparameters"] = {
        "gmm": {
            "mixture_concentration": {"a": 2.0, "o": 0.5},
            "mean_prior": {"p": -1.5},
            "mean_precision": {"c": 0.2},
            "covariance_scale": {"o": 3.0},
            "covariance_df_offset": {"p": 4.0},
            "kmeans_n_init": 4,
        }
    }

    gmm = load_config(_write_config(tmp_path, values)).hyperparameters.gmm

    assert gmm.mixture_concentration.a == 2.0
    assert gmm.mixture_concentration.p == 1.0
    assert gmm.mixture_concentration.o == 0.5
    assert gmm.mean_prior.p == -1.5
    assert gmm.mean_precision.c == 0.2
    assert gmm.covariance_scale.o == 3.0
    assert gmm.covariance_df_offset.p == 4.0
    assert gmm.kmeans_n_init == 4


def test_single_accepts_interaction_field_for_a_shared_schema(tmp_path: Path) -> None:
    values = _valid_config()
    values["mode"] = "single-csl"
    values["interaction"] = "no_communication"

    config = load_config(_write_config(tmp_path, values))

    assert config.interaction == "no_communication"


@pytest.mark.parametrize("filename", ["single.yaml", "inter.yaml"])
def test_shipped_examples_are_valid(filename: str) -> None:
    example = Path(__file__).parents[1] / "examples" / filename

    config = load_config(example)

    assert config.mode in {"single-csl", "inter-csl"}
