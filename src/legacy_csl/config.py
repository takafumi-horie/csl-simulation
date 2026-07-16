"""Strict configuration models for the standalone legacy CSL package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast

import yaml


Mode: TypeAlias = Literal["single-csl", "inter-csl"]
Rescaling: TypeAlias = Literal[
    "legacy_theta_sum", "empirical_unigram", "none"
]
ThetaSampler: TypeAlias = Literal["legacy_importance", "dirichlet"]
AttributeExclusivity: TypeAlias = Literal["distinct", "iid"]
Interaction: TypeAlias = Literal["mh", "no_communication", "all_accept"]


class ConfigError(ValueError):
    """Report an invalid configuration value or shape.

    設定値または設定の構造が不正であることを示す。
    """


@dataclass(frozen=True, slots=True)
class Run:
    """Configure repeated inference runs.

    反復推論の実行を設定する。

    Args:
        name (str): Name used for the output directory.
            出力ディレクトリに使う名前。
        iterations (int): Number of inference iterations per experiment.
            各実験の推論反復数。
        experiments (int): Number of independent experiments.
            独立に実行する実験数。
        seed (int): Base random seed.
            基準とする乱数シード。
    """

    name: str
    iterations: int = 200
    experiments: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate run values after construction.

        構築後に実行設定値を検証する。

        Raises:
            ConfigError: If a run value is invalid.
                実行設定値が不正な場合。
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("run.name must be a non-empty string")
        _require_positive_int(self.iterations, "run.iterations")
        _require_positive_int(self.experiments, "run.experiments")
        _require_int(self.seed, "run.seed")


@dataclass(frozen=True, slots=True)
class Paths:
    """Store input and output locations.

    入力と出力の場所を保持する。

    Args:
        data_root (Path): Root of the processed data dump.
            処理済みデータダンプのルート。
        output_root (Path): Root under which run outputs are written.
            実行結果を書き込むルート。
    """

    data_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        """Validate path value types after construction.

        構築後にパス値の型を検証する。

        Raises:
            ConfigError: If either value is not a path.
                いずれかの値がパスでない場合。
        """
        if not isinstance(self.data_root, Path):
            raise ConfigError("paths.data_root must be a pathlib.Path")
        if not isinstance(self.output_root, Path):
            raise ConfigError("paths.output_root must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class WordModel:
    """Configure the categorical word model.

    カテゴリカル単語モデルを設定する。

    Args:
        rescaling (Rescaling): Word-probability rescaling method.
            単語確率の補正方法。
        theta_sampler (ThetaSampler): Theta update method.
            theta の更新方法。
        importance_candidates (int): Candidate count for importance resampling.
            importance resampling の候補数。
    """

    rescaling: Rescaling = "legacy_theta_sum"
    theta_sampler: ThetaSampler = "legacy_importance"
    importance_candidates: int = 10_000

    def __post_init__(self) -> None:
        """Validate compatible word-model options after construction.

        構築後に単語モデル設定の整合性を検証する。

        Raises:
            ConfigError: If an option is invalid or incompatible.
                オプションが不正または整合しない場合。
        """
        _require_choice(
            self.rescaling,
            {"legacy_theta_sum", "empirical_unigram", "none"},
            "word_model.rescaling",
        )
        _require_choice(
            self.theta_sampler,
            {"legacy_importance", "dirichlet"},
            "word_model.theta_sampler",
        )
        _require_positive_int(
            self.importance_candidates, "word_model.importance_candidates"
        )
        if (
            self.theta_sampler == "legacy_importance"
            and self.rescaling != "legacy_theta_sum"
        ):
            raise ConfigError(
                "word_model.theta_sampler='legacy_importance' requires "
                "word_model.rescaling='legacy_theta_sum'"
            )


@dataclass(frozen=True, slots=True)
class AttributeModel:
    """Configure attribute-sequence sampling.

    属性列のサンプリングを設定する。

    Args:
        exclusivity (AttributeExclusivity): ``distinct`` applies simple MEC2;
            ``iid`` samples uniformly with replacement.
            ``distinct`` は単純 MEC2、``iid`` は重複ありの一様
            サンプリング。
    """

    exclusivity: AttributeExclusivity = "distinct"

    def __post_init__(self) -> None:
        """Validate the attribute sampling mode after construction.

        構築後に属性サンプリング方式を検証する。

        Raises:
            ConfigError: If the mode is unsupported.
                サポート外の方式の場合。
        """
        _require_choice(
            self.exclusivity, {"distinct", "iid"}, "attribute_model.exclusivity"
        )


@dataclass(frozen=True, slots=True)
class Freeze:
    """Select model components that inference must not update.

    推論時に更新しないモデル要素を選択する。

    Args:
        gmm (bool): Freeze Gaussian-mixture parameters and assignments.
            ガウス混合のパラメータと割当てを固定する。
        theta (bool): Freeze category-to-word distributions.
            カテゴリから単語への分布を固定する。
        attributes (bool): Freeze attribute sequences.
            属性列を固定する。
    """

    gmm: bool = False
    theta: bool = False
    attributes: bool = False

    def __post_init__(self) -> None:
        """Validate freeze flags after construction.

        構築後に固定フラグを検証する。

        Raises:
            ConfigError: If a flag is not boolean.
                フラグが真偽値でない場合。
        """
        _require_bool(self.gmm, "freeze.gmm")
        _require_bool(self.theta, "freeze.theta")
        _require_bool(self.attributes, "freeze.attributes")


@dataclass(frozen=True, slots=True)
class AttributeCategoryCounts:
    """Store the category count for each sensory attribute.

    各感覚属性のカテゴリ数を保持する。

    Args:
        a (int): Action/haptic category count.
            動作・触覚属性のカテゴリ数。
        p (int): Position category count.
            位置属性のカテゴリ数。
        o (int): Object category count.
            物体属性のカテゴリ数。
        c (int): Color category count.
            色属性のカテゴリ数。
    """

    a: int = 10
    p: int = 10
    o: int = 10
    c: int = 10

    def __post_init__(self) -> None:
        """Validate all category counts after construction.

        構築後に全カテゴリ数を検証する。

        Raises:
            ConfigError: If a count is not a positive integer.
                カテゴリ数が正の整数でない場合。
        """
        for attribute in ("a", "p", "o", "c"):
            _require_positive_int(
                getattr(self, attribute),
                f"hyperparameters.category_counts.{attribute}",
            )

    def as_mapping(self) -> Mapping[str, int]:
        """Return category counts keyed by attribute.

        属性名をキーとするカテゴリ数を返す。

        Returns:
            Mapping[str, int]: A newly allocated category-count mapping.
                新しく作成したカテゴリ数のマッピング。
        """
        return {"a": self.a, "p": self.p, "o": self.o, "c": self.c}


@dataclass(frozen=True, slots=True)
class AttributeScalars:
    """Store one floating-point hyperparameter per sensory attribute.

    感覚属性ごとに1つの浮動小数点ハイパーパラメータを保持する。

    Args:
        a: Action/haptic value.
            動作・触覚属性の値。
        p: Position value.
            位置属性の値。
        o: Object value.
            物体属性の値。
        c: Color value.
            色属性の値。
    """

    a: float
    p: float
    o: float
    c: float

    def as_mapping(self) -> Mapping[str, float]:
        """Return values keyed by attribute.

        属性名をキーとする値を返す。

        Returns:
            Newly allocated attribute mapping.
            新しく作成した属性マッピング。
        """
        return {"a": self.a, "p": self.p, "o": self.o, "c": self.c}


@dataclass(frozen=True, slots=True)
class GmmHyperparameters:
    """Configure the legacy Gaussian--inverse-Wishart mixture.

    legacy Gaussian--inverse-Wishart 混合モデルを設定する。

    Scalar mean priors and covariance scales are expanded to every dimension.
    These defaults reproduce the values built by the original ``config_hyper``.

    平均事前と共分散スケールのスカラー値は全次元へ展開する。既定値は旧
    ``config_hyper`` が構築していた値と一致する。
    """

    mixture_concentration: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(1.0, 1.0, 1.0, 1.0)
    )
    frozen_mixture_concentration: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(0.1, 0.1, 0.1, 0.1)
    )
    mean_prior: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(0.0, 0.0, 0.0, 0.0)
    )
    mean_precision: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(0.001, 0.001, 0.001, 0.001)
    )
    covariance_scale: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(0.01, 0.01, 0.01, 0.01)
    )
    covariance_df_offset: AttributeScalars = field(
        default_factory=lambda: AttributeScalars(2.0, 2.0, 2.0, 2.0)
    )
    kmeans_n_init: int = 10

    def __post_init__(self) -> None:
        """Validate all Gaussian mixture hyperparameters.

        全GMMハイパーパラメータを検証する。

        Raises:
            ConfigError: If a value is outside the valid domain.
                値が有効範囲外の場合。
        """
        for field_name in (
            "mixture_concentration",
            "frozen_mixture_concentration",
            "mean_precision",
            "covariance_scale",
            "covariance_df_offset",
        ):
            values = getattr(self, field_name).as_mapping()
            for attribute, value in values.items():
                _require_positive_float(
                    value, f"hyperparameters.gmm.{field_name}.{attribute}"
                )
        for attribute, value in self.mean_prior.as_mapping().items():
            _require_finite_float(
                value, f"hyperparameters.gmm.mean_prior.{attribute}"
            )
        _require_positive_int(self.kmeans_n_init, "hyperparameters.gmm.kmeans_n_init")


@dataclass(frozen=True, slots=True)
class CoreHyperparameters:
    """Store the minimal hyperparameters used by categorical legacy CSL.

    categorical legacy CSL で使う必要最小限の
    ハイパーパラメータを保持する。

    Args:
        category_counts (AttributeCategoryCounts): Category counts by attribute.
            属性ごとのカテゴリ数。
        vocabulary_size (int): Number of categorical word types.
            カテゴリカル単語の語彙数。
        word_length (int): Number of words/attributes in each utterance.
            各発話に含む単語・属性の数。
        gamma (float): Symmetric Dirichlet concentration for theta.
            theta の対称 Dirichlet 集中度。
    """

    category_counts: AttributeCategoryCounts = field(
        default_factory=AttributeCategoryCounts
    )
    vocabulary_size: int = 14
    word_length: int = 4
    gamma: float = 0.1
    gmm: GmmHyperparameters = field(default_factory=GmmHyperparameters)

    def __post_init__(self) -> None:
        """Validate core hyperparameters after construction.

        構築後に主要ハイパーパラメータを検証する。

        Raises:
            ConfigError: If any hyperparameter is outside its domain.
                ハイパーパラメータが定義域外の場合。
        """
        _require_positive_int(
            self.vocabulary_size, "hyperparameters.vocabulary_size"
        )
        _require_positive_int(self.word_length, "hyperparameters.word_length")
        _require_positive_float(self.gamma, "hyperparameters.gamma")


@dataclass(frozen=True, slots=True)
class Config:
    """Represent a fully validated standalone CSL configuration.

    検証済みのスタンドアロン CSL 設定を表す。

    Args:
        mode (Mode): ``single-csl`` or ``inter-csl``.
            ``single-csl`` または ``inter-csl``。
        run (Run): Repeated-run settings.
            反復実行の設定。
        paths (Paths): Input and output roots.
            入出力のルート。
        word_model (WordModel): Categorical word settings.
            カテゴリカル単語の設定。
        attribute_model (AttributeModel): Attribute-sequence settings.
            属性列の設定。
        interaction (Interaction): Inter-agent acceptance behavior.
            エージェント間の受理動作。
        freeze (Freeze): Model components held fixed during inference.
            推論中に固定するモデル要素。
        hyperparameters (CoreHyperparameters): Minimal model hyperparameters.
            モデルの必要最小限のハイパーパラメータ。
    """

    mode: Mode
    run: Run
    paths: Paths
    word_model: WordModel = field(default_factory=WordModel)
    attribute_model: AttributeModel = field(default_factory=AttributeModel)
    interaction: Interaction = "mh"
    freeze: Freeze = field(default_factory=Freeze)
    hyperparameters: CoreHyperparameters = field(default_factory=CoreHyperparameters)

    def __post_init__(self) -> None:
        """Validate top-level options and cross-section invariants.

        トップレベルの選択肢とセクション間の
        不変条件を検証する。

        Raises:
            ConfigError: If an option or cross-section combination is invalid.
                選択肢またはセクション間の組み合わせが
                不正な場合。
        """
        _require_choice(self.mode, {"single-csl", "inter-csl"}, "mode")
        _require_choice(
            self.interaction,
            {"mh", "no_communication", "all_accept"},
            "interaction",
        )
        if (
            self.attribute_model.exclusivity == "distinct"
            and self.hyperparameters.word_length > 4
        ):
            raise ConfigError(
                "hyperparameters.word_length must be at most 4 when "
                "attribute_model.exclusivity='distinct'"
            )


def load_config(path: str | Path) -> Config:
    """Load and strictly validate a YAML configuration.

    YAML 設定を読み込み、厳密に検証する。

    Relative paths are resolved relative to the YAML file, so commands behave
    consistently from any working directory.
    相対パスは YAML ファイル基準で解決し、任意の作業
    ディレクトリから同じ動作にする。

    Args:
        path (str | Path): YAML configuration path.
            YAML 設定ファイルのパス。

    Returns:
        Config: A fully resolved and validated configuration.
            解決・検証済みの設定。

    Raises:
        ConfigError: If YAML structure, keys, or values are invalid.
            YAML の構造、キー、または値が不正な場合。
        OSError: If the configuration cannot be read.
            設定ファイルを読み込めない場合。
    """
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    root = _mapping(raw, "configuration")
    _reject_unknown(
        root,
        {
            "mode",
            "run",
            "paths",
            "word_model",
            "attribute_model",
            "interaction",
            "freeze",
            "hyperparameters",
        },
        "configuration",
    )
    for required in ("mode", "run", "paths"):
        _require_key(root, required, "configuration")

    run = _parse_run(root["run"])
    paths = _parse_paths(root["paths"], config_path.parent)
    word_model = _parse_word_model(root.get("word_model", {}))
    attribute_model = _parse_attribute_model(root.get("attribute_model", {}))
    freeze = _parse_freeze(root.get("freeze", {}))
    hyperparameters = _parse_hyperparameters(root.get("hyperparameters", {}))
    return Config(
        mode=cast(Mode, _string(root["mode"], "mode")),
        run=run,
        paths=paths,
        word_model=word_model,
        attribute_model=attribute_model,
        interaction=cast(
            Interaction, _string(root.get("interaction", "mh"), "interaction")
        ),
        freeze=freeze,
        hyperparameters=hyperparameters,
    )


def dump_resolved_config(config: Config, path: str | Path) -> None:
    """Write all resolved configuration values as portable YAML.

    デフォルトを含む解決済み設定値を移植可能な YAML で
    書き込む。

    Args:
        config (Config): Validated configuration to serialize.
            書き込む検証済み設定。
        path (str | Path): Destination YAML path.
            出力先 YAML パス。

    Raises:
        OSError: If the destination cannot be written.
            出力先に書き込めない場合。
    """
    destination = Path(path)
    values = asdict(config)
    values["paths"] = {
        "data_root": str(config.paths.data_root),
        "output_root": str(config.paths.output_root),
    }
    destination.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _parse_run(value: Any) -> Run:
    values = _mapping(value, "run")
    _reject_unknown(values, {"name", "iterations", "experiments", "seed"}, "run")
    _require_key(values, "name", "run")
    return Run(
        name=_string(values["name"], "run.name"),
        iterations=values.get("iterations", 200),
        experiments=values.get("experiments", 1),
        seed=values.get("seed", 0),
    )


def _parse_paths(value: Any, base: Path) -> Paths:
    values = _mapping(value, "paths")
    _reject_unknown(values, {"data_root", "output_root"}, "paths")
    _require_key(values, "data_root", "paths")
    _require_key(values, "output_root", "paths")
    return Paths(
        data_root=_resolve_path(values["data_root"], base, "paths.data_root"),
        output_root=_resolve_path(values["output_root"], base, "paths.output_root"),
    )


def _parse_word_model(value: Any) -> WordModel:
    values = _mapping(value, "word_model")
    _reject_unknown(
        values,
        {"rescaling", "theta_sampler", "importance_candidates"},
        "word_model",
    )
    return WordModel(
        rescaling=cast(
            Rescaling,
            _string(
                values.get("rescaling", "legacy_theta_sum"),
                "word_model.rescaling",
            ),
        ),
        theta_sampler=cast(
            ThetaSampler,
            _string(
                values.get("theta_sampler", "legacy_importance"),
                "word_model.theta_sampler",
            ),
        ),
        importance_candidates=values.get("importance_candidates", 10_000),
    )


def _parse_attribute_model(value: Any) -> AttributeModel:
    values = _mapping(value, "attribute_model")
    _reject_unknown(values, {"exclusivity"}, "attribute_model")
    return AttributeModel(
        exclusivity=cast(
            AttributeExclusivity,
            _string(
                values.get("exclusivity", "distinct"),
                "attribute_model.exclusivity",
            ),
        )
    )


def _parse_freeze(value: Any) -> Freeze:
    values = _mapping(value, "freeze")
    _reject_unknown(values, {"gmm", "theta", "attributes"}, "freeze")
    return Freeze(
        gmm=values.get("gmm", False),
        theta=values.get("theta", False),
        attributes=values.get("attributes", False),
    )


def _parse_hyperparameters(value: Any) -> CoreHyperparameters:
    values = _mapping(value, "hyperparameters")
    _reject_unknown(
        values,
        {"category_counts", "vocabulary_size", "word_length", "gamma", "gmm"},
        "hyperparameters",
    )
    counts_value = _mapping(
        values.get("category_counts", {}), "hyperparameters.category_counts"
    )
    _reject_unknown(
        counts_value, {"a", "p", "o", "c"}, "hyperparameters.category_counts"
    )
    counts = AttributeCategoryCounts(
        a=counts_value.get("a", 10),
        p=counts_value.get("p", 10),
        o=counts_value.get("o", 10),
        c=counts_value.get("c", 10),
    )
    gmm = _parse_gmm(values.get("gmm", {}))
    return CoreHyperparameters(
        category_counts=counts,
        vocabulary_size=values.get("vocabulary_size", 14),
        word_length=values.get("word_length", 4),
        gamma=values.get("gamma", 0.1),
        gmm=gmm,
    )


def _parse_gmm(value: Any) -> GmmHyperparameters:
    values = _mapping(value, "hyperparameters.gmm")
    fields = {
        "mixture_concentration",
        "frozen_mixture_concentration",
        "mean_prior",
        "mean_precision",
        "covariance_scale",
        "covariance_df_offset",
        "kmeans_n_init",
    }
    _reject_unknown(values, fields, "hyperparameters.gmm")
    defaults = GmmHyperparameters()
    return GmmHyperparameters(
        mixture_concentration=_parse_attribute_scalars(
            values.get("mixture_concentration", {}),
            "hyperparameters.gmm.mixture_concentration",
            defaults.mixture_concentration,
        ),
        frozen_mixture_concentration=_parse_attribute_scalars(
            values.get("frozen_mixture_concentration", {}),
            "hyperparameters.gmm.frozen_mixture_concentration",
            defaults.frozen_mixture_concentration,
        ),
        mean_prior=_parse_attribute_scalars(
            values.get("mean_prior", {}),
            "hyperparameters.gmm.mean_prior",
            defaults.mean_prior,
        ),
        mean_precision=_parse_attribute_scalars(
            values.get("mean_precision", {}),
            "hyperparameters.gmm.mean_precision",
            defaults.mean_precision,
        ),
        covariance_scale=_parse_attribute_scalars(
            values.get("covariance_scale", {}),
            "hyperparameters.gmm.covariance_scale",
            defaults.covariance_scale,
        ),
        covariance_df_offset=_parse_attribute_scalars(
            values.get("covariance_df_offset", {}),
            "hyperparameters.gmm.covariance_df_offset",
            defaults.covariance_df_offset,
        ),
        kmeans_n_init=values.get("kmeans_n_init", defaults.kmeans_n_init),
    )


def _parse_attribute_scalars(
    value: Any, field_name: str, defaults: AttributeScalars
) -> AttributeScalars:
    values = _mapping(value, field_name)
    _reject_unknown(values, {"a", "p", "o", "c"}, field_name)
    return AttributeScalars(
        a=values.get("a", defaults.a),
        p=values.get("p", defaults.p),
        o=values.get("o", defaults.o),
        c=values.get("c", defaults.c),
    )


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{field_name} keys must be strings")
    return value


def _reject_unknown(
    values: Mapping[str, Any], allowed: set[str], field_name: str
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {field_name}: {', '.join(repr(key) for key in unknown)}"
        )


def _require_key(values: Mapping[str, Any], key: str, field_name: str) -> None:
    if key not in values:
        raise ConfigError(f"missing required key: {field_name}.{key}")


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    return value


def _resolve_path(value: Any, base: Path, field_name: str) -> Path:
    text = _string(value, field_name)
    if not text.strip():
        raise ConfigError(f"{field_name} must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_choice(value: Any, choices: set[str], field_name: str) -> None:
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(sorted(choices))
        raise ConfigError(f"{field_name} must be one of: {expected}")


def _require_int(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise ConfigError(f"{field_name} must be an integer")


def _require_positive_int(value: Any, field_name: str) -> None:
    _require_int(value, field_name)
    if value <= 0:
        raise ConfigError(f"{field_name} must be positive")


def _require_positive_float(value: Any, field_name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{field_name} must be a finite positive number")


def _require_finite_float(value: Any, field_name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{field_name} must be a finite number")


def _require_bool(value: Any, field_name: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{field_name} must be a boolean")
