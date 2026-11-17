"""Categorical CSL inference with optional original averaging heuristics.

categorical CSL 推論と選択式の原典平均化ヒューリスティクス。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import invwishart, multivariate_normal
from sklearn.cluster import KMeans

from .attributes import AttributeSampler
from .config import Config
from .data import ATTRIBUTES, ObservationDataset
from .giw import GiwPrior
from .rescaling import legacy_theta_sum_rescaling, sample_theta

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
CategoryAssignments: TypeAlias = IntArray | tuple[IntArray, ...]


@dataclass(slots=True)
class CslState:
    """Own every inferred random variable for one categorical CSL agent.

    categorical CSL エージェント1体の全推論確率変数を保持する。

    ``z['a']`` is a ``(D,)`` array. The remaining assignments are ragged tuples
    with one ``(M_d,)`` array per document, matching all observed objects.

    ``z['a']`` は ``(D,)`` 配列で、他属性は全観測物体に対応する文書別
    ``(M_d,)`` 配列の ragged tuple である。
    """

    z: dict[str, CategoryAssignments]
    pi: dict[str, FloatArray]
    means: dict[str, FloatArray]
    covariances: dict[str, FloatArray]
    theta: FloatArray
    attributes: list[tuple[str, ...]]
    words: tuple[IntArray, ...]
    vocabulary: tuple[str, ...]

    @property
    def variances(self) -> Mapping[str, FloatArray]:
        """Return copied full covariances under the former public name.

        以前の公開名でコピーした full covariance を返す。

        Returns:
            Read-only outer mapping of copied covariances.
            コピー済み共分散の読取専用外側mapping。
        """
        return MappingProxyType(
            {key: value.copy() for key, value in self.covariances.items()}
        )

    def copy(self) -> CslState:
        """Return an independent deep copy.

        独立した深いコピーを返す。

        Returns:
            Copied state.
            コピー済み状態。
        """
        return CslState(
            z={key: _copy_assignments(value) for key, value in self.z.items()},
            pi={key: value.copy() for key, value in self.pi.items()},
            means={key: value.copy() for key, value in self.means.items()},
            covariances={
                key: value.copy() for key, value in self.covariances.items()
            },
            theta=self.theta.copy(),
            attributes=list(self.attributes),
            words=tuple(value.copy() for value in self.words),
            vocabulary=tuple(self.vocabulary),
        )


class CslModel:
    """Run categorical CSL updates, optionally using original heuristics.

    categorical CSL の更新を実行し、任意で原典ヒューリスティクスを適用する。

    The categorical scope infers every variable inferred by the original code:
    all-object ``z``, ``pi``, full-covariance GIW ``(mu, Sigma)``, ``theta``,
    and attribute columns ``F``. Fixed observed words are not inferred in
    Single-CSL; Inter-CSL updates them through the outer MH naming game.

    categorical 対象では旧コードが推論した全変数、すなわち全物体 ``z``、
    ``pi``、full-covariance GIW ``(mu, Sigma)``、``theta``、属性列 ``F`` を
    推論する。Single-CSL の観測単語は固定で、Inter-CSL では外側のMH naming
    gameが単語を更新する。
    """

    def __init__(
        self,
        dataset: ObservationDataset,
        config: Config,
        *,
        rng: np.random.Generator,
        initial_words: Sequence[npt.ArrayLike] | npt.ArrayLike | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        """Initialize parameters with the same distributions as legacy CSL.

        legacy CSL と同じ分布でパラメータを初期化する。

        Args:
            dataset: Full multi-object observations.
                全物体を含む観測。
            config: Validated experiment configuration.
                検証済み実験設定。
            rng: Explicit random-number generator.
                明示的乱数生成器。
            initial_words: Optional fixed/ragged observed words.
                任意の固定・ragged観測単語。
            vocabulary: Vocabulary in theta-column order.
                theta列順の語彙。

        Raises:
            ValueError: If dimensions, vocabulary, or frozen truth are invalid.
                次元、語彙、固定用正解が不正な場合。
        """
        if getattr(config.original_heuristics, "execution_profile", "current") == "original":
            raise ValueError(
                "original execution requires joint initialize_original_models(); "
                "the standalone CslModel constructor cannot preserve shared RNG order"
            )
        self.dataset = dataset
        self.config = config
        self.rng = rng
        self.category_counts = dict(
            config.hyperparameters.category_counts.as_mapping()
        )
        self.offsets = _category_offsets(self.category_counts)
        self.num_categories = sum(self.category_counts.values())
        self.attribute_sampler = AttributeSampler(
            ATTRIBUTES, config.attribute_model.exclusivity, rng
        )
        words = self._coerce_words(initial_words)
        if vocabulary is None:
            vocabulary = tuple(
                str(index) for index in range(config.hyperparameters.vocabulary_size)
            )
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must be non-empty and unique")
        if words is not None and any(
            np.any(word_d < 0) or np.any(word_d >= len(self.vocabulary))
            for word_d in words
        ):
            raise ValueError("word indexes must be within the supplied vocabulary")
        lengths = (
            [len(word_d) for word_d in words]
            if words is not None
            else [config.hyperparameters.word_length] * dataset.num_documents
        )
        attributes = [
            self.attribute_sampler.sample_prior(length) for length in lengths
        ]
        pi = self._sample_pi_prior()
        means, covariances = self._initialize_gaussians()
        z = self._sample_z_prior(pi)
        theta = np.asarray(
            [
                rng.dirichlet(
                    np.full(len(self.vocabulary), config.hyperparameters.gamma)
                )
                for _ in range(self.num_categories)
            ],
            dtype=np.float64,
        )
        if words is None:
            words = tuple(np.zeros(length, dtype=np.int64) for length in lengths)
        self.state = CslState(
            z, pi, means, covariances, theta, attributes, words, self.vocabulary
        )
        if initial_words is None:
            drawn = self.draw_words()
            self.state.words = tuple(word_d.copy() for word_d in drawn)
        if config.freeze.gmm:
            self._initialize_frozen_gmm()
        self._validate_state(self.state)

    @classmethod
    def from_state(
        cls,
        dataset: ObservationDataset,
        config: Config,
        state: CslState,
        *,
        rng: np.random.Generator,
    ) -> CslModel:
        """Rebuild a model without resampling a loaded checkpoint.

        読み込んだチェックポイントを再サンプルせずモデルを再構築する。

        Args:
            dataset: Full observations.
                全観測。
            config: Experiment configuration.
                実験設定。
            state: Loaded state.
                読み込んだ状態。
            rng: Explicit generator for subsequent transitions.
                以後の遷移に使う明示的generator。

        Returns:
            Reconstructed model.
            再構築済みモデル。
        """
        model = cls.__new__(cls)
        model.dataset = dataset
        model.config = config
        model.rng = rng
        model.category_counts = dict(
            config.hyperparameters.category_counts.as_mapping()
        )
        model.offsets = _category_offsets(model.category_counts)
        model.num_categories = sum(model.category_counts.values())
        model.attribute_sampler = AttributeSampler(
            ATTRIBUTES, config.attribute_model.exclusivity, rng
        )
        model.vocabulary = tuple(state.vocabulary)
        model.state = state.copy()
        model._validate_state(model.state)
        return model

    def word_theta(self) -> FloatArray:
        """Return theta used by word generation, legacy MH, and prediction.

        単語生成・legacy MH・予測で使うthetaを返す。学習z/Fの選択は別途行う。

        Returns:
            Independent category-by-word probability matrix.
            独立したカテゴリ×単語確率行列。
        """
        if self.config.word_model.rescaling == "legacy_theta_sum":
            return legacy_theta_sum_rescaling(self.state.theta)
        return self.state.theta.copy()

    def _inference_theta(self) -> FloatArray:
        """Select theta for learning z/F only. 学習z/F専用のthetaを選ぶ。"""
        if self.config.original_heuristics.inference_theta == "raw":
            return self.state.theta.copy()
        return self.word_theta()

    def effective_theta(
        self, words: Sequence[npt.ArrayLike] | npt.ArrayLike | None = None
    ) -> FloatArray:
        """Compatibility alias for :meth:`word_theta`.

        :meth:`word_theta` の互換alias。

        Args:
            words: Ignored; empirical correction belongs only to Inter MH.
                未使用。経験的補正はInter MH専用である。

        Returns:
            Word-consumer theta.
            単語consumer用theta。
        """
        del words
        return self.word_theta()

    def step(self, words: Sequence[npt.ArrayLike] | npt.ArrayLike | None = None) -> None:
        """Perform one learning sweep with the configured sampling rules.

        指定したサンプリング規則で学習sweepを1回実行する。

        Args:
            words: Optional accepted/fixed word corpus for this sweep.
                このsweepで使う任意の受理済み・固定単語コーパス。
        """
        if self.config.original_heuristics.execution_profile == "original":
            raise ValueError(
                "original learning requires joint initialize_original_models; "
                "from_state reconstructs prediction models, not a learning resume"
            )
        if words is not None:
            coerced = self._coerce_words(words)
            assert coerced is not None
            self._validate_word_shapes(coerced)
            self.state.words = coerced
        theta = self._inference_theta()
        if not self.config.freeze.gmm:
            self._sample_z(theta)
            self._sample_pi()
            self._sample_gaussians()
        # Original theta averaging includes the F-before-theta scan. With one
        # sample, preserve the existing sampler-dependent scans exactly.
        if (
            self.config.word_model.theta_sampler == "legacy_importance"
            or self.config.original_heuristics.theta_mean_samples > 1
        ):
            if not self.config.freeze.attributes:
                self._sample_attributes(theta)
            if not self.config.freeze.theta:
                self._sample_theta()
            return
        if not self.config.freeze.theta:
            self._sample_theta()
        if not self.config.freeze.attributes:
            self._sample_attributes(self._inference_theta())

    def draw_words(self) -> tuple[IntArray, ...]:
        """Draw a ragged word corpus from the current latent state.

        現在の潜在状態からragged単語コーパスを生成する。

        Returns:
            One integer word array per document.
            文書ごとの整数単語配列。
        """
        theta = self.word_theta()
        result: list[IntArray] = []
        for doc_id, attributes in enumerate(self.state.attributes):
            word_d = np.empty(len(attributes), dtype=np.int64)
            for word_id, attribute in enumerate(attributes):
                category = self.global_category(
                    attribute, self.attended_category(attribute, doc_id)
                )
                word_d[word_id] = self.rng.choice(len(self.vocabulary), p=theta[category])
            result.append(word_d)
        return tuple(result)

    def observation_category_logweights(
        self,
        attribute: str,
        observation: npt.ArrayLike,
    ) -> FloatArray:
        """Calculate ``log pi + log Gaussian`` for one speaker observation.

        話者の1観測に対する ``log pi + log Gaussian`` を計算する。

        Args:
            attribute: Sensory attribute of the observation.
                観測の感覚属性。
            observation: One feature vector.
                1つの特徴ベクトル。

        Returns:
            One posterior log-weight per attribute-local category.
            属性内カテゴリごとの事後対数重み。

        Raises:
            KeyError: If the attribute is unknown.
                属性が未知の場合。
            ValueError: If the observation dimension is invalid.
                観測次元が不正な場合。
        """
        if attribute not in self.category_counts:
            raise KeyError(attribute)
        value = np.asarray(observation, dtype=np.float64)
        expected_shape = (self.state.means[attribute].shape[1],)
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(
                f"observation for {attribute!r} must have shape {expected_shape}"
            )
        return np.log(self.state.pi[attribute] + 1e-300) + np.asarray(
            [
                multivariate_normal.logpdf(
                    value,
                    mean=self.state.means[attribute][category],
                    cov=self.state.covariances[attribute][category],
                    allow_singular=False,
                )
                for category in range(self.category_counts[attribute])
            ],
            dtype=np.float64,
        )

    def sample_categories_from_observations(
        self,
    ) -> dict[str, CategoryAssignments]:
        """Infer speaker categories from every observed action and object.

        全行動・全物体の観測から話者カテゴリを推論する。

        The learned ``pi``, means, and covariances are held fixed. Categories
        are sampled from ``pi_k * N(x | mu_k, Sigma_k)`` without a word
        likelihood. Object attributes retain all ragged object assignments;
        attention is applied later only when generating words.

        学習済み ``pi``、平均、共分散を固定し、単語尤度を使わず
        ``pi_k * N(x | mu_k, Sigma_k)`` からサンプルする。物体属性は全物体の
        ragged割当てを保持し、attentionは後の単語生成時にのみ適用する。

        Returns:
            Action ``(D,)`` and ragged all-object assignments for ``p/o/c``.
            Actionの ``(D,)`` と ``p/o/c`` の全物体ragged割当て。
        """
        sampled: dict[str, CategoryAssignments] = {}
        for attr in ATTRIBUTES:
            documents = tuple(
                np.asarray(
                    [
                        _sample_logweights(
                            self.observation_category_logweights(attr, observation),
                            self.rng,
                        )
                        for observation in observations_d
                    ],
                    dtype=np.int64,
                )
                for observations_d in self.dataset.document_arrays(attr)
            )
            if attr == "a":
                sampled[attr] = np.asarray(
                    [categories[0] for categories in documents], dtype=np.int64
                )
            else:
                sampled[attr] = documents
        return sampled

    def sample_categories_without_observations(
        self, words: Sequence[npt.ArrayLike]
    ) -> dict[str, IntArray]:
        """Sample attended predictive categories using only words and pi.

        単語とpiのみから注視対象の予測カテゴリをサンプルする。

        Args:
            words: Received word corpus. Prediction uses canonical F=(a,p,o,c).
                受信単語コーパス。予測ではcanonical F=(a,p,o,c)を使う。

        Returns:
            Sampled ``(D,)`` categories by attribute.
            属性別のサンプル済み ``(D,)`` カテゴリ。

        Raises:
            ValueError: If an utterance is not of canonical length four.
                発話長がcanonicalな4でない場合。
        """
        word_corpus = self._coerce_words(words)
        assert word_corpus is not None
        if any(len(word_d) != len(ATTRIBUTES) for word_d in word_corpus):
            raise ValueError("legacy prediction requires four words per utterance")
        theta = self.word_theta()
        predicted = {
            attr: np.empty(self.dataset.num_documents, dtype=np.int64)
            for attr in ATTRIBUTES
        }
        for doc_id, word_d in enumerate(word_corpus):
            for word_id, attr in enumerate(ATTRIBUTES):
                logweights = np.log(self.state.pi[attr] + 1e-300)
                offset = self.offsets[attr]
                category_theta = theta[
                    offset : offset + self.category_counts[attr], word_d[word_id]
                ]
                logweights += np.log(category_theta + 1e-300)
                predicted[attr][doc_id] = _sample_logweights(logweights, self.rng)
        return predicted

    def sample_observations(
        self, categories: Mapping[str, IntArray]
    ) -> dict[str, FloatArray]:
        """Sample predictive observations from full-covariance Gaussians.

        full-covariance Gaussianから予測観測をサンプルする。

        Args:
            categories: Predictive category vector for each attribute.
                属性ごとの予測カテゴリベクトル。

        Returns:
            Sampled ``(D, dimension)`` observations by attribute.
            属性別のサンプル済み ``(D, dimension)`` 観測。
        """
        return {
            attr: np.asarray(
                [
                    self.rng.multivariate_normal(
                        self.state.means[attr][category],
                        self.state.covariances[attr][category],
                    )
                    for category in categories[attr]
                ],
                dtype=np.float64,
            )
            for attr in ATTRIBUTES
        }

    def attended_category(self, attribute: str, doc_id: int) -> int:
        """Return the category that conditions words in one document.

        1文書の単語を条件付けるカテゴリを返す。

        Args:
            attribute: Sensory attribute.
                感覚属性。
            doc_id: Document index.
                文書番号。

        Returns:
            Attribute-local category.
            属性内カテゴリ。
        """
        assignments = self.state.z[attribute]
        if attribute == "a":
            assert isinstance(assignments, np.ndarray)
            return int(assignments[doc_id])
        assert isinstance(assignments, tuple)
        return int(assignments[doc_id][self.dataset.attentions[doc_id]])

    def attended_z(self) -> Mapping[str, IntArray]:
        """Return copied attended categories behind a read-only mapping.

        コピーした注視対象カテゴリを読み取り専用mappingで返す。

        Returns:
            Read-only attended category mapping.
            読み取り専用の注視対象カテゴリmapping。
        """
        return MappingProxyType(
            {
                attr: np.asarray(
                    [
                        self.attended_category(attr, doc_id)
                        for doc_id in range(self.dataset.num_documents)
                    ],
                    dtype=np.int64,
                )
                for attr in ATTRIBUTES
            }
        )

    def z_logweights(
        self,
        attribute: str,
        doc_id: int,
        object_id: int,
        theta: FloatArray | None = None,
        *,
        include_observation: bool = True,
    ) -> FloatArray:
        """Calculate the exact legacy conditional log-weights for one ``z``.

        1つの ``z`` に対する正確なlegacy条件付き対数重みを計算する。

        Args:
            attribute: Sensory attribute.
                感覚属性。
            doc_id: Document index.
                文書番号。
            object_id: Object index; action always uses zero.
                物体番号。Actionは常に0。
            theta: Optional word-consumer theta. If omitted, uses
                ``word_theta()`` regardless of ``inference_theta``. Learning
                explicitly supplies the configured raw/rescaled theta.
                任意の単語consumer用theta。省略時は ``inference_theta`` に
                関わらず ``word_theta()`` を使う。学習時には設定に応じた
                補正前・補正後thetaを明示的に渡す。
            include_observation: Include the Gaussian likelihood.
                Gaussian尤度を含めるか。

        Returns:
            One log-weight per attribute-local category.
            属性内カテゴリごとの対数重み。
        """
        if theta is None:
            theta = self.word_theta()
        observations_d = self.dataset.document_arrays(attribute)[doc_id]
        if object_id < 0 or object_id >= len(observations_d):
            raise IndexError(object_id)
        logweights = np.log(self.state.pi[attribute] + 1e-300)
        if include_observation:
            observation = observations_d[object_id]
            logweights = self.observation_category_logweights(
                attribute, observation
            )
        attended = attribute == "a" or object_id == self.dataset.attentions[doc_id]
        if attended:
            selected_words = [
                word
                for word, assigned_attr in zip(
                    self.state.words[doc_id], self.state.attributes[doc_id]
                )
                if assigned_attr == attribute
            ]
            if selected_words:
                offset = self.offsets[attribute]
                for category in range(self.category_counts[attribute]):
                    logweights[category] += np.log(
                        theta[offset + category, selected_words] + 1e-300
                    ).sum()
        return logweights

    def pi_posterior_parameters(self, attribute: str) -> FloatArray:
        """Return legacy Dirichlet posterior parameters for ``pi``.

        ``pi`` のlegacy Dirichlet事後パラメータを返す。

        Args:
            attribute: Sensory attribute.
                感覚属性。

        Returns:
            Prior concentration plus all-object category counts.
            事前集中度と全物体カテゴリ頻度の和。
        """
        concentration = (
            self.config.hyperparameters.gmm.mixture_concentration.as_mapping()[
                attribute
            ]
        )
        counts = np.bincount(
            _flatten_assignments(self.state.z[attribute]),
            minlength=self.category_counts[attribute],
        )
        return counts.astype(np.float64) + concentration

    def theta_posterior_parameters(self) -> FloatArray:
        """Return category-by-word Dirichlet posterior parameters.

        カテゴリ×単語Dirichlet事後パラメータを返す。

        Returns:
            ``gamma`` plus attended ``(z,F,w)`` counts.
            ``gamma`` と注視対象 ``(z,F,w)`` 頻度の和。
        """
        alpha = np.full(
            (self.num_categories, len(self.vocabulary)),
            self.config.hyperparameters.gamma,
            dtype=np.float64,
        )
        for doc_id, attributes in enumerate(self.state.attributes):
            for word, attr in zip(self.state.words[doc_id], attributes):
                category = self.global_category(
                    attr, self.attended_category(attr, doc_id)
                )
                alpha[category, word] += 1.0
        return alpha

    def readonly_z(self) -> Mapping[str, CategoryAssignments]:
        """Return deep-copied all-object assignments.

        深くコピーした全物体割当てを返す。

        Returns:
            Read-only outer mapping of copied assignments.
            コピー済み割当ての読み取り専用外側mapping。
        """
        return MappingProxyType(
            {key: _copy_assignments(value) for key, value in self.state.z.items()}
        )

    def global_category(self, attribute: str, local_category: int) -> int:
        """Convert an attribute-local category to a theta row.

        属性内カテゴリをtheta行へ変換する。

        Args:
            attribute: Attribute name.
                属性名。
            local_category: Attribute-local category.
                属性内カテゴリ。

        Returns:
            Global theta row.
            theta全体行。
        """
        return self.offsets[attribute] + int(local_category)

    def _sample_pi_prior(self) -> dict[str, FloatArray]:
        concentrations = self.config.hyperparameters.gmm.mixture_concentration.as_mapping()
        return {
            attr: self.rng.dirichlet(
                np.full(self.category_counts[attr], concentrations[attr])
            )
            for attr in ATTRIBUTES
        }

    def _initialize_gaussians(self) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
        gmm = self.config.hyperparameters.gmm
        scale_values = gmm.covariance_scale.as_mapping()
        means: dict[str, FloatArray] = {}
        covariances: dict[str, FloatArray] = {}
        for attr in ATTRIBUTES:
            observations = self.dataset.flatten(attr)
            if len(observations) < self.category_counts[attr]:
                raise ValueError(
                    f"attribute {attr!r} has fewer observations than categories"
                )
            random_state = int(self.rng.integers(0, np.iinfo(np.int32).max))
            means[attr] = KMeans(
                n_clusters=self.category_counts[attr],
                init="k-means++",
                n_init=gmm.kmeans_n_init,
                random_state=random_state,
            ).fit(observations).cluster_centers_.astype(np.float64)
            covariance = np.eye(observations.shape[1]) * scale_values[attr]
            covariances[attr] = np.repeat(
                covariance[np.newaxis, :, :], self.category_counts[attr], axis=0
            )
        return means, covariances

    def _sample_z_prior(
        self, pi: Mapping[str, FloatArray]
    ) -> dict[str, CategoryAssignments]:
        z: dict[str, CategoryAssignments] = {
            "a": np.asarray(
                [
                    self.rng.choice(len(pi["a"]), p=pi["a"])
                    for _ in range(self.dataset.num_documents)
                ],
                dtype=np.int64,
            )
        }
        for attr in ATTRIBUTES[1:]:
            z[attr] = tuple(
                np.asarray(
                    [
                        self.rng.choice(len(pi[attr]), p=pi[attr])
                        for _ in range(len(observations_d))
                    ],
                    dtype=np.int64,
                )
                for observations_d in self.dataset.observations[attr]
            )
        return z

    def _sample_z(self, theta: FloatArray) -> None:
        for attr in ATTRIBUTES:
            assignments = self.state.z[attr]
            observation_docs = self.dataset.document_arrays(attr)
            for doc_id, observations_d in enumerate(observation_docs):
                for object_id, observation in enumerate(observations_d):
                    del observation
                    logweights = self.z_logweights(
                        attr, doc_id, object_id, theta
                    )
                    sampled = _sample_logweights(logweights, self.rng)
                    if attr == "a":
                        assert isinstance(assignments, np.ndarray)
                        assignments[doc_id] = sampled
                    else:
                        assert isinstance(assignments, tuple)
                        assignments[doc_id][object_id] = sampled

    def _sample_pi(self) -> None:
        for attr in ATTRIBUTES:
            self.state.pi[attr] = self.rng.dirichlet(
                self.pi_posterior_parameters(attr)
            )

    def _sample_gaussians(self, *, use_heuristics: bool = True) -> None:
        heuristics = self.config.original_heuristics
        for attr in ATTRIBUTES:
            observations = self.dataset.flatten(attr)
            categories = _flatten_assignments(self.state.z[attr])
            prior = self._giw_prior(attr, observations.shape[1])
            means = np.empty_like(self.state.means[attr])
            covariances = np.empty_like(self.state.covariances[attr])
            for category in range(self.category_counts[attr]):
                selected = observations[categories == category]
                dimension = observations.shape[1]
                if (
                    use_heuristics
                    and len(selected) == 0
                    and heuristics.empty_component_mean == "original_uniform"
                ):
                    # The original empty-component heuristic draws mu before
                    # Sigma, with a different range for position.
                    low, high = (-0.5, 0.5) if attr == "p" else (0.0, 1.0)
                    means[category] = self.rng.uniform(low, high, size=dimension)
                    covariances[category] = np.asarray(
                        invwishart.rvs(
                            df=prior.nu0, scale=prior.V0, random_state=self.rng
                        ),
                        dtype=np.float64,
                    ).reshape(dimension, dimension)
                elif (
                    use_heuristics
                    and len(selected) > 0
                    and heuristics.gaussian_mean_samples > 1
                ):
                    mean, precision, scale, degrees = prior.posterior_parameters(
                        selected
                    )
                    covariance = np.asarray(
                        invwishart.rvs(
                            df=degrees, scale=scale, random_state=self.rng
                        ),
                        dtype=np.float64,
                    ).reshape(dimension, dimension)
                    samples = np.asarray(
                        multivariate_normal.rvs(
                            mean=mean,
                            cov=covariance / precision,
                            size=heuristics.gaussian_mean_samples,
                            random_state=self.rng,
                        ),
                        dtype=np.float64,
                    ).reshape(heuristics.gaussian_mean_samples, dimension)
                    means[category] = samples.mean(axis=0)
                    covariances[category] = covariance
                else:
                    means[category], covariances[category] = prior.sample_posterior(
                        selected, rng=self.rng
                    )
            self.state.means[attr] = means
            self.state.covariances[attr] = covariances

    def _sample_attributes(self, theta: FloatArray) -> None:
        for doc_id, word_d in enumerate(self.state.words):
            table = np.empty((len(word_d), len(ATTRIBUTES)), dtype=np.float64)
            for word_id, word in enumerate(word_d):
                for attr_id, attr in enumerate(ATTRIBUTES):
                    category = self.global_category(
                        attr, self.attended_category(attr, doc_id)
                    )
                    table[word_id, attr_id] = np.log(theta[category, word] + 1e-300)
            self.state.attributes[doc_id] = self.attribute_sampler.sample_posterior(
                len(word_d), table
            )

    def _sample_theta(self) -> None:
        alpha = self.theta_posterior_parameters()
        sample_count = self.config.original_heuristics.theta_mean_samples
        if sample_count > 1:
            self.state.theta = np.asarray(
                [self.rng.dirichlet(row, size=sample_count).mean(axis=0) for row in alpha]
            )
            return
        counts = np.bincount(
            np.concatenate(self.state.words), minlength=len(self.vocabulary)
        )
        self.state.theta = sample_theta(
            alpha,
            rng=self.rng,
            sampler=self.config.word_model.theta_sampler,
            word_counts=counts,
            rescaling_mode=self.config.word_model.rescaling,
            n_candidates=self.config.word_model.importance_candidates,
        )

    def _initialize_frozen_gmm(self) -> None:
        if self.dataset.true_categories is None:
            raise ValueError("freeze.gmm requires full ground-truth categories")
        self.state.z = {
            attr: _copy_assignments(self.dataset.true_categories[attr])
            for attr in ATTRIBUTES
        }
        concentrations = self.config.hyperparameters.gmm.frozen_mixture_concentration.as_mapping()
        for attr in ATTRIBUTES:
            categories = _flatten_assignments(self.state.z[attr])
            counts = np.bincount(categories, minlength=self.category_counts[attr])
            self.state.pi[attr] = self.rng.dirichlet(counts + concentrations[attr])
        # Learning heuristics must not alter the existing frozen initialization.
        self._sample_gaussians(use_heuristics=False)

    def _giw_prior(self, attr: str, dimension: int) -> GiwPrior:
        gmm = self.config.hyperparameters.gmm
        return GiwPrior(
            m0=np.full(dimension, gmm.mean_prior.as_mapping()[attr]),
            kappa0=gmm.mean_precision.as_mapping()[attr],
            V0=np.eye(dimension) * gmm.covariance_scale.as_mapping()[attr],
            nu0=dimension + gmm.covariance_df_offset.as_mapping()[attr],
        )

    def _coerce_words(
        self, words: Sequence[npt.ArrayLike] | npt.ArrayLike | None
    ) -> tuple[IntArray, ...] | None:
        if words is None:
            return None
        array = np.asarray(words)
        if array.ndim == 2 and array.dtype != object:
            result = tuple(np.asarray(row, dtype=np.int64).copy() for row in array)
        else:
            result = tuple(np.asarray(row, dtype=np.int64).copy() for row in words)
        if len(result) != self.dataset.num_documents or any(
            row.ndim != 1 or len(row) == 0 for row in result
        ):
            raise ValueError("words must contain one non-empty 1-D sequence per document")
        return result

    def _validate_word_shapes(self, words: tuple[IntArray, ...]) -> None:
        expected = [len(sequence) for sequence in self.state.attributes]
        actual = [len(sequence) for sequence in words]
        if actual != expected:
            raise ValueError(f"word lengths must match F: expected {expected}, got {actual}")
        if any(np.any(row < 0) or np.any(row >= len(self.vocabulary)) for row in words):
            raise ValueError("word indexes must be within the vocabulary")

    def _validate_state(self, state: CslState) -> None:
        if len(state.words) != self.dataset.num_documents:
            raise ValueError("state word document count is inconsistent")
        self._validate_word_shapes(state.words)
        if state.theta.shape != (self.num_categories, len(state.vocabulary)):
            raise ValueError("theta shape is inconsistent")
        if not np.allclose(state.theta.sum(axis=1), 1.0):
            raise ValueError("theta rows must sum to one")
        for attr in ATTRIBUTES:
            dimension = self.dataset.flatten(attr).shape[1]
            categories = _flatten_assignments(state.z[attr])
            if np.any(categories < 0) or np.any(categories >= self.category_counts[attr]):
                raise ValueError(f"z[{attr!r}] contains an invalid category")
            if state.pi[attr].shape != (
                self.category_counts[attr],
            ) or not np.isclose(state.pi[attr].sum(), 1.0):
                raise ValueError(f"pi[{attr!r}] is invalid")
            if state.means[attr].shape != (self.category_counts[attr], dimension):
                raise ValueError(f"mean[{attr!r}] shape is invalid")
            expected_covariance = (self.category_counts[attr], dimension, dimension)
            if state.covariances[attr].shape != expected_covariance:
                raise ValueError(f"covariance[{attr!r}] shape is invalid")
            if any(np.linalg.eigvalsh(matrix).min() <= 0 for matrix in state.covariances[attr]):
                raise ValueError(f"covariance[{attr!r}] must be positive definite")


def _category_offsets(category_counts: Mapping[str, int]) -> dict[str, int]:
    offset = 0
    result: dict[str, int] = {}
    for attr in ATTRIBUTES:
        result[attr] = offset
        offset += category_counts[attr]
    return result


def _copy_assignments(value: CategoryAssignments) -> CategoryAssignments:
    if isinstance(value, np.ndarray):
        return value.copy()
    return tuple(item.copy() for item in value)


def _flatten_assignments(value: CategoryAssignments) -> IntArray:
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    return np.concatenate(value).astype(np.int64, copy=False)


def _sample_logweights(logweights: FloatArray, rng: np.random.Generator) -> int:
    shifted = logweights - np.max(logweights)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))
