"""Core categorical Uni-CSL state and inference updates.

categorical Uni-CSL の状態と推論更新の中核実装。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .attributes import AttributeSampler
from .config import Config
from .data import ATTRIBUTES, ObservationDataset
from .rescaling import rescale_theta, sample_theta


@dataclass(slots=True)
class CslState:
    """Own all mutable random variables for one CSL agent.

    1体の CSL エージェントに属する可変確率変数を保持する。

    Args:
        z: Document category assignments keyed by attribute.
            属性別の文書カテゴリ割当て。
        pi: Category probabilities keyed by attribute.
            属性別のカテゴリ確率。
        means: Gaussian means keyed by attribute.
            属性別のガウス平均。
        variances: Diagonal Gaussian variances keyed by attribute.
            属性別の対角ガウス分散。
        theta: Global category-to-word probabilities.
            全カテゴリから単語への確率。
        attributes: Attribute sequence for every document.
            文書ごとの属性列。
        words: Integer word sequence for every document.
            文書ごとの整数単語列。
    """

    z: dict[str, npt.NDArray[np.int64]]
    pi: dict[str, npt.NDArray[np.float64]]
    means: dict[str, npt.NDArray[np.float64]]
    variances: dict[str, npt.NDArray[np.float64]]
    theta: npt.NDArray[np.float64]
    attributes: list[tuple[str, ...]]
    words: npt.NDArray[np.int64]

    def copy(self) -> CslState:
        """Return a deep independent state copy.

        独立した状態の深いコピーを返す。

        Returns:
            Independent state copy.
            独立した状態コピー。
        """
        return CslState(
            z={key: value.copy() for key, value in self.z.items()},
            pi={key: value.copy() for key, value in self.pi.items()},
            means={key: value.copy() for key, value in self.means.items()},
            variances={key: value.copy() for key, value in self.variances.items()},
            theta=self.theta.copy(),
            attributes=list(self.attributes),
            words=self.words.copy(),
        )


class CslModel:
    """Run categorical Gibbs-style updates for one CSL agent.

    1体のエージェントについて categorical Gibbs 型更新を実行する。

    This class deliberately composes probability helpers instead of reproducing
    the inheritance-heavy legacy variable hierarchy. All word-probability
    consumers use :meth:`effective_theta`, which guarantees one consistent
    rescaling definition for words, z, F, and Inter-CSL MH ratios.

    継承中心の旧変数階層は再現せず確率ヘルパーを合成する。単語、z、F、
    Inter-CSL の MH 比はすべて :meth:`effective_theta` を通じ、同一の
    リスケーリング定義を参照する。
    """

    def __init__(
        self,
        dataset: ObservationDataset,
        config: Config,
        *,
        rng: np.random.Generator,
        initial_words: npt.ArrayLike | None = None,
    ) -> None:
        """Initialize an agent and its owned state.

        エージェントと所有状態を初期化する。

        Args:
            dataset: Attended observations for the agent.
                エージェントの注視対象観測。
            config: Validated experiment configuration.
                検証済み実験設定。
            rng: Explicit generator owned by this model.
                このモデルが所有する明示的乱数生成器。
            initial_words: Optional fixed/initial ``(D, L)`` word matrix.
                任意の固定・初期 ``(D, L)`` 単語行列。

        Raises:
            ValueError: If supplied words do not match the dataset/configuration.
                単語列がデータまたは設定と一致しない場合。
        """
        self.dataset = dataset
        self.config = config
        self.rng = rng
        self.category_counts = dict(
            config.hyperparameters.category_counts.as_mapping()
        )
        self.offsets = _category_offsets(self.category_counts)
        self.num_categories = sum(self.category_counts.values())
        self.vocabulary_size = config.hyperparameters.vocabulary_size
        self.word_length = config.hyperparameters.word_length
        self.attribute_sampler = AttributeSampler(
            ATTRIBUTES, config.attribute_model.exclusivity, rng
        )
        words = self._initialize_words(initial_words)
        z = {
            attr: rng.integers(0, self.category_counts[attr], dataset.num_documents)
            for attr in ATTRIBUTES
        }
        pi = {
            attr: np.full(self.category_counts[attr], 1.0 / self.category_counts[attr])
            for attr in ATTRIBUTES
        }
        means, variances = self._initialize_gaussians(z)
        theta = np.asarray(
            [
                rng.dirichlet(
                    np.full(
                        self.vocabulary_size,
                        config.hyperparameters.gamma,
                        dtype=np.float64,
                    )
                )
                for _ in range(self.num_categories)
            ]
        )
        attributes = [
            self.attribute_sampler.sample_prior(self.word_length)
            for _ in range(dataset.num_documents)
        ]
        self.state = CslState(z, pi, means, variances, theta, attributes, words)

    @classmethod
    def from_state(
        cls,
        dataset: ObservationDataset,
        config: Config,
        state: CslState,
        *,
        rng: np.random.Generator,
    ) -> CslModel:
        """Rebuild a model around a validated loaded state.

        読み込んだ検証済み状態からモデルを再構築する。

        Args:
            dataset: Agent observations.
                エージェント観測。
            config: Experiment configuration.
                実験設定。
            state: Loaded model state.
                読み込んだモデル状態。
            rng: Explicit random generator.
                明示的乱数生成器。

        Returns:
            Reconstructed model.
            再構築したモデル。
        """
        model = cls(dataset, config, rng=rng, initial_words=state.words)
        model._validate_state(state)
        model.state = state.copy()
        return model

    def effective_theta(
        self, words: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Return the configured read-only word-probability view as a copy.

        設定された単語確率を独立コピーとして返す。

        Args:
            words: Word matrix used to form empirical unigram counts.
                経験的ユニグラム頻度に使う単語行列。

        Returns:
            Rescaled category-by-word probability matrix.
            補正済みカテゴリ×単語確率行列。
        """
        empirical = None
        if self.config.word_model.rescaling == "empirical_unigram":
            word_array = self.state.words if words is None else np.asarray(words)
            empirical = np.bincount(
                word_array.astype(np.int64).reshape(-1),
                minlength=self.vocabulary_size,
            ).astype(np.float64) + 1.0
        return rescale_theta(
            self.state.theta,
            self.config.word_model.rescaling,
            empirical_unigram=empirical,
        )

    def step(self, words: npt.ArrayLike | None = None) -> None:
        """Perform one complete inference sweep.

        1回の完全な推論スイープを実行する。

        Args:
            words: Optional accepted/fixed word matrix used in this sweep.
                このスイープで使う任意の受理済み・固定単語行列。
        """
        if words is not None:
            self.state.words = self._validated_words(words)
        effective_theta = self.effective_theta(self.state.words)
        if not self.config.freeze.gmm:
            self._sample_z(effective_theta)
            self._update_gaussians()
            self._sample_pi()
        if not self.config.freeze.attributes:
            self._sample_attributes(effective_theta)
        if not self.config.freeze.theta:
            self._sample_theta()

    def draw_words(self) -> npt.NDArray[np.int64]:
        """Draw a new word matrix from the current rescaled theta.

        現在の補正済み theta から新しい単語行列を生成する。

        Returns:
            Sampled ``(D, L)`` integer matrix.
            サンプルした ``(D, L)`` 整数行列。
        """
        probabilities = self.effective_theta(self.state.words)
        words = np.empty_like(self.state.words)
        for doc_id, attributes in enumerate(self.state.attributes):
            for word_id, attr in enumerate(attributes):
                category = self.global_category(attr, self.state.z[attr][doc_id])
                words[doc_id, word_id] = self.rng.choice(
                    self.vocabulary_size, p=probabilities[category]
                )
        return words

    def predict_observations(
        self, words: npt.ArrayLike
    ) -> tuple[dict[str, npt.NDArray[np.int64]], dict[str, npt.NDArray[np.float64]]]:
        """Predict categories and modality means from received words.

        受信単語列からカテゴリとモダリティ平均を予測する。

        Args:
            words: Received ``(D, L)`` word matrix.
                受信した ``(D, L)`` 単語行列。

        Returns:
            Predicted categories and observation means by attribute.
            属性別の予測カテゴリと観測平均。
        """
        word_array = self._validated_words(words)
        theta = self.effective_theta(word_array)
        predicted_z: dict[str, npt.NDArray[np.int64]] = {}
        predicted_x: dict[str, npt.NDArray[np.float64]] = {}
        for attr in ATTRIBUTES:
            values = np.empty(self.dataset.num_documents, dtype=np.int64)
            for doc_id in range(self.dataset.num_documents):
                positions = [
                    index
                    for index, assigned in enumerate(self.state.attributes[doc_id])
                    if assigned == attr
                ]
                logweights = np.log(self.state.pi[attr] + 1e-300)
                for category in range(self.category_counts[attr]):
                    global_id = self.global_category(attr, category)
                    if positions:
                        logweights[category] += np.log(
                            theta[global_id, word_array[doc_id, positions]] + 1e-300
                        ).sum()
                values[doc_id] = int(np.argmax(logweights))
            predicted_z[attr] = values
            predicted_x[attr] = self.state.means[attr][values].copy()
        return predicted_z, predicted_x

    def global_category(self, attribute: str, local_category: int) -> int:
        """Convert an attribute-local category into a theta row.

        属性内カテゴリを theta の行番号に変換する。

        Args:
            attribute: Attribute name.
                属性名。
            local_category: Category index within the attribute.
                属性内カテゴリ番号。

        Returns:
            Global category row.
            全体カテゴリ行番号。
        """
        return self.offsets[attribute] + int(local_category)

    def readonly_z(self) -> Mapping[str, npt.NDArray[np.int64]]:
        """Return copied category assignments behind a read-only mapping.

        コピーしたカテゴリ割当てを読み取り専用マッピングで返す。

        Returns:
            Read-only attribute-to-category mapping.
            読み取り専用の属性・カテゴリマッピング。
        """
        return MappingProxyType({key: value.copy() for key, value in self.state.z.items()})

    def _initialize_words(
        self, initial_words: npt.ArrayLike | None
    ) -> npt.NDArray[np.int64]:
        if initial_words is None:
            return self.rng.integers(
                0,
                self.vocabulary_size,
                size=(self.dataset.num_documents, self.word_length),
            )
        return self._validated_words(initial_words)

    def _validated_words(self, words: npt.ArrayLike) -> npt.NDArray[np.int64]:
        array = np.asarray(words, dtype=np.int64)
        expected = (self.dataset.num_documents, self.word_length)
        if array.shape != expected:
            raise ValueError(f"words must have shape {expected}, got {array.shape}")
        if np.any(array < 0) or np.any(array >= self.vocabulary_size):
            raise ValueError("word indexes must be within the configured vocabulary")
        return array.copy()

    def _initialize_gaussians(
        self, z: Mapping[str, npt.NDArray[np.int64]]
    ) -> tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
    ]:
        means: dict[str, npt.NDArray[np.float64]] = {}
        variances: dict[str, npt.NDArray[np.float64]] = {}
        for attr in ATTRIBUTES:
            observations = np.asarray(self.dataset.observations[attr], dtype=np.float64)
            global_variance = np.var(observations, axis=0) + 1e-3
            attr_means = []
            for category in range(self.category_counts[attr]):
                selected = observations[z[attr] == category]
                if selected.size:
                    attr_means.append(selected.mean(axis=0))
                else:
                    attr_means.append(observations[self.rng.integers(len(observations))])
            means[attr] = np.asarray(attr_means, dtype=np.float64)
            variances[attr] = np.repeat(
                global_variance[np.newaxis, :], self.category_counts[attr], axis=0
            )
        return means, variances

    def _sample_z(self, theta: npt.NDArray[np.float64]) -> None:
        for attr in ATTRIBUTES:
            observations = np.asarray(self.dataset.observations[attr], dtype=np.float64)
            for doc_id, observation in enumerate(observations):
                positions = [
                    index
                    for index, assigned in enumerate(self.state.attributes[doc_id])
                    if assigned == attr
                ]
                logweights = np.log(self.state.pi[attr] + 1e-300)
                for category in range(self.category_counts[attr]):
                    logweights[category] += _diagonal_gaussian_logpdf(
                        observation,
                        self.state.means[attr][category],
                        self.state.variances[attr][category],
                    )
                    if positions:
                        global_id = self.global_category(attr, category)
                        logweights[category] += np.log(
                            theta[
                                global_id,
                                self.state.words[doc_id, positions],
                            ]
                            + 1e-300
                        ).sum()
                self.state.z[attr][doc_id] = _sample_logweights(logweights, self.rng)

    def _update_gaussians(self) -> None:
        for attr in ATTRIBUTES:
            observations = np.asarray(self.dataset.observations[attr], dtype=np.float64)
            global_mean = observations.mean(axis=0)
            global_variance = np.var(observations, axis=0) + 1e-3
            for category in range(self.category_counts[attr]):
                selected = observations[self.state.z[attr] == category]
                if selected.size == 0:
                    self.state.means[attr][category] = global_mean
                    self.state.variances[attr][category] = global_variance
                    continue
                count = len(selected)
                posterior_mean = (selected.sum(axis=0) + global_mean) / (count + 1)
                centered = selected - posterior_mean
                posterior_variance = (
                    np.square(centered).sum(axis=0) + global_variance
                ) / (count + 1)
                self.state.means[attr][category] = posterior_mean
                self.state.variances[attr][category] = np.maximum(
                    posterior_variance, 1e-6
                )

    def _sample_pi(self) -> None:
        for attr in ATTRIBUTES:
            counts = np.bincount(
                self.state.z[attr], minlength=self.category_counts[attr]
            )
            self.state.pi[attr] = self.rng.dirichlet(counts + 1.0)

    def _sample_attributes(self, theta: npt.NDArray[np.float64]) -> None:
        for doc_id in range(self.dataset.num_documents):
            table = np.empty((self.word_length, len(ATTRIBUTES)), dtype=np.float64)
            for word_id, word in enumerate(self.state.words[doc_id]):
                for attr_id, attr in enumerate(ATTRIBUTES):
                    category = self.global_category(attr, self.state.z[attr][doc_id])
                    table[word_id, attr_id] = np.log(theta[category, word] + 1e-300)
            self.state.attributes[doc_id] = self.attribute_sampler.sample_posterior(
                self.word_length, table
            )

    def _sample_theta(self) -> None:
        alpha = np.full(
            (self.num_categories, self.vocabulary_size),
            self.config.hyperparameters.gamma,
            dtype=np.float64,
        )
        for doc_id, attributes in enumerate(self.state.attributes):
            for word_id, attr in enumerate(attributes):
                category = self.global_category(attr, self.state.z[attr][doc_id])
                alpha[category, self.state.words[doc_id, word_id]] += 1.0
        word_counts = np.bincount(
            self.state.words.reshape(-1), minlength=self.vocabulary_size
        )
        self.state.theta = sample_theta(
            alpha,
            rng=self.rng,
            sampler=self.config.word_model.theta_sampler,
            word_counts=word_counts,
            rescaling_mode=self.config.word_model.rescaling,
            n_candidates=self.config.word_model.importance_candidates,
        )

    def _validate_state(self, state: CslState) -> None:
        self._validated_words(state.words)
        if state.theta.shape != (self.num_categories, self.vocabulary_size):
            raise ValueError("loaded theta shape is inconsistent with configuration")
        if len(state.attributes) != self.dataset.num_documents:
            raise ValueError("loaded attribute sequence count is inconsistent")


def _category_offsets(category_counts: Mapping[str, int]) -> dict[str, int]:
    offset = 0
    offsets: dict[str, int] = {}
    for attr in ATTRIBUTES:
        offsets[attr] = offset
        offset += category_counts[attr]
    return offsets


def _diagonal_gaussian_logpdf(
    value: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    variance: npt.NDArray[np.float64],
) -> float:
    return float(
        -0.5
        * (
            len(value) * np.log(2.0 * np.pi)
            + np.log(variance).sum()
            + (np.square(value - mean) / variance).sum()
        )
    )


def _sample_logweights(
    logweights: npt.NDArray[np.float64], rng: np.random.Generator
) -> int:
    shifted = logweights - np.max(logweights)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))

