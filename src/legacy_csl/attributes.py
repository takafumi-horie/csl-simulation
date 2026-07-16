"""Attribute-column sampling for Single-CSL and Inter-CSL."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt


AttributeMode: TypeAlias = Literal["distinct", "iid"]
WordAttributeLogProb: TypeAlias = (
    npt.NDArray[np.float64] | Callable[[int, int], float]
)


class AttributeSampler:
    """Sample attribute columns under a distinct or IID assignment model.

    非重複または独立同分布の割当モデルから
    属性列をサンプリングする。

    The log-probability input represents ``log p(word[position] | attribute)``.
    It can be either a ``(length, num_attributes)`` array or a callable receiving
    ``(position, attribute_index)``.  In ``distinct`` mode, complete ordered
    assignments are sampled jointly.  In ``iid`` mode, positions are sampled
    independently and repeated attributes are allowed.

    対数確率入力は ``log p(word[position] | attribute)`` を表し、
    ``(length, num_attributes)`` 配列、または ``(position, attribute_index)``
    を受け取る callable を指定できる。``distinct`` モードでは
    順序付き割当全体を同時にサンプリングし、``iid`` モードでは
    各位置を独立に
    サンプリングして属性の重複を許す。

    Args:
        attributes (Sequence[str]): Unique attribute names.
            一意な属性名。
        mode (AttributeMode): ``"distinct"`` for simple MEC2 or ``"iid"``
            for independent assignments with replacement.
            単純MEC2には ``"distinct"``、復元抽出の独立割当には
            ``"iid"`` を指定する。
        rng (numpy.random.Generator): Explicit random-number generator.
            明示的な乱数生成器。

    Raises:
        ValueError: If attributes are empty or duplicated, or mode is unknown.
            属性が空、重複している、または
            モードが未定義の場合。
    """

    def __init__(
        self,
        attributes: Sequence[str],
        mode: AttributeMode,
        rng: np.random.Generator,
    ) -> None:
        attribute_tuple = tuple(attributes)
        if not attribute_tuple:
            raise ValueError("attributes must not be empty")
        if len(set(attribute_tuple)) != len(attribute_tuple):
            raise ValueError("attributes must be unique")
        if mode not in ("distinct", "iid"):
            raise ValueError(f"unknown attribute sampling mode: {mode!r}")

        self._attributes = attribute_tuple
        self._mode = mode
        self._rng = rng

    @property
    def attributes(self) -> tuple[str, ...]:
        """Return the immutable attribute names.

        変更不能な属性名を返す。

        Returns:
            tuple[str, ...]: Attribute names in sampling-table order.
                サンプリング表の順序で並んだ属性名。
        """
        return self._attributes

    @property
    def mode(self) -> AttributeMode:
        """Return the assignment mode.

        割当モードを返す。

        Returns:
            AttributeMode: ``"distinct"`` or ``"iid"``.
                ``"distinct"`` または ``"iid"``。
        """
        return self._mode

    def sample_prior(self, length: int) -> tuple[str, ...]:
        """Sample an attribute column from its uniform prior.

        一様事前分布から属性列をサンプリングする。

        ``distinct`` uses uniform sampling without replacement, equivalent to
        ``random.sample``.  ``iid`` samples each position uniformly with
        replacement.

        ``distinct`` は ``random.sample`` と同等の
        一様な非復元抽出を行う。
        ``iid`` は各位置を一様に復元抽出する。

        Args:
            length (int): Number of word positions.
                単語位置の数。

        Returns:
            tuple[str, ...]: Sampled attribute column.
                サンプリングした属性列。

        Raises:
            ValueError: If length is invalid or exceeds the number of attributes
                in ``distinct`` mode.
                長さが不正、または ``distinct`` モードで
                属性数を超える場合。
        """
        self._validate_length(length)
        replace = self._mode == "iid"
        indexes = self._rng.choice(
            len(self._attributes), size=length, replace=replace
        )
        return tuple(self._attributes[int(index)] for index in indexes)

    def posterior_logweights(
        self,
        length: int,
        word_attribute_logprob: WordAttributeLogProb,
    ) -> npt.NDArray[np.float64]:
        """Compute posterior log-weights before normalization.

        正規化前の事後対数重みを計算する。

        For ``distinct``, the result is one-dimensional and follows
        :meth:`candidate_sequences` order.  Each weight is the summed word
        log-likelihood plus the uniform permutation log-prior.  For ``iid``,
        the result has shape ``(length, num_attributes)`` and contains each
        position's word log-likelihood plus its uniform categorical log-prior.

        ``distinct`` では :meth:`candidate_sequences` と同じ順序の
        1次元配列を返し、各重みは単語対数尤度の和と
        一様な順列事前分布の和である。
        ``iid`` では ``(length, num_attributes)`` 形状を返し、各位置の単語
        対数尤度に一様カテゴリ事前分布を加えた値となる。

        Args:
            length (int): Number of word positions.
                単語位置の数。
            word_attribute_logprob (WordAttributeLogProb): Log-probability table
                or a callable indexed by word position and attribute index.
                単語位置と属性番号で索引する
                対数確率表または callable。

        Returns:
            numpy.ndarray: Log-weights; shape depends on the assignment mode.
                割当モードに応じた形状の対数重み。

        Raises:
            ValueError: If length or log-probabilities are invalid.
                長さまたは対数確率が不正な場合。
        """
        self._validate_length(length)
        table = self._materialize_logprob_table(length, word_attribute_logprob)
        if self._mode == "iid":
            return table - math.log(len(self._attributes))

        candidates = self._candidate_index_sequences(length)
        word_positions = np.arange(length)
        likelihoods = np.fromiter(
            (
                table[word_positions, candidate].sum()
                for candidate in candidates
            ),
            dtype=np.float64,
            count=len(candidates),
        )
        return likelihoods - math.log(len(candidates))

    def posterior_probabilities(
        self,
        length: int,
        word_attribute_logprob: WordAttributeLogProb,
    ) -> npt.NDArray[np.float64]:
        """Compute normalized posterior probabilities.

        正規化した事後確率を計算する。

        In ``distinct`` mode all candidate probabilities sum to one.  In
        ``iid`` mode each row (word position) sums to one.

        ``distinct`` モードでは候補確率全体、``iid`` モードでは各行
        （各単語位置）の確率が1になる。

        Args:
            length (int): Number of word positions.
                単語位置の数。
            word_attribute_logprob (WordAttributeLogProb): Log-probability table
                or a callable indexed by word position and attribute index.
                単語位置と属性番号で索引する
                対数確率表または callable。

        Returns:
            numpy.ndarray: Normalized probabilities; shape depends on mode.
                モードに応じた形状の正規化済み確率。

        Raises:
            ValueError: If no assignment has finite probability.
                有限な確率を持つ割当が存在しない場合。
        """
        logweights = self.posterior_logweights(length, word_attribute_logprob)
        axis = None if self._mode == "distinct" else 1
        return _normalize_logweights(logweights, axis=axis)

    def sample_posterior(
        self,
        length: int,
        word_attribute_logprob: WordAttributeLogProb,
    ) -> tuple[str, ...]:
        """Sample an attribute column from the posterior distribution.

        事後分布から属性列をサンプリングする。

        Args:
            length (int): Number of word positions.
                単語位置の数。
            word_attribute_logprob (WordAttributeLogProb): Log-probability table
                or a callable indexed by word position and attribute index.
                単語位置と属性番号で索引する
                対数確率表または callable。

        Returns:
            tuple[str, ...]: Sampled attribute column.
                サンプリングした属性列。

        Raises:
            ValueError: If the input cannot define a valid posterior.
                入力から有効な事後分布を定義できない場合。
        """
        probabilities = self.posterior_probabilities(
            length, word_attribute_logprob
        )
        if self._mode == "distinct":
            candidate_id = int(self._rng.choice(len(probabilities), p=probabilities))
            return self.candidate_sequences(length)[candidate_id]

        indexes = [
            int(self._rng.choice(len(self._attributes), p=row))
            for row in probabilities
        ]
        return tuple(self._attributes[index] for index in indexes)

    def candidate_sequences(self, length: int) -> tuple[tuple[str, ...], ...]:
        """Enumerate ordered candidates for simple MEC2.

        単純MEC2の順序付き候補を列挙する。

        Args:
            length (int): Number of word positions.
                単語位置の数。

        Returns:
            tuple[tuple[str, ...], ...]: Permutations in lexicographic input order.
                入力順を基準とする辞書式順序の順列。

        Raises:
            ValueError: If called in ``iid`` mode or length is invalid.
                ``iid`` モードで呼び出した、または
                長さが不正な場合。
        """
        if self._mode != "distinct":
            raise ValueError("candidate sequences are only defined for distinct mode")
        self._validate_length(length)
        return tuple(itertools.permutations(self._attributes, length))

    def _candidate_index_sequences(
        self, length: int
    ) -> tuple[tuple[int, ...], ...]:
        """Enumerate attribute-index permutations.

        属性番号の順列を列挙する。

        Args:
            length (int): Number of word positions.
                単語位置の数。

        Returns:
            tuple[tuple[int, ...], ...]: Attribute-index permutations.
                属性番号の順列。
        """
        return tuple(itertools.permutations(range(len(self._attributes)), length))

    def _materialize_logprob_table(
        self,
        length: int,
        source: WordAttributeLogProb,
    ) -> npt.NDArray[np.float64]:
        """Materialize and validate a word-by-attribute log-probability table.

        単語×属性の対数確率表を実体化して検証する。

        Args:
            length (int): Number of word positions.
                単語位置の数。
            source (WordAttributeLogProb): Array or callable source.
                配列または callable の入力元。

        Returns:
            numpy.ndarray: A copied float64 table.
                コピーされた float64 表。

        Raises:
            ValueError: If shape or values are invalid.
                形状または値が不正な場合。
        """
        shape = (length, len(self._attributes))
        if callable(source):
            table = np.fromiter(
                (
                    float(source(position, attribute))
                    for position in range(length)
                    for attribute in range(len(self._attributes))
                ),
                dtype=np.float64,
                count=length * len(self._attributes),
            ).reshape(shape)
        else:
            table = np.array(source, dtype=np.float64, copy=True)
            if table.shape != shape:
                raise ValueError(
                    f"word_attribute_logprob must have shape {shape}, "
                    f"got {table.shape}"
                )

        if np.isnan(table).any() or np.isposinf(table).any():
            raise ValueError("log-probabilities must be finite or negative infinity")
        return table

    def _validate_length(self, length: int) -> None:
        """Validate a requested attribute-column length.

        指定された属性列長を検証する。

        Args:
            length (int): Requested number of positions.
                指定された位置数。

        Raises:
            ValueError: If length is negative, non-integral, or too large.
                長さが負、整数でない、または大きすぎる場合。
        """
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative integer")
        if self._mode == "distinct" and length > len(self._attributes):
            raise ValueError(
                "distinct mode requires length not to exceed the number of "
                f"attributes ({length} > {len(self._attributes)})"
            )


def _normalize_logweights(
    logweights: npt.NDArray[np.float64],
    *,
    axis: int | None,
) -> npt.NDArray[np.float64]:
    """Normalize log-weights stably along an axis.

    指定軸に沿って対数重みを数値的に安定して正規化する。

    Args:
        logweights (numpy.ndarray): Unnormalized log-weights.
            未正規化の対数重み。
        axis (int | None): Normalization axis, or ``None`` for the whole array.
            正規化する軸。配列全体の場合は ``None``。

    Returns:
        numpy.ndarray: Normalized probabilities.
            正規化した確率。

    Raises:
        ValueError: If any normalized slice has no finite weight.
            正規化対象に有限な重みが一つもない場合。
    """
    maxima = np.max(logweights, axis=axis, keepdims=True)
    if not np.all(np.isfinite(maxima)):
        raise ValueError("each posterior distribution needs a finite log-weight")
    weights = np.exp(logweights - maxima)
    totals = weights.sum(axis=axis, keepdims=True)
    return weights / totals
