"""Word-probability rescaling and theta sampling utilities.

単語確率のリスケーリングと theta サンプリングの補助機能。
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt


FloatArray: TypeAlias = npt.NDArray[np.float64]
RescalingMode: TypeAlias = Literal["legacy_theta_sum", "empirical_unigram", "none"]
ThetaSampler: TypeAlias = Literal["legacy_importance", "dirichlet"]


def _as_probability_matrix(theta: npt.ArrayLike) -> FloatArray:
    """Validate and copy a category-by-word probability matrix.

    カテゴリ×単語の確率行列を検証して複製する。

    Args:
        theta (ArrayLike): Category-by-word probability matrix.
            カテゴリ×単語の確率行列。

    Returns:
        NDArray[np.float64]: A validated, independent float64 copy.
            検証済みの独立した float64 複製。

    Raises:
        ValueError: If the matrix is empty, non-finite, negative, or its rows
            do not sum to one.
            行列が空、非有限、負、または行和が1でない場合。
    """
    array = np.asarray(theta, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("theta must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("theta must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError("theta must contain only non-negative values")
    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("every theta row must sum to one")
    return array.copy()


def _normalize_rows(weights: FloatArray) -> FloatArray:
    """Normalize each matrix row without mutating the input.

    入力を変更せずに各行を正規化する。

    Args:
        weights (NDArray[np.float64]): Non-negative matrix with positive rows.
            各行が正の質量を持つ非負行列。

    Returns:
        NDArray[np.float64]: Row-normalized matrix.
            行ごとに正規化した行列。
    """
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        raise ValueError("rescaling must leave positive finite mass in every row")
    return np.asarray(weights / row_sums, dtype=np.float64)


def legacy_theta_sum_rescaling(theta: npt.ArrayLike) -> FloatArray:
    """Apply the original theta-column-sum rescaling.

    For each word ``w``, this computes ``R_w = sum_z theta[z, w]``, divides
    every category row by ``R``, and normalizes that row again.

    各単語 ``w`` に対し ``R_w = sum_z theta[z, w]`` を計算し、各カテゴリ
    行を ``R`` で割った後に再度行正規化する。

    Args:
        theta (ArrayLike): Category-by-word probability matrix.
            カテゴリ×単語の確率行列。

    Returns:
        NDArray[np.float64]: Rescaled row-stochastic matrix.
            リスケーリング済みの行確率行列。

    Raises:
        ValueError: If theta is invalid or any word has zero total mass.
            theta が不正、または列の総質量がゼロの場合。
    """
    probabilities = _as_probability_matrix(theta)
    rescaling_term = probabilities.sum(axis=0)
    if np.any(rescaling_term <= 0.0):
        raise ValueError("every theta column must have positive mass")
    return _normalize_rows(probabilities / rescaling_term)


def legacy_rescaling_term(theta: npt.ArrayLike) -> FloatArray:
    """Compute the original per-word theta-sum term.

    従来の単語ごとの theta 列和項を計算する。

    Args:
        theta (ArrayLike): Category-by-word probability matrix.
            カテゴリ×単語の確率行列。

    Returns:
        NDArray[np.float64]: ``R_w = sum_z theta[z, w]`` for every word.
            各単語の ``R_w = sum_z theta[z, w]``。

    Raises:
        ValueError: If theta is invalid or any word has zero total mass.
            theta が不正、または列の総質量がゼロの場合。
    """
    rescaling_term = _as_probability_matrix(theta).sum(axis=0)
    if np.any(rescaling_term <= 0.0):
        raise ValueError("every theta column must have positive mass")
    return np.asarray(rescaling_term, dtype=np.float64)


def empirical_unigram_rescaling(
    theta: npt.ArrayLike,
    unigram: npt.ArrayLike,
) -> FloatArray:
    """Rescale theta by an empirical unigram distribution.

    The supplied values may be counts or probabilities. Only their relative
    magnitudes matter because each resulting theta row is normalized.

    経験的ユニグラム分布で theta をリスケーリングする。入力は頻度でも
    確率でもよく、行正規化のため相対的大きさだけが影響する。

    Args:
        theta (ArrayLike): Category-by-word probability matrix.
            カテゴリ×単語の確率行列。
        unigram (ArrayLike): Positive empirical count or probability per word.
            単語ごとの正の経験頻度または確率。

    Returns:
        NDArray[np.float64]: Rescaled row-stochastic matrix.
            リスケーリング済みの行確率行列。

    Raises:
        ValueError: If theta or the unigram vector is invalid.
            theta またはユニグラムベクトルが不正な場合。
    """
    probabilities = _as_probability_matrix(theta)
    unigram_array = np.asarray(unigram, dtype=np.float64)
    if unigram_array.ndim != 1 or unigram_array.shape[0] != probabilities.shape[1]:
        raise ValueError("unigram must be one-dimensional and match theta columns")
    if not np.all(np.isfinite(unigram_array)) or np.any(unigram_array <= 0.0):
        raise ValueError("unigram must contain only positive finite values")
    return _normalize_rows(probabilities / unigram_array[np.newaxis, :])


def rescale_theta(
    theta: npt.ArrayLike,
    mode: RescalingMode = "legacy_theta_sum",
    *,
    empirical_unigram: npt.ArrayLike | None = None,
) -> FloatArray:
    """Dispatch theta rescaling by mode.

    指定した方式に応じて theta のリスケーリングを実行する。

    Args:
        theta (ArrayLike): Category-by-word probability matrix.
            カテゴリ×単語の確率行列。
        mode (RescalingMode): ``legacy_theta_sum``, ``empirical_unigram``, or
            ``none``.
            ``legacy_theta_sum``、``empirical_unigram``、``none`` のいずれか。
        empirical_unigram (ArrayLike | None): Required only by
            ``empirical_unigram``.
            ``empirical_unigram`` の場合にのみ必要なユニグラム。

    Returns:
        NDArray[np.float64]: An independent float64 probability matrix.
            独立した float64 確率行列。

    Raises:
        ValueError: If the mode or its arguments are invalid.
            方式またはその引数が不正な場合。
    """
    if mode == "legacy_theta_sum":
        return legacy_theta_sum_rescaling(theta)
    if mode == "empirical_unigram":
        if empirical_unigram is None:
            raise ValueError("empirical_unigram is required for empirical rescaling")
        return empirical_unigram_rescaling(theta, empirical_unigram)
    if mode == "none":
        return _as_probability_matrix(theta)
    raise ValueError(f"unknown rescaling mode: {mode!r}")


def _as_dirichlet_parameters(alpha_count: npt.ArrayLike) -> FloatArray:
    """Validate and copy category-by-word Dirichlet parameters.

    カテゴリ×単語のディリクレパラメータを検証して複製する。

    Args:
        alpha_count (ArrayLike): Positive Dirichlet parameters.
            正のディリクレパラメータ。

    Returns:
        NDArray[np.float64]: Validated float64 parameters.
            検証済みの float64 パラメータ。

    Raises:
        ValueError: If parameters are not a positive finite 2-D matrix.
            パラメータが正の有限値からなる2次元行列でない場合。
    """
    alpha = np.asarray(alpha_count, dtype=np.float64)
    if alpha.ndim != 2 or 0 in alpha.shape:
        raise ValueError("alpha_count must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha_count must contain only positive finite values")
    return alpha.copy()


def sample_dirichlet_theta(
    alpha_count: npt.ArrayLike,
    *,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw one theta matrix from independent posterior Dirichlet rows.

    独立な事後ディリクレ分布の各行から theta をサンプルする。

    Args:
        alpha_count (ArrayLike): Category-by-word posterior parameters.
            カテゴリ×単語の事後パラメータ。
        rng (np.random.Generator): Explicit random-number generator.
            明示的な乱数生成器。

    Returns:
        NDArray[np.float64]: Sampled category-by-word probabilities.
            サンプルしたカテゴリ×単語確率。

    Raises:
        ValueError: If the posterior parameters are invalid.
            事後パラメータが不正な場合。
    """
    alpha = _as_dirichlet_parameters(alpha_count)
    return np.asarray([rng.dirichlet(row) for row in alpha], dtype=np.float64)


def sample_legacy_importance_theta(
    alpha_count: npt.ArrayLike,
    word_counts: npt.ArrayLike,
    *,
    rng: np.random.Generator,
    n_candidates: int = 10_000,
) -> FloatArray:
    """Sample theta with the original rescaling importance correction.

    Candidate rows are drawn from their posterior Dirichlet distributions. A
    candidate is then selected with log weight
    ``-sum_w word_counts[w] * log(sum_z theta[z, w])``.

    候補の各行を事後ディリクレ分布からサンプルし、
    ``-sum_w word_counts[w] * log(sum_z theta[z, w])`` を対数重みとして
    候補を1つ選ぶ。

    Args:
        alpha_count (ArrayLike): Category-by-word posterior parameters.
            カテゴリ×単語の事後パラメータ。
        word_counts (ArrayLike): Non-negative observed count for each word.
            単語ごとの非負の観測頻度。
        rng (np.random.Generator): Explicit random-number generator.
            明示的な乱数生成器。
        n_candidates (int): Number of importance candidates; the legacy
            default is 10,000.
            importance 候補数。legacy 方式の既定値は10,000。

    Returns:
        NDArray[np.float64]: The selected category-by-word probability matrix.
            選択したカテゴリ×単語確率行列。

    Raises:
        ValueError: If parameters, counts, or candidate count are invalid.
            パラメータ、頻度、または候補数が不正な場合。
    """
    alpha = _as_dirichlet_parameters(alpha_count)
    counts = np.asarray(word_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.shape[0] != alpha.shape[1]:
        raise ValueError("word_counts must be one-dimensional and match alpha columns")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("word_counts must contain only non-negative finite values")
    if isinstance(n_candidates, bool) or not isinstance(
        n_candidates, (int, np.integer)
    ):
        raise ValueError("n_candidates must be a positive integer")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be a positive integer")

    candidates = np.empty((n_candidates, *alpha.shape), dtype=np.float64)
    for category, row in enumerate(alpha):
        candidates[:, category, :] = rng.dirichlet(row, size=n_candidates)

    observed = counts > 0.0
    if np.any(observed):
        column_sums = candidates[:, :, observed].sum(axis=1)
        log_weights = -(np.log(column_sums) @ counts[observed])
    else:
        log_weights = np.zeros(n_candidates, dtype=np.float64)

    infinite = np.isposinf(log_weights)
    if np.any(infinite):
        selected = int(rng.choice(np.flatnonzero(infinite)))
        return candidates[selected].copy()
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("importance log weights must be finite")
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("importance weights could not be normalized")
    selected = int(rng.choice(n_candidates, p=weights / weight_sum))
    return candidates[selected].copy()


def sample_theta(
    alpha_count: npt.ArrayLike,
    *,
    rng: np.random.Generator,
    sampler: ThetaSampler = "legacy_importance",
    word_counts: npt.ArrayLike | None = None,
    rescaling_mode: RescalingMode = "legacy_theta_sum",
    n_candidates: int = 10_000,
) -> FloatArray:
    """Dispatch posterior theta sampling by method.

    指定した方式に応じて事後 theta サンプリングを実行する。

    Args:
        alpha_count (ArrayLike): Category-by-word posterior parameters.
            カテゴリ×単語の事後パラメータ。
        rng (np.random.Generator): Explicit random-number generator.
            明示的な乱数生成器。
        sampler (ThetaSampler): ``legacy_importance`` or ``dirichlet``.
            ``legacy_importance`` または ``dirichlet``。
        word_counts (ArrayLike | None): Required by ``legacy_importance``.
            ``legacy_importance`` で必要な単語頻度。
        rescaling_mode (RescalingMode): Active word-rescaling mode.
            有効な単語リスケーリング方式。
        n_candidates (int): Importance candidate count.
            importance 候補数。

    Returns:
        NDArray[np.float64]: Sampled theta matrix.
            サンプルした theta 行列。

    Raises:
        ValueError: If the method or required combination is invalid.
            方式または必要な組み合わせが不正な場合。
    """
    if sampler == "dirichlet":
        return sample_dirichlet_theta(alpha_count, rng=rng)
    if sampler == "legacy_importance":
        if rescaling_mode != "legacy_theta_sum":
            raise ValueError(
                "legacy_importance requires rescaling_mode='legacy_theta_sum'"
            )
        if word_counts is None:
            raise ValueError("word_counts is required for legacy_importance")
        return sample_legacy_importance_theta(
            alpha_count,
            word_counts,
            rng=rng,
            n_candidates=n_candidates,
        )
    raise ValueError(f"unknown theta sampler: {sampler!r}")
