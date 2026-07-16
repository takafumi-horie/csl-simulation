"""Gaussian--Inverse-Wishart prior and posterior sampling.

Gaussian--Inverse-Wishart 事前分布と事後サンプリング。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import invwishart, multivariate_normal

FloatArray = npt.NDArray[np.float64]
PosteriorParameters = tuple[FloatArray, float, FloatArray, float]


@dataclass(frozen=True, slots=True, init=False)
class GiwPrior:
    """Store a validated Gaussian--Inverse-Wishart prior.

    検証済み Gaussian--Inverse-Wishart 事前分布を保持する。

    Args:
        m0: Prior mean of the Gaussian mean.
            ガウス平均の事前平均。
        kappa0: Prior precision multiplier for the Gaussian mean.
            ガウス平均の事前精度係数。
        V0: Full scale matrix of the inverse-Wishart distribution.
            逆Wishart分布の完全なスケール行列。
        nu0: Degrees of freedom of the inverse-Wishart distribution.
            逆Wishart分布の自由度。

    Raises:
        ValueError: If a parameter is non-finite, has an invalid shape or domain,
            or if ``V0`` is not symmetric positive definite.
            パラメータが非有限、不正な形状・定義域、または ``V0`` が
            対称正定値でない場合。
    """

    _m0: FloatArray
    _kappa0: float
    _V0: FloatArray
    _nu0: float

    def __init__(
        self,
        m0: npt.ArrayLike,
        kappa0: float,
        V0: npt.ArrayLike,
        nu0: float,
    ) -> None:
        mean = np.asarray(m0, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("m0 must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(mean)):
            raise ValueError("m0 must contain only finite values")

        precision = _positive_finite_scalar(kappa0, "kappa0")
        scale = _validated_spd_matrix(V0, mean.size, "V0")
        degrees = _finite_scalar(nu0, "nu0")
        if degrees <= mean.size - 1:
            raise ValueError("nu0 must be greater than dimension - 1")

        mean = mean.copy()
        scale = scale.copy()
        mean.flags.writeable = False
        scale.flags.writeable = False
        object.__setattr__(self, "_m0", mean)
        object.__setattr__(self, "_kappa0", precision)
        object.__setattr__(self, "_V0", scale)
        object.__setattr__(self, "_nu0", degrees)

    @property
    def m0(self) -> FloatArray:
        """Return a copy of the prior mean.

        事前平均のコピーを返す。

        Returns:
            Copy of the prior mean vector.
            事前平均ベクトルのコピー。
        """
        return self._m0.copy()

    @property
    def kappa0(self) -> float:
        """Return the Gaussian precision multiplier.

        ガウス平均の精度係数を返す。

        Returns:
            Prior precision multiplier.
            事前精度係数。
        """
        return self._kappa0

    @property
    def V0(self) -> FloatArray:
        """Return a copy of the inverse-Wishart scale matrix.

        逆Wishartスケール行列のコピーを返す。

        Returns:
            Copy of the full scale matrix.
            完全なスケール行列のコピー。
        """
        return self._V0.copy()

    @property
    def nu0(self) -> float:
        """Return the inverse-Wishart degrees of freedom.

        逆Wishart分布の自由度を返す。

        Returns:
            Prior degrees of freedom.
            事前自由度。
        """
        return self._nu0

    @property
    def dimension(self) -> int:
        """Return the observation dimension.

        観測次元を返す。

        Returns:
            Observation dimension.
            観測次元。
        """
        return int(self._m0.size)

    def posterior_parameters(
        self, observations: npt.ArrayLike
    ) -> PosteriorParameters:
        """Calculate conjugate posterior parameters.

        共役事後分布のパラメータを計算する。

        Args:
            observations: Observation matrix with shape ``(n, dimension)``.
                ``(n, dimension)`` 形状の観測行列。

        Returns:
            ``(m_n, kappa_n, V_n, nu_n)`` in legacy CSL convention.
            legacy CSL 規約の ``(m_n, kappa_n, V_n, nu_n)``。

        Raises:
            ValueError: If observations have an invalid shape or values.
                観測の形状または値が不正な場合。
        """
        return posterior_parameters(self, observations)

    def sample_prior(
        self, *, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray]:
        """Draw ``(mu, Sigma)`` from the prior.

        事前分布から ``(mu, Sigma)`` をサンプルする。

        Args:
            rng: Explicit NumPy random generator.
                明示的なNumPy乱数生成器。

        Returns:
            Sampled mean vector and full covariance matrix.
            サンプルされた平均ベクトルと完全共分散行列。
        """
        return sample_prior(self, rng=rng)

    def sample_posterior(
        self,
        observations: npt.ArrayLike,
        *,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        """Draw ``(mu, Sigma)`` from the conjugate posterior.

        共役事後分布から ``(mu, Sigma)`` をサンプルする。

        Args:
            observations: Observation matrix with shape ``(n, dimension)``.
                ``(n, dimension)`` 形状の観測行列。
            rng: Explicit NumPy random generator.
                明示的なNumPy乱数生成器。

        Returns:
            Sampled posterior mean vector and full covariance matrix.
            サンプルされた事後平均ベクトルと完全共分散行列。

        Raises:
            ValueError: If observations have an invalid shape or values.
                観測の形状または値が不正な場合。
        """
        return sample_posterior(self, observations, rng=rng)


def posterior_parameters(
    prior: GiwPrior, observations: npt.ArrayLike
) -> PosteriorParameters:
    """Calculate legacy CSL Gaussian--Inverse-Wishart posterior parameters.

    legacy CSL の Gaussian--Inverse-Wishart 事後パラメータを計算する。

    Args:
        prior: Gaussian--Inverse-Wishart prior.
            Gaussian--Inverse-Wishart 事前分布。
        observations: Observation matrix with shape ``(n, dimension)``.
            ``(n, dimension)`` 形状の観測行列。

    Returns:
        ``(m_n, kappa_n, V_n, nu_n)``. The scale follows the exact legacy
        formula ``V0 + sum(x x^T) + kappa0 m0 m0^T - kappa_n m_n m_n^T``.
        ``(m_n, kappa_n, V_n, nu_n)``。スケールには legacy と同一の式
        ``V0 + sum(x x^T) + kappa0 m0 m0^T - kappa_n m_n m_n^T`` を用いる。

    Raises:
        ValueError: If observations have an invalid shape or values, or if the
            resulting scale is not positive definite.
            観測が不正、または事後スケールが正定値でない場合。
    """
    values = _validated_observations(observations, prior.dimension)
    count = len(values)
    if count == 0:
        return prior.m0, prior.kappa0, prior.V0, prior.nu0

    kappa_n = prior._kappa0 + count
    m_n = (prior._kappa0 * prior._m0 + values.sum(axis=0)) / kappa_n
    V_n = (
        prior._V0
        + values.T @ values
        + prior._kappa0 * np.outer(prior._m0, prior._m0)
        - kappa_n * np.outer(m_n, m_n)
    )
    V_n = (V_n + V_n.T) / 2.0
    _validate_positive_definite(V_n, "posterior V")
    return m_n, kappa_n, V_n, prior._nu0 + count


def sample_prior(
    prior: GiwPrior, *, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Draw a Gaussian mean and full covariance from a GIW prior.

    GIW事前分布からガウス平均と完全共分散をサンプルする。

    Args:
        prior: Gaussian--Inverse-Wishart prior.
            Gaussian--Inverse-Wishart 事前分布。
        rng: Explicit NumPy random generator.
            明示的なNumPy乱数生成器。

    Returns:
        Sampled ``(mu, Sigma)``.
        サンプルされた ``(mu, Sigma)``。
    """
    return _sample_parameters(
        prior._m0, prior._kappa0, prior._V0, prior._nu0, rng
    )


def sample_posterior(
    prior: GiwPrior,
    observations: npt.ArrayLike,
    *,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Draw a Gaussian mean and full covariance from the GIW posterior.

    GIW事後分布からガウス平均と完全共分散をサンプルする。

    Args:
        prior: Gaussian--Inverse-Wishart prior.
            Gaussian--Inverse-Wishart 事前分布。
        observations: Observation matrix with shape ``(n, dimension)``.
            ``(n, dimension)`` 形状の観測行列。
        rng: Explicit NumPy random generator.
            明示的なNumPy乱数生成器。

    Returns:
        Sampled posterior ``(mu, Sigma)``.
        サンプルされた事後 ``(mu, Sigma)``。

    Raises:
        ValueError: If observations have an invalid shape or values.
            観測の形状または値が不正な場合。
    """
    m_n, kappa_n, V_n, nu_n = posterior_parameters(prior, observations)
    return _sample_parameters(m_n, kappa_n, V_n, nu_n, rng)


def _sample_parameters(
    mean: FloatArray,
    precision: float,
    scale: FloatArray,
    degrees: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    covariance = np.asarray(
        invwishart.rvs(df=degrees, scale=scale, random_state=rng),
        dtype=np.float64,
    ).reshape((mean.size, mean.size))
    sampled_mean = np.asarray(
        multivariate_normal.rvs(
            mean=mean,
            cov=covariance / precision,
            random_state=rng,
        ),
        dtype=np.float64,
    ).reshape(mean.shape)
    return sampled_mean, covariance


def _validated_observations(
    observations: npt.ArrayLike, dimension: int
) -> FloatArray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 1 and values.size == 0:
        values = values.reshape((0, dimension))
    elif values.ndim == 1 and values.size == dimension:
        values = values.reshape((1, dimension))
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"observations must have shape (n, {dimension})")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must contain only finite values")
    return values


def _validated_spd_matrix(
    value: npt.ArrayLike, dimension: int, name: str
) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    _validate_positive_definite(matrix, name)
    return matrix


def _validate_positive_definite(matrix: FloatArray, name: str) -> None:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error


def _positive_finite_scalar(value: float, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar
