"""Tests for attribute-column sampling."""

import math

import numpy as np
import pytest

from legacy_csl.attributes import AttributeSampler


def test_distinct_candidates_and_samples_never_repeat_attributes() -> None:
    """Distinct mode enumerates and samples only unique assignments.

    distinct モードは重複しない割当だけを
    列挙・サンプリングする。
    """
    sampler = AttributeSampler(
        ("action", "position", "object"), "distinct", np.random.default_rng(7)
    )
    logprobs = np.zeros((2, 3), dtype=np.float64)

    assert sampler.candidate_sequences(2) == (
        ("action", "position"),
        ("action", "object"),
        ("position", "action"),
        ("position", "object"),
        ("object", "action"),
        ("object", "position"),
    )
    for _ in range(30):
        prior = sampler.sample_prior(2)
        posterior = sampler.sample_posterior(2, logprobs)
        assert len(set(prior)) == 2
        assert len(set(posterior)) == 2


def test_iid_posterior_can_repeat_attributes_with_callable_logprobs() -> None:
    """IID mode independently selects a deterministic repeated attribute.

    IID モードは決定的な同一属性を各位置で独立に選択できる。
    """
    sampler = AttributeSampler(
        ("action", "object"), "iid", np.random.default_rng(1)
    )

    def fake_logprob(position: int, attribute: int) -> float:
        """Favor the action attribute at every position.

        すべての位置で action 属性を選好する。

        Args:
            position (int): Word position.
                単語位置。
            attribute (int): Attribute index.
                属性番号。

        Returns:
            float: Zero for action and negative infinity otherwise.
                action なら0、それ以外なら負の無限大。
        """
        del position
        return 0.0 if attribute == 0 else -math.inf

    assert sampler.sample_posterior(3, fake_logprob) == (
        "action",
        "action",
        "action",
    )


def test_distinct_posterior_matches_manual_normalization() -> None:
    """Simple MEC2 probabilities match the two-permutation calculation.

    単純MEC2の確率が2順列の手計算と一致する。
    """
    sampler = AttributeSampler(
        ("action", "object"), "distinct", np.random.default_rng(2)
    )
    logprobs = np.log(np.array([[0.8, 0.2], [0.3, 0.7]]))

    logweights = sampler.posterior_logweights(2, logprobs)
    probabilities = sampler.posterior_probabilities(2, logprobs)

    np.testing.assert_allclose(
        logweights,
        np.log(np.array([0.8 * 0.7 / 2.0, 0.2 * 0.3 / 2.0])),
    )
    np.testing.assert_allclose(probabilities, np.array([0.56, 0.06]) / 0.62)
    assert probabilities.sum() == pytest.approx(1.0)


def test_iid_posterior_normalizes_each_position() -> None:
    """IID posterior rows match their categorical probabilities.

    IID事後分布の各行が対応するカテゴリ確率と一致する。
    """
    sampler = AttributeSampler(
        ("action", "object"), "iid", np.random.default_rng(3)
    )
    expected = np.array([[0.8, 0.2], [0.3, 0.7]])

    probabilities = sampler.posterior_probabilities(2, np.log(expected))

    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


def test_distinct_rejects_more_words_than_attributes() -> None:
    """Distinct mode rejects columns longer than its attribute set.

    distinct モードは属性数より長い列を拒否する。
    """
    sampler = AttributeSampler(
        ("action", "object"), "distinct", np.random.default_rng(4)
    )

    with pytest.raises(ValueError, match="not to exceed"):
        sampler.sample_prior(3)
    with pytest.raises(ValueError, match="not to exceed"):
        sampler.sample_posterior(3, np.zeros((3, 2)))


@pytest.mark.parametrize("mode", ["distinct", "iid"])
def test_seed_reproduces_prior_and_posterior(mode: str) -> None:
    """An equal seed reproduces both prior and posterior draws.

    同じseedで事前・事後のサンプルを再現できる。

    Args:
        mode (str): Assignment mode under test.
            テスト対象の割当モード。
    """
    left = AttributeSampler(
        ("action", "position", "object"), mode, np.random.default_rng(42)
    )
    right = AttributeSampler(
        ("action", "position", "object"), mode, np.random.default_rng(42)
    )
    logprobs = np.log(np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]))

    assert left.sample_prior(2) == right.sample_prior(2)
    assert left.sample_posterior(2, logprobs) == right.sample_posterior(
        2, logprobs
    )


def test_logprob_input_validation() -> None:
    """Invalid table shapes and impossible rows are rejected.

    不正な表形状と確率ゼロしかない行を拒否する。
    """
    sampler = AttributeSampler(
        ("action", "object"), "iid", np.random.default_rng(5)
    )

    with pytest.raises(ValueError, match="shape"):
        sampler.posterior_probabilities(2, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite log-weight"):
        sampler.posterior_probabilities(
            2, np.array([[0.0, 0.0], [-math.inf, -math.inf]])
        )
