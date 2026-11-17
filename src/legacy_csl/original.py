"""Original execution kernel with one local MT19937 stream per trial.

This independently implemented path preserves the arithmetic and draw order of
our fixed modernized INTER reference. It never loads or executes legacy source.
The SINGLE path applies the shared kernel to its fixed observed corpus.
"""

from __future__ import annotations

from itertools import permutations
from math import log
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from scipy.stats import invwishart, multivariate_normal
from sklearn.cluster import KMeans

from .attributes import AttributeSampler
from .config import Config
from .data import ATTRIBUTES, ObservationDataset
from .model import (
    CategoryAssignments,
    CslModel,
    CslState,
    FloatArray,
    IntArray,
    _category_offsets,
    _flatten_assignments,
)


class OriginalExecutionError(ValueError):
    """A numerical execution failure already annotated with model context."""


class OriginalCslModel(CslModel):
    """Compatibility learning model, created only by the joint factory.

    Prediction continues to reconstruct the ordinary ``CslModel`` from state.
    """

    rng: np.random.RandomState

    def _error(self, message: str, **context: object) -> OriginalExecutionError:
        details = dict(
            experiment=self.experiment_id,
            seed=self.effective_seed,
            iteration=self.iteration,
            agent=self.agent,
            **context,
        )
        return OriginalExecutionError(
            "original execution: "
            + ", ".join(f"{key}={value}" for key, value in details.items())
            + ": "
            + message
        )

    def _probabilities(self, values: FloatArray, **context: object) -> FloatArray:
        total = np.sum(values)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise self._error("probabilities must be finite and nonnegative", **context)
        if not np.isfinite(total) or total <= 0:
            raise self._error(
                "probabilities have a nonpositive or nonfinite sum", **context
            )
        return values / total

    def _multinomial(self, probabilities: FloatArray, **context: object) -> int:
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or not np.isclose(np.sum(probabilities), 1.0)
        ):
            raise self._error("invalid multinomial probabilities", **context)
        try:
            return list(self.rng.multinomial(1, probabilities)).index(1)
        except ValueError as exc:
            raise self._error(str(exc), **context) from exc

    def word_theta(self) -> FloatArray:
        """Rescale using sequential column sums and original row operations."""
        theta = self.state.theta
        column_sums = [
            sum(theta[category][word] for category in range(len(theta)))
            for word in range(len(self.vocabulary))
        ]
        result = np.array(theta)
        if not np.all(np.isfinite(column_sums)) or np.any(np.asarray(column_sums) <= 0):
            raise self._error(
                "theta column sums must be finite and positive", phase="rescaling"
            )
        for row in result:
            row /= column_sums
            total = np.sum(row)
            if not np.isfinite(total) or total <= 0:
                raise self._error("invalid rescaled theta row", phase="rescaling")
            row /= total
        return result

    def draw_words(self) -> tuple[IntArray, ...]:
        """Draw fresh proposals with the reference's probability-choice API."""
        theta = self.word_theta()
        choices = list(range(len(self.vocabulary)))
        return tuple(
            np.asarray(
                [
                    self.rng.choice(
                        choices,
                        p=theta[
                            self.global_category(
                                attr, self.attended_category(attr, doc_id)
                            )
                        ],
                    )
                    for attr in attributes
                ],
                dtype=np.int64,
            )
            for doc_id, attributes in enumerate(self.state.attributes)
        )

    def step(
        self, words: Sequence[npt.ArrayLike] | npt.ArrayLike | None = None
    ) -> None:
        """Perform z → pi → Gaussian → F → theta with the shared stream."""
        self.iteration += 1
        if words is not None:
            coerced = self._coerce_words(words)
            assert coerced is not None
            self._validate_word_shapes(coerced)
            self.state.words = coerced
        self._sample_z(self.state.theta)
        self._sample_pi()
        self._sample_gaussians()
        self._sample_attributes(self.state.theta)
        self._sample_theta()

    def _sample_z(self, theta: FloatArray) -> None:
        for attr in ATTRIBUTES:
            assignments = self.state.z[attr]
            for doc_id, observations in enumerate(self.dataset.document_arrays(attr)):
                for object_id, observation in enumerate(observations):
                    context = dict(
                        phase="z", attribute=attr, document=doc_id, object=object_id
                    )
                    attended = (
                        attr == "a" or object_id == self.dataset.attentions[doc_id]
                    )
                    if attr == "o":
                        try:
                            weights = np.asarray(
                                [log(value) for value in self.state.pi[attr]]
                            )
                            for category in range(self.category_counts[attr]):
                                for word, word_attr in zip(
                                    self.state.words[doc_id],
                                    self.state.attributes[doc_id],
                                ):
                                    if word_attr == attr and attended:
                                        weights[category] = weights[category] + log(
                                            theta[self.offsets[attr] + category][word]
                                        )
                                weights[category] = weights[
                                    category
                                ] + multivariate_normal.logpdf(
                                    observation,
                                    mean=self.state.means[attr][category],
                                    cov=self.state.covariances[attr][category],
                                )
                            weights = weights - np.max(weights)
                            weights = weights - logsumexp(weights)
                            probabilities = np.exp(weights)
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            raise self._error(str(exc), **context) from exc
                    else:
                        weights = np.array(self.state.pi[attr])
                        try:
                            for category in range(self.category_counts[attr]):
                                for word, word_attr in zip(
                                    self.state.words[doc_id],
                                    self.state.attributes[doc_id],
                                ):
                                    if word_attr == attr and attended:
                                        weights[category] = (
                                            weights[category]
                                            * theta[self.offsets[attr] + category][word]
                                        )
                                weights[category] = weights[
                                    category
                                ] * multivariate_normal.pdf(
                                    observation,
                                    mean=self.state.means[attr][category],
                                    cov=self.state.covariances[attr][category],
                                )
                        except (ValueError, np.linalg.LinAlgError) as exc:
                            raise self._error(str(exc), **context) from exc
                        probabilities = self._probabilities(weights, **context)
                    sampled = self._multinomial(probabilities, **context)
                    if attr == "a":
                        assert isinstance(assignments, np.ndarray)
                        assignments[doc_id] = sampled
                    else:
                        assert isinstance(assignments, tuple)
                        assignments[doc_id][object_id] = sampled

    def _sample_gaussians(self, *, use_heuristics: bool = True) -> None:
        del use_heuristics
        for attr in ATTRIBUTES:
            observations = self.dataset.flatten(attr)
            categories = _flatten_assignments(self.state.z[attr])
            dimension = observations.shape[1]
            prior = self._giw_prior(attr, dimension)
            for category in range(self.category_counts[attr]):
                selected = [
                    np.array(value) for value in observations[categories == category]
                ]
                count = len(selected)
                context = dict(phase="Gaussian", attribute=attr, category=category)
                try:
                    if count:
                        empirical_mean = sum(selected) / float(count)
                        precision = prior.kappa0 + count
                        mean = (
                            prior.kappa0 * prior.m0 + count * empirical_mean
                        ) / precision
                        degrees = prior.nu0 + count
                        scatter = sum(
                            np.dot(
                                np.array([value - empirical_mean]).T,
                                np.array([value - empirical_mean]),
                            )
                            for value in selected
                        )
                        displacement = np.array([empirical_mean - prior.m0])
                        scale = (
                            prior.V0
                            + scatter
                            + (prior.kappa0 * count / precision)
                            * np.dot(displacement.T, displacement)
                        )
                    else:
                        low, high = (-0.5, 0.5) if attr == "p" else (0.0, 1.0)
                        self.state.means[attr][category] = np.asarray(
                            [self.rng.uniform(low, high) for _ in range(dimension)]
                        )
                        scale, degrees = prior.V0, prior.nu0
                    if not np.all(np.isfinite(scale)):
                        raise ValueError("nonfinite posterior scale")
                    np.linalg.cholesky(scale)
                    covariance = np.asarray(
                        invwishart.rvs(df=degrees, scale=scale, random_state=self.rng),
                        dtype=np.float64,
                    ).reshape(dimension, dimension)
                    self.state.covariances[attr][category] = covariance
                    if count:
                        samples = multivariate_normal.rvs(
                            mean=mean,
                            cov=covariance / precision,
                            size=100,
                            random_state=self.rng,
                        )
                        self.state.means[attr][category] = np.mean(
                            np.asarray(samples).reshape(100, dimension), 0
                        )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    raise self._error(str(exc), **context) from exc

    def _sample_attributes(self, theta: FloatArray) -> None:
        for doc_id, words in enumerate(self.state.words):
            frames = list(permutations(ATTRIBUTES, len(words)))
            weights = [1.0 for _ in frames]
            for index, frame in enumerate(frames):
                for word, attr in zip(words, frame):
                    category = self.global_category(
                        attr, self.attended_category(attr, doc_id)
                    )
                    weights[index] = weights[index] * theta[category][word]
            context = dict(phase="F", document=doc_id)
            probabilities = self._probabilities(np.asarray(weights), **context)
            self.state.attributes[doc_id] = frames[
                self._multinomial(probabilities, **context)
            ]

    def _sample_theta(self) -> None:
        self.state.theta = np.asarray(
            [
                sum(self.rng.dirichlet(row, 100)) / 100.0
                for row in self.theta_posterior_parameters()
            ]
        )
        if (
            not np.all(np.isfinite(self.state.theta))
            or np.any(self.state.theta < 0)
            or np.any(np.sum(self.state.theta, axis=1) <= 0)
        ):
            raise self._error(
                "sampled theta must be finite, nonnegative, with positive row sums",
                phase="theta",
            )


def initialize_original_models(
    datasets: Sequence[ObservationDataset],
    config: Config,
    *,
    rng: np.random.RandomState,
    initial_words: Sequence[npt.ArrayLike] | npt.ArrayLike | None = None,
    vocabulary: Sequence[str] | None = None,
    experiment_id: int | None = None,
) -> tuple[OriginalCslModel, ...]:
    """Jointly initialize A/B, or U with observed words, without extra draws.

    Validate every input before consuming the stream. INTER vocabulary coverage
    is checked after its actual word draws, with no retry or added vocabulary.
    """
    if (
        getattr(config.original_heuristics, "execution_profile", "current")
        != "original"
    ):
        raise ValueError(
            "initialize_original_models requires execution_profile='original'"
        )
    if not isinstance(rng, np.random.RandomState):
        raise TypeError("original execution requires numpy.random.RandomState")
    expected = 2 if config.mode == "inter-csl" else 1
    if len(datasets) != expected:
        raise ValueError(
            f"original {config.mode} requires {expected} datasets in A/B or U order"
        )
    if config.mode == "inter-csl" and initial_words is not None:
        raise ValueError("original INTER initializes its internal words independently")
    if config.mode == "single-csl" and initial_words is None:
        raise ValueError("original SINGLE requires its fixed observed word corpus")
    vocabulary = (
        tuple(vocabulary)
        if vocabulary is not None
        else tuple(
            str(index) for index in range(config.hyperparameters.vocabulary_size)
        )
    )
    if len(vocabulary) != config.hyperparameters.vocabulary_size or len(
        set(vocabulary)
    ) != len(vocabulary):
        raise ValueError(
            "original vocabulary must be unique and match hyperparameters.vocabulary_size"
        )
    models: list[OriginalCslModel] = []
    # These assignments and validations never sample; all stochastic work follows.
    for agent, dataset in zip(("A", "B") if expected == 2 else ("U",), datasets):
        model = OriginalCslModel.__new__(OriginalCslModel)
        model.dataset, model.config, model.rng = dataset, config, rng
        model.agent, model.experiment_id, model.iteration = agent, experiment_id, 0
        model.effective_seed = config.run.seed + (experiment_id or 0)
        model.category_counts = dict(
            config.hyperparameters.category_counts.as_mapping()
        )
        model.offsets = _category_offsets(model.category_counts)
        model.num_categories = sum(model.category_counts.values())
        model.vocabulary = vocabulary
        model.attribute_sampler = AttributeSampler(
            ATTRIBUTES, config.attribute_model.exclusivity, rng
        )
        words = model._coerce_words(initial_words)
        if words is not None:
            if any(len(row) != 4 for row in words):
                raise model._error("observed word lengths must be 4", phase="input")
            if any(np.any(row < 0) or np.any(row >= len(vocabulary)) for row in words):
                raise model._error(
                    "observed word IDs must be within vocabulary", phase="input"
                )
            if set(np.concatenate(words).tolist()) != set(range(len(vocabulary))):
                raise model._error(
                    "observed corpus effective vocabulary does not match configured vocabulary",
                    phase="input",
                )
        for attr in ATTRIBUTES:
            if len(dataset.flatten(attr)) < model.category_counts[attr]:
                raise model._error(
                    "fewer observations than categories", phase="input", attribute=attr
                )
        # Temporary placeholders do not consume any random numbers.
        model.state = CslState(
            {}, {}, {}, {}, np.empty((0, 0)), [], words or (), vocabulary
        )
        models.append(model)
    for model in models:
        z: dict[str, CategoryAssignments] = {
            "a": np.asarray(
                [
                    int(rng.uniform(0, model.category_counts["a"]))
                    for _ in range(model.dataset.num_documents)
                ],
                dtype=np.int64,
            )
        }
        for attr in ATTRIBUTES[1:]:
            z[attr] = tuple(
                np.asarray(
                    [
                        int(rng.uniform(0, model.category_counts[attr]))
                        for _ in range(count)
                    ],
                    dtype=np.int64,
                )
                for count in model.dataset.num_objects
            )
        model.state.z = z
        for _ in range(model.dataset.num_documents):
            remaining = list(ATTRIBUTES)
            frame = []
            for position in range(4):
                frame.append(remaining.pop(int(rng.uniform(0, 4 - position))))
            model.state.attributes.append(tuple(frame))
        if expected == 2:
            model.state.words = tuple(
                np.asarray(
                    [rng.choice(list(range(len(vocabulary)))) for _ in range(4)],
                    dtype=np.int64,
                )
                for _ in range(model.dataset.num_documents)
            )
    if expected == 2:
        for model in models:
            missing = sorted(
                set(range(len(vocabulary)))
                - set(np.concatenate(model.state.words).tolist())
            )
            if missing:
                raise model._error(
                    f"initial internal words omit vocabulary IDs {missing}; original column indexing is undefined",
                    phase="initialization",
                )
    for model in models:
        model.state.theta = np.asarray(
            [
                sum(
                    rng.dirichlet(
                        np.array([config.hyperparameters.gamma for _ in vocabulary]),
                        100,
                    )
                )
                / 100.0
                for _ in range(model.num_categories)
            ]
        )
    for model in models:
        for attr in ATTRIBUTES:
            observations = model.dataset.flatten(attr)
            try:
                model.state.means[attr] = (
                    KMeans(
                        n_clusters=model.category_counts[attr],
                        init="k-means++",
                        n_init=10,
                        random_state=rng,
                    )
                    .fit(observations)
                    .cluster_centers_
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise model._error(
                    str(exc), phase="initialization", attribute=attr
                ) from exc
            model.state.covariances[attr] = np.asarray(
                [
                    np.eye(observations.shape[1]) * 0.05
                    for _ in range(model.category_counts[attr])
                ]
            )
    for model in models:
        for attr in ATTRIBUTES:
            betas = rng.beta(1, 1, model.category_counts[attr])
            remaining = np.append(1, np.cumprod(1 - betas[:-1]))
            probabilities = betas * remaining
            model.state.pi[attr] = probabilities / probabilities.sum()
        model._validate_state(model.state)
    return tuple(models)
