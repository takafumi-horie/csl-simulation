"""Hand-written synthetic reference for original execution tests.

This module deliberately imports no legacy_csl code. Observations, parameters,
assignments, row offsets, all-object counts, GIW posterior and every sampling
loop are computed here. Only the fixed NumPy/SciPy/KMeans primitives are shared.
It needs neither the original source tree nor external datasets.
"""

from copy import deepcopy
from itertools import permutations
from math import log

import numpy as np
from scipy.special import logsumexp
from scipy.stats import invwishart, multivariate_normal
from sklearn.cluster import KMeans

MODALITIES = ("a", "p", "o", "c")
SETTINGS = {
    "categories": {"a": 2, "p": 3, "o": 4, "c": 5},
    "dimensions": {"a": 3, "p": 2, "o": 4, "c": 2},
    "concentration": {"a": 0.21, "p": 0.32, "o": 0.43, "c": 0.54},
    "mean": {"a": 0.15, "p": -0.1, "o": 0.25, "c": 0.05},
    "precision": {"a": 0.03, "p": 0.04, "o": 0.05, "c": 0.06},
    "scale": {"a": 0.07, "p": 0.08, "o": 0.09, "c": 0.10},
    "df_offset": {"a": 2.0, "p": 3.0, "o": 4.0, "c": 5.0},
    "vocabulary": 5,
    "gamma": 0.3,
}


def synthetic_inputs():
    """Two distinct agents, ragged objects and nonzero attended indices."""
    result = []
    for agent in range(2):
        observations = {key: [] for key in MODALITIES}
        attention = []
        for document in range(8):
            count = 2 + document % 2
            attention.append((document + 1 + agent) % count)
            for attribute_index, attribute in enumerate(MODALITIES):
                dimension = SETTINGS["dimensions"][attribute]
                rows = []
                for obj in range(1 if attribute == "a" else count):
                    rows.append(
                        [
                            0.3
                            + 0.19
                            * np.sin(
                                (document + 1) * (coordinate + 1) + obj * 1.7 + agent
                            )
                            + document * 0.025
                            + obj * 0.08
                            + attribute_index * 0.04
                            for coordinate in range(dimension)
                        ]
                    )
                observations[attribute].append(np.asarray(rows, dtype=float))
        result.append((observations, np.asarray(attention)))
    return result


def row_number(state, attention, attribute, document, settings=SETTINGS):
    offset = 0
    for modality in MODALITIES:
        if modality == attribute:
            break
        offset += settings["categories"][modality]
    local = state["z"][attribute][document][
        0 if attribute == "a" else attention[document]
    ]
    return offset + local


def initialize(inputs, rng, settings=SETTINGS):
    states = []
    for observations, _ in inputs:
        state = {
            "z": {},
            "frames": [],
            "words": [],
            "pi": {},
            "means": {},
            "covariances": {},
        }
        for attribute in MODALITIES:
            state["z"][attribute] = [
                [int(rng.uniform(0, settings["categories"][attribute])) for _ in rows]
                for rows in observations[attribute]
            ]
        for _ in observations["a"]:
            remaining = ["a", "p", "o", "c"]
            frame = []
            for token in range(4):
                picked = int(rng.uniform(0, 4 - token))
                frame.append(remaining[picked])
                del remaining[picked]
            state["frames"].append(tuple(frame))
        for _ in observations["a"]:
            state["words"].append(
                [rng.choice(list(range(settings["vocabulary"]))) for _ in range(4)]
            )
        states.append(state)
    for state in states:
        state["theta"] = np.array(
            [
                sum(
                    rng.dirichlet(
                        np.array([settings["gamma"]] * settings["vocabulary"]), 100
                    )
                )
                / 100.0
                for _ in range(sum(settings["categories"].values()))
            ]
        )
    for state, (observations, _) in zip(states, inputs):
        for attribute in MODALITIES:
            matrix = np.array(
                [row for document in observations[attribute] for row in document]
            )
            state["means"][attribute] = (
                KMeans(
                    n_clusters=settings["categories"][attribute],
                    init="k-means++",
                    n_init=10,
                    random_state=rng,
                )
                .fit(matrix)
                .cluster_centers_
            )
            state["covariances"][attribute] = np.array(
                [
                    np.eye(settings["dimensions"][attribute]) * 0.05
                    for _ in range(settings["categories"][attribute])
                ]
            )
    for state in states:
        for attribute in MODALITIES:
            breaks = rng.beta(1, 1, settings["categories"][attribute])
            weights = breaks * np.append(1, np.cumprod(1 - breaks[:-1]))
            state["pi"][attribute] = weights / weights.sum()
    return states


def corrected(theta):
    column_sums = [sum(row[word] for row in theta) for word in range(theta.shape[1])]
    result = np.array(theta)
    for row in result:
        row[:] = row / column_sums
        row[:] = row / np.sum(row)
    return result


def proposal(state, attention, rng, settings=SETTINGS, *, rescale=True):
    theta = corrected(state["theta"]) if rescale else state["theta"]
    return [
        [
            rng.choice(
                list(range(settings["vocabulary"])),
                p=theta[row_number(state, attention, attr, document, settings)],
            )
            for attr in frame
        ]
        for document, frame in enumerate(state["frames"])
    ]


def accept(state, attention, suggested, rng, *, rescale=True):
    theta = corrected(state["theta"]) if rescale else state["theta"]
    accepted = deepcopy(state["words"])
    for document, frame in enumerate(state["frames"]):
        for token, attr in enumerate(frame):
            row = row_number(state, attention, attr, document)
            old, new = accepted[document][token], suggested[document][token]
            ratio = min(1, theta[row][new] / theta[row][old])
            uniform = rng.rand()
            if ratio >= uniform:
                accepted[document][token] = new
    return accepted


def sample_z(state, observations, attention, rng, settings=SETTINGS):
    offset = 0
    for attr in MODALITIES:
        for document, rows in enumerate(observations[attr]):
            for obj, row in enumerate(rows):
                active = attr == "a" or obj == attention[document]
                if attr == "o":
                    weights = np.array([log(p) for p in state["pi"][attr]])
                    for category in range(settings["categories"][attr]):
                        for token in range(4):
                            if active and state["frames"][document][token] == attr:
                                weights[category] += log(
                                    state["theta"][offset + category][
                                        state["words"][document][token]
                                    ]
                                )
                        weights[category] += multivariate_normal.logpdf(
                            row,
                            mean=state["means"][attr][category],
                            cov=state["covariances"][attr][category],
                        )
                    weights = weights - np.max(weights)
                    weights = np.exp(weights - logsumexp(weights))
                else:
                    weights = np.array(state["pi"][attr])
                    for category in range(settings["categories"][attr]):
                        for token in range(4):
                            if active and state["frames"][document][token] == attr:
                                weights[category] *= state["theta"][offset + category][
                                    state["words"][document][token]
                                ]
                        weights[category] *= multivariate_normal.pdf(
                            row,
                            mean=state["means"][attr][category],
                            cov=state["covariances"][attr][category],
                        )
                    weights = weights / np.sum(weights)
                state["z"][attr][document][obj] = list(
                    rng.multinomial(1, weights)
                ).index(1)
        offset += settings["categories"][attr]


def mixture_parameters(state, settings=SETTINGS):
    parameters = {}
    for attr in MODALITIES:
        counts = [0] * settings["categories"][attr]
        for document in state["z"][attr]:
            for category in document:
                counts[category] += 1
        parameters[attr] = np.array(
            [count + settings["concentration"][attr] for count in counts]
        )
    return parameters


def sample_pi(state, rng, settings=SETTINGS):
    for attr, parameters in mixture_parameters(state, settings).items():
        state["pi"][attr] = rng.dirichlet(parameters)


def sample_gaussians(state, observations, rng, settings=SETTINGS):
    for attr in MODALITIES:
        dimension = settings["dimensions"][attr]
        k0 = settings["precision"][attr]
        m0 = np.full(dimension, settings["mean"][attr])
        v0 = np.eye(dimension) * settings["scale"][attr]
        nu0 = dimension + settings["df_offset"][attr]
        for category in range(settings["categories"][attr]):
            selected = []
            for document, rows in enumerate(observations[attr]):
                for obj, row in enumerate(rows):
                    if state["z"][attr][document][obj] == category:
                        selected.append(np.array(row))
            count = len(selected)
            if count:
                xbar = sum(selected) / float(count)
                precision = k0 + count
                mean = (k0 * m0 + count * xbar) / precision
                scale = (
                    v0
                    + sum(
                        [
                            np.dot(np.array([row - xbar]).T, np.array([row - xbar]))
                            for row in selected
                        ]
                    )
                    + (k0 * count / precision)
                    * np.dot(np.array([xbar - m0]).T, np.array([xbar - m0]))
                )
                covariance = np.asarray(
                    invwishart.rvs(df=nu0 + count, scale=scale, random_state=rng)
                ).reshape(dimension, dimension)
                draws = multivariate_normal.rvs(
                    mean=mean, cov=covariance / precision, size=100, random_state=rng
                )
                state["means"][attr][category] = np.mean(
                    np.asarray(draws).reshape(100, dimension), 0
                )
            else:
                lower, upper = (-0.5, 0.5) if attr == "p" else (0, 1)
                state["means"][attr][category] = np.array(
                    [rng.uniform(lower, upper) for _ in range(dimension)]
                )
                covariance = np.asarray(
                    invwishart.rvs(df=nu0, scale=v0, random_state=rng)
                ).reshape(dimension, dimension)
            state["covariances"][attr][category] = covariance


def sample_frames(state, attention, rng, settings=SETTINGS):
    for document in range(len(attention)):
        frames = list(permutations(["a", "p", "o", "c"], 4))
        probabilities = [1.0] * len(frames)
        for candidate, frame in enumerate(frames):
            for token, attr in enumerate(frame):
                row = row_number(state, attention, attr, document, settings)
                probabilities[candidate] *= state["theta"][row][
                    state["words"][document][token]
                ]
        probabilities = probabilities / np.sum(probabilities)
        state["frames"][document] = frames[
            list(rng.multinomial(1, probabilities)).index(1)
        ]


def theta_parameters(state, attention, settings=SETTINGS):
    parameters = [
        np.array([settings["gamma"]] * settings["vocabulary"])
        for _ in range(sum(settings["categories"].values()))
    ]
    for document, words in enumerate(state["words"]):
        for token, word in enumerate(words):
            attr = state["frames"][document][token]
            row = row_number(state, attention, attr, document, settings)
            parameters[row][word] = parameters[row][word] + 1
    return parameters


def sample_theta(state, attention, rng, settings=SETTINGS):
    state["theta"] = np.array(
        [
            sum(rng.dirichlet(parameters, 100)) / 100.0
            for parameters in theta_parameters(state, attention, settings)
        ]
    )


def step(state, observations, attention, rng, settings=SETTINGS):
    sample_z(state, observations, attention, rng, settings)
    sample_pi(state, rng, settings)
    sample_gaussians(state, observations, rng, settings)
    sample_frames(state, attention, rng, settings)
    sample_theta(state, attention, rng, settings)
