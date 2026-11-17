"""Save current-profile trajectories using the package selected by PYTHONPATH.

Run this identical script in separate baseline and feature processes. No original
source, audit adapter, or external data is used.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from legacy_csl.config import (
    AttributeCategoryCounts,
    Config,
    CoreHyperparameters,
    Paths,
    OriginalHeuristics,
    Run,
    WordModel,
)
from legacy_csl.data import load_dataset, load_single_words, write_compact_fixture
from legacy_csl.model import CslModel
from legacy_csl.runner import accept_words
from legacy_csl.state_io import save_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--explicit-current", action="store_true", help="Explicitly set the new profile"
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    data = args.output_root / "data"
    write_compact_fixture(data, num_documents=8)
    conditions = [
        (mode, sampler, rescaling, False)
        for mode in ("single-csl", "inter-csl")
        for sampler, rescaling in (
            ("legacy_importance", "legacy_theta_sum"),
            ("dirichlet", "legacy_theta_sum"),
            ("dirichlet", "none"),
            ("dirichlet", "empirical_unigram"),
        )
    ] + [
        (mode, "dirichlet", "legacy_theta_sum", True)
        for mode in ("single-csl", "inter-csl")
    ]
    for mode, sampler, rescaling, enabled in conditions:
        name = f"{mode}_four_heuristics" if enabled else f"{mode}_{sampler}_{rescaling}"
        config = Config(
            mode=mode,
            run=Run(name=name, seed=21, iterations=4),
            paths=Paths(data_root=data, output_root=args.output_root),
            word_model=WordModel(
                theta_sampler=sampler,
                rescaling=rescaling,
                importance_candidates=8,
            ),
            hyperparameters=CoreHyperparameters(
                category_counts=AttributeCategoryCounts(2, 2, 2, 2),
                vocabulary_size=8,
            ),
        )
        heuristics = (
            {
                "theta_mean_samples": 100,
                "inference_theta": "raw",
                "gaussian_mean_samples": 100,
                "empty_component_mean": "original_uniform",
            }
            if enabled
            else {}
        )
        if args.explicit_current:
            heuristics["execution_profile"] = "current"
        config = replace(config, original_heuristics=OriginalHeuristics(**heuristics))
        seed = np.random.SeedSequence(21).spawn(1)[0]
        if mode == "single-csl":
            dataset = load_dataset(data, "U")
            corpus = load_single_words(data, 8)
            models = {
                "U": CslModel(
                    dataset,
                    config,
                    rng=np.random.default_rng(seed),
                    initial_words=corpus.encoded,
                    vocabulary=corpus.vocabulary,
                )
            }
        else:
            models = {
                agent: CslModel(
                    load_dataset(data, agent),
                    config,
                    rng=np.random.default_rng(child),
                )
                for agent, child in zip(("A", "B"), seed.spawn(2), strict=True)
            }
        proposals = {
            agent: tuple(row.copy() for row in model.state.words)
            for agent, model in models.items()
        }
        for step in range(5):
            target = args.output_root / name / f"step{step}"
            for agent, model in models.items():
                save_state(model.state, target, agent)
                (target / f"rng_{agent}.json").write_text(
                    json.dumps(
                        model.rng.bit_generator.state,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            if step == 4:
                break
            if mode == "single-csl":
                models["U"].step(corpus.encoded)
            else:
                for listener, speaker in (("A", "B"), ("B", "A")):
                    model = models[listener]
                    model.step(accept_words(model, proposals[speaker], "mh"))
                    proposals[listener] = model.draw_words()
    print("Saved 10 current-profile configurations at initialization and after 4 steps")


if __name__ == "__main__":
    main()
