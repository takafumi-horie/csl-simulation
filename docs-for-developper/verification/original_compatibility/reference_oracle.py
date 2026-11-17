"""Verification-only translation of the independently supplied Python 2 source.

No production sampling code is imported here. Original files are supplied by
the caller and hash-checked. The caller may save translated text for inspection.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import re
import warnings
from pathlib import Path

import numpy as np

SOURCE_HASHES = {
    "mhr_CSL_learn_py2_notactile.py":
        "1380a1fb3c8e91140cf9ba5fc68d811aa629f47e85630eb9116f135a430b3e77",
    "__init__.py":
        "c59c78233ab6e7a0877a23bebb06faabc7a80e54ef0bd1c436e4a705ae870610",
    "rescaling_term.py":
        "a31a09ed5fc96c3055c8f955ad1bdfd9dc9365ce4234ca30ed4df16983f278bd",
}
ATTRS = ("a", "p", "o", "c")


def checked_sources(root: Path) -> dict[str, str]:
    result = {}
    for name, expected in SOURCE_HASHES.items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"unexpected original source: {name}")
        result[name] = data.decode("utf-8")
    return result


def original_inputs(root: Path, data_root: Path) -> tuple:
    """Run the original reader; discard its diagnostic file writes."""
    sources = checked_sources(root)
    source = sources["mhr_CSL_learn_py2_notactile.py"]
    reader = source[source.index("def data_read("):source.index("\n#def para_save")]
    reader = reader.replace("xrange(", "range(")
    reader = reader.replace("len(itemList)/12", "len(itemList)//12")

    def read_or_capture(path, mode="r"):
        if mode == "r":
            return builtins.open(path, mode)
        if mode == "w":
            return io.StringIO()
        raise ValueError(mode)

    namespace = {"np": np, "open": read_or_capture}
    exec(compile(sources["__init__.py"], "original_config", "exec"), namespace)
    namespace.update(D=30, datafolder=str(data_root) + "/",
                     datafolder1=str(data_root / "agentA") + "/",
                     datafolder2=str(data_root / "agentB") + "/", filename="audit")
    exec(compile(reader, "original_reader", "exec"), namespace)
    return namespace["data_read"]("testss", "audit", 11, 40)


def _insert_once(source: str, anchor: str, insertion: str) -> str:
    if source.count(anchor) != 1:
        raise ValueError(f"ambiguous instrumentation anchor: {anchor!r}")
    return source.replace(anchor, insertion + anchor, 1)


def translate(root: Path, seed: int) -> tuple[str, dict]:
    sources = checked_sources(root)
    source = sources["mhr_CSL_learn_py2_notactile.py"]
    source = source.split("if __name__ == '__main__':")[0]
    replacements = {
        "from __init__ import *": "",
        "from rescaling_term import get_rescaled_theta": "",
        "len(itemList)/12": "len(itemList)//12",
        "np.random.seed()": f"np.random.seed({seed})",
        "sp.misc.logsumexp": "sp.special.logsumexp",
        "init='k-means++')": "init='k-means++', n_init=10)",
        "list(cw_A)": "sorted(cw_A)",
        "list(cw_B)": "sorted(cw_B)",
    }
    counts = {}
    for before, after in replacements.items():
        counts[before] = source.count(before)
        source = source.replace(before, after)
    source, batched = re.subn(
        r"\[multivariate_normal\.rvs\(mean=mN, cov=(Sig_[apoc]_[AB]\[k\])/kN\)"
        r" for i in xrange\(100\)\]",
        r"multivariate_normal.rvs(mean=mN, cov=\1/kN, size=100)", source,
    )
    assert batched == 8
    source = _insert_once(source, "      ###初期値を保存",
                          "      snapshot('initial', 0, locals())\n")
    for agent in ("A", "B"):
        anchor = ("        ## replicate #########################\n"
                  "        #MHによる取捨選択\n        for d in xrange(D):\n"
                  f"            for n in xrange(N_{agent}[d]):")
        source = _insert_once(source, anchor,
                              f"        snapshot('proposal_{agent}', iter+1, locals())\n")
        # Each end marker follows a unique listener-specific assignment.
        preceding = f"                    accept_num_{agent}[iter] += 1\n\n"
        if source.count(preceding) != 1:
            raise ValueError("acceptance anchor missing")
        source = source.replace(preceding,
            preceding + f"        snapshot('accepted_{agent}', iter+1, locals())\n", 1)
        assignment = (f"        theta_{agent} = [sum(dirichlet(temp[i],100))/100.0"
                      " for i in xrange(L)]")
        start = source.index(assignment)
        end = source.index("\n", start)
        source = source[:end] + (
            f"\n        snapshot('updated_{agent}', iter+1, locals())") + source[end:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    converted = str(RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
                    .refactor_string(source, "original_reference"))
    return converted, {"replacement_counts": counts, "gaussian_batch_count": batched,
                       "translated_sha256": hashlib.sha256(converted.encode()).hexdigest(),
                       "original_sha256": SOURCE_HASHES}


def _pack(rows) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate(rows).astype(np.int64),
            np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype(np.int64))


def original_payload(local: dict, agent: str) -> dict[str, np.ndarray]:
    result = {"z_a": np.asarray(local[f"za_{agent}"], dtype=np.int64).copy(),
              "theta": np.asarray(local[f"theta_{agent}"]).copy()}
    for attr in ATTRS:
        for prefix, native in (("pi", "pi"), ("Mu", "mean"), ("Sig", "covariance")):
            result[f"{native}_{attr}"] = np.asarray(local[f"{prefix}_{attr}_{agent}"]).copy()
        if attr != "a":
            flat, offsets = _pack(local[f"z{attr}_{agent}"])
            result[f"z_{attr}_flat"], result[f"z_{attr}_offsets"] = flat, offsets
    result["words_flat"], result["word_offsets"] = _pack(local[f"w_dn_{agent}"])
    result["attributes"] = np.asarray(local[f"Fd_{agent}"], dtype="U1")
    result["vocabulary"] = np.asarray(local[f"W_list_{agent}"], dtype="U20")
    return result


def rng_payload(rng_state: tuple) -> dict[str, np.ndarray]:
    return {"keys": rng_state[1].copy(),
            "position": np.asarray(rng_state[2], dtype=np.int64),
            "has_gauss": np.asarray(rng_state[3], dtype=np.int64),
            "cached_gaussian": np.asarray(rng_state[4], dtype=np.float64)}


def run_reference(root: Path, inputs: tuple, seed: int, iterations: int,
                  snapshot, translated_path: Path | None = None) -> dict:
    """Capture every boundary while preserving the process-global RNG."""
    converted, manifest = translate(root, seed)
    if translated_path is not None:
        translated_path.parent.mkdir(parents=True, exist_ok=True)
        translated_path.write_text(converted)
    sources = checked_sources(root)
    namespace = {"__name__": "verification_original", "print": lambda *a, **k: None}
    exec(compile(converted, "translated_reference", "exec"), namespace)
    exec(compile(sources["__init__.py"], "original_config", "exec"), namespace)
    helper = sources["rescaling_term.py"].split("def sample_rescaled_theta")[0]
    exec(compile(helper, "original_rescaling", "exec"), namespace)

    def capture(stage, iteration, local):
        payload = {agent: original_payload(local, agent) for agent in ("A", "B")}
        if stage.startswith("proposal"):
            listener = stage[-1]
            payload["proposal"] = {
                "words": np.asarray(local[f"new_w_dn_{listener}_samp_set"], dtype=np.int64)}
        payload["rng"] = rng_payload(np.random.get_state())
        snapshot(stage, iteration, payload)

    namespace.update(D=30, num_iter=iterations, para_save=lambda *a: None, snapshot=capture)
    before = np.random.get_state()
    try:
        namespace["simulate"]("unused/", "testss", "verification", 11, 40, *inputs)
    finally:
        np.random.set_state(before)
    return manifest
