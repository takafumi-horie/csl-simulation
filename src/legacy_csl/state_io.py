"""Checkpoint and compatibility CSV serialization.

チェックポイントと互換 CSV のシリアライズ。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import ATTRIBUTES
from .model import CslState


def save_state(state: CslState, directory: Path, agent: str) -> None:
    """Save a native checkpoint and major legacy-compatible CSV files.

    ネイティブチェックポイントと主要 legacy 互換 CSV を保存する。

    Args:
        state: Agent state to save.
            保存するエージェント状態。
        directory: Experiment directory.
            実験ディレクトリ。
        agent: Agent identifier used in filenames.
            ファイル名に使うエージェント識別子。
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "theta": state.theta,
        "words": state.words,
    }
    for attr in ATTRIBUTES:
        payload[f"z_{attr}"] = state.z[attr]
        payload[f"pi_{attr}"] = state.pi[attr]
        payload[f"mean_{attr}"] = state.means[attr]
        payload[f"variance_{attr}"] = state.variances[attr]
    np.savez_compressed(directory / f"state_{agent}.npz", **payload)
    (directory / f"attributes_{agent}.json").write_text(
        json.dumps(state.attributes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_compatibility_csv(state, directory / "csv", agent)


def load_state(directory: Path, agent: str) -> CslState:
    """Load a state written by :func:`save_state`.

    :func:`save_state` が保存した状態を読み込む。

    Args:
        directory: Experiment directory.
            実験ディレクトリ。
        agent: Agent identifier.
            エージェント識別子。

    Returns:
        Reconstructed mutable state owned by the caller.
        呼出側が所有する復元済み可変状態。
    """
    with np.load(directory / f"state_{agent}.npz", allow_pickle=False) as archive:
        z = {attr: archive[f"z_{attr}"].astype(np.int64).copy() for attr in ATTRIBUTES}
        pi = {attr: archive[f"pi_{attr}"].astype(np.float64).copy() for attr in ATTRIBUTES}
        means = {
            attr: archive[f"mean_{attr}"].astype(np.float64).copy()
            for attr in ATTRIBUTES
        }
        variances = {
            attr: archive[f"variance_{attr}"].astype(np.float64).copy()
            for attr in ATTRIBUTES
        }
        theta = archive["theta"].astype(np.float64).copy()
        words = archive["words"].astype(np.int64).copy()
    raw_attributes = json.loads(
        (directory / f"attributes_{agent}.json").read_text(encoding="utf-8")
    )
    attributes = [tuple(sequence) for sequence in raw_attributes]
    return CslState(z, pi, means, variances, theta, attributes, words)


def write_metric_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous metric dictionaries as UTF-8 CSV.

    同じキーを持つ指標辞書を UTF-8 CSV に保存する。

    Args:
        path: Destination CSV path.
            出力 CSV パス。
        rows: Metric rows; empty input writes no file.
            指標行。空の場合はファイルを作らない。
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_compatibility_csv(state: CslState, directory: Path, agent: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for attr in ATTRIBUTES:
        np.savetxt(directory / f"z{attr}_{agent}.csv", state.z[attr], delimiter=",", fmt="%d")
        np.savetxt(directory / f"pi_{attr}_{agent}.csv", state.pi[attr], delimiter=",")
        np.savetxt(directory / f"Mu_{attr}_{agent}.csv", state.means[attr], delimiter=",")
        np.savetxt(
            directory / f"Sig_{attr}_{agent}.csv",
            state.variances[attr],
            delimiter=",",
        )
    np.savetxt(directory / f"theta_{agent}.csv", state.theta, delimiter=",")
    np.savetxt(directory / f"uttered_words_{agent}.csv", state.words, delimiter=",", fmt="%d")
    with (directory / f"F_{agent}.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(state.attributes)
    with (directory / f"W_list_{agent}.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(range(state.theta.shape[1]))
    np.savetxt(
        directory / f"word_length_{agent}.csv",
        np.full(state.words.shape[0], state.words.shape[1]),
        delimiter=",",
        fmt="%d",
    )

