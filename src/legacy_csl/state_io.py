"""Checkpoint and legacy-compatible CSV serialization.

チェックポイントとlegacy互換CSVのシリアライズ。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import ATTRIBUTES
from .model import CslState


def save_state(state: CslState, directory: Path, agent: str) -> None:
    """Save a lossless native checkpoint and major compatibility CSVs.

    欠損のないネイティブcheckpointと主要互換CSVを保存する。

    Args:
        state: Agent state to save.
            保存するエージェント状態。
        directory: Experiment directory.
            実験ディレクトリ。
        agent: Agent identifier used in filenames.
            ファイル名に使うエージェント識別子。
    """
    directory.mkdir(parents=True, exist_ok=True)
    words_flat, word_offsets = _pack_ragged(state.words)
    payload: dict[str, np.ndarray] = {
        "theta": state.theta,
        "words_flat": words_flat,
        "word_offsets": word_offsets,
        "z_a": np.asarray(state.z["a"], dtype=np.int64),
    }
    for attr in ATTRIBUTES[1:]:
        assignments = state.z[attr]
        assert isinstance(assignments, tuple)
        flat, offsets = _pack_ragged(assignments)
        payload[f"z_{attr}_flat"] = flat
        payload[f"z_{attr}_offsets"] = offsets
    for attr in ATTRIBUTES:
        payload[f"pi_{attr}"] = state.pi[attr]
        payload[f"mean_{attr}"] = state.means[attr]
        payload[f"covariance_{attr}"] = state.covariances[attr]
    np.savez_compressed(directory / f"state_{agent}.npz", **payload)
    metadata = {
        "attributes": state.attributes,
        "vocabulary": state.vocabulary,
    }
    (directory / f"metadata_{agent}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_state_csvs(state, directory / "csv", agent)


def write_state_csvs(state: CslState, directory: Path, agent: str) -> None:
    """Write the complete legacy-compatible CSV snapshot for one agent.

    1 agent分のlegacy互換CSV snapshot一式を保存する。

    Args:
        state: Agent state to serialize.
            保存するエージェント状態。
        directory: Directory that directly receives the CSV files.
            CSVファイルを直接保存するディレクトリ。
        agent: Agent identifier used in filenames.
            ファイル名に使うエージェント識別子。
    """
    directory.mkdir(parents=True, exist_ok=True)
    for attr in ATTRIBUTES:
        assignments = state.z[attr]
        rows = (
            [[int(value)] for value in assignments]
            if isinstance(assignments, np.ndarray)
            else [value.astype(int).tolist() for value in assignments]
        )
        _write_rows(directory / f"z{attr}_{agent}.csv", rows)
        np.savetxt(
            directory / f"pi_{attr}_{agent}.csv", state.pi[attr], delimiter=","
        )
        np.savetxt(
            directory / f"Mu_{attr}_{agent}.csv", state.means[attr], delimiter=","
        )
        _write_covariance_blocks(
            directory / f"Sig_{attr}_{agent}.csv", state.covariances[attr]
        )
    np.savetxt(directory / f"theta_{agent}.csv", state.theta, delimiter=",")
    _write_rows(
        directory / f"uttered_words_{agent}.csv",
        [row.astype(int).tolist() for row in state.words],
    )
    _write_rows(directory / f"F_{agent}.csv", state.attributes)
    _write_rows(directory / f"W_list_{agent}.csv", [state.vocabulary])
    np.savetxt(
        directory / f"word_length_{agent}.csv",
        np.asarray([len(row) for row in state.words]),
        delimiter=",",
        fmt="%d",
    )


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
        z = {"a": archive["z_a"].astype(np.int64).copy()}
        for attr in ATTRIBUTES[1:]:
            z[attr] = _unpack_ragged(
                archive[f"z_{attr}_flat"], archive[f"z_{attr}_offsets"]
            )
        pi = {
            attr: archive[f"pi_{attr}"].astype(np.float64).copy()
            for attr in ATTRIBUTES
        }
        means = {
            attr: archive[f"mean_{attr}"].astype(np.float64).copy()
            for attr in ATTRIBUTES
        }
        covariances = {
            attr: archive[f"covariance_{attr}"].astype(np.float64).copy()
            for attr in ATTRIBUTES
        }
        theta = archive["theta"].astype(np.float64).copy()
        words = _unpack_ragged(archive["words_flat"], archive["word_offsets"])
    metadata = json.loads(
        (directory / f"metadata_{agent}.json").read_text(encoding="utf-8")
    )
    attributes = [tuple(sequence) for sequence in metadata["attributes"]]
    vocabulary = tuple(map(str, metadata["vocabulary"]))
    return CslState(
        z, pi, means, covariances, theta, attributes, words, vocabulary
    )


def write_metric_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous metric dictionaries as UTF-8 CSV.

    同じキーを持つ指標辞書をUTF-8 CSVへ保存する。

    Args:
        path: Destination CSV path.
            出力CSVパス。
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


def _pack_ragged(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum([len(value) for value in values])]
    )
    flat = np.concatenate(values).astype(np.int64, copy=False)
    return flat, offsets


def _unpack_ragged(flat: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(flat, dtype=np.int64)
    boundaries = np.asarray(offsets, dtype=np.int64)
    return tuple(
        values[boundaries[index] : boundaries[index + 1]].copy()
        for index in range(len(boundaries) - 1)
    )


def _write_rows(path: Path, rows: object) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def _write_covariance_blocks(path: Path, covariances: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for category, covariance in enumerate(covariances):
            writer.writerows(covariance.tolist())
            if category + 1 < len(covariances):
                writer.writerow([])
