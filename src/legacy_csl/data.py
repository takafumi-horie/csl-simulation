"""Input adapters for processed legacy CSL observations.

処理済み legacy CSL 観測の入力アダプター。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt

ATTRIBUTES = ("a", "p", "o", "c")
_JOINT_MINS = {
    "head": np.array([-40, -70, -55, -35, -50, 0], dtype=np.float64),
    "right_arm": np.array(
        [-95, 0, -37, 15.5, -90, -90, -20, 0, 9.6, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.float64,
    ),
    "torso": np.array([-50, -30, -10], dtype=np.float64),
}
_JOINT_MAXS = {
    "head": np.array([30, 60, 55, 15, 50, 90], dtype=np.float64),
    "right_arm": np.array(
        [10, 160.8, 80, 106, 90, 0, 40, 60, 90, 90, 180, 90, 180, 90, 180, 270],
        dtype=np.float64,
    ),
    "torso": np.array([50, 30, 70], dtype=np.float64),
}


@dataclass(frozen=True, slots=True)
class ObservationDataset:
    """Store one attended observation vector per document and modality.

    文書・モダリティごとに注視対象の観測ベクトルを保持する。

    Args:
        observations: Attribute-to-matrix mapping with shape ``(D, dimension)``.
            ``(D, dimension)`` 形状の属性別行列。
        true_categories: Optional ground-truth category vectors.
            任意の正解カテゴリベクトル。
    """

    observations: Mapping[str, npt.NDArray[np.float64]]
    true_categories: Mapping[str, npt.NDArray[np.int64]] | None = None

    def __post_init__(self) -> None:
        """Validate shapes and ownership.

        形状と所有権を検証する。
        """
        missing = set(ATTRIBUTES) - set(self.observations)
        if missing:
            raise ValueError(f"missing observation attributes: {sorted(missing)}")
        sizes = {np.asarray(value).shape[0] for value in self.observations.values()}
        if len(sizes) != 1:
            raise ValueError("all observation matrices must have the same row count")

    @property
    def num_documents(self) -> int:
        """Return the number of documents.

        文書数を返す。

        Returns:
            Number of documents.
            文書数。
        """
        return int(np.asarray(self.observations[ATTRIBUTES[0]]).shape[0])


def load_dataset(data_root: Path, agent: str) -> ObservationDataset:
    """Load an agent dataset from compact NPZ or legacy datadump files.

    小型 NPZ または legacy datadump からエージェントデータを読む。

    Args:
        data_root: Root containing ``agentU``, ``agentA``, and ``agentB``.
            ``agentU``、``agentA``、``agentB`` を含むルート。
        agent: Agent identifier (``U``, ``A``, or ``B``).
            エージェント識別子。

    Returns:
        Loaded observations and optional labels.
        読み込んだ観測と任意の正解ラベル。

    Raises:
        FileNotFoundError: If neither supported representation exists.
            対応する入力形式が存在しない場合。
    """
    compact_path = data_root / f"agent{agent}.npz"
    if compact_path.exists():
        return _load_compact_dataset(compact_path)
    agent_root = data_root / f"agent{agent}"
    if agent_root.exists():
        return _load_legacy_dataset(agent_root, data_root, agent)
    raise FileNotFoundError(
        f"expected {compact_path} or legacy directory {agent_root}"
    )


def load_uni_words(data_root: Path, num_documents: int) -> npt.NDArray[np.int64]:
    """Load fixed Uni-CSL words from compact JSON or legacy CSV.

    Uni-CSL の固定単語列を小型 JSON または legacy CSV から読む。

    Args:
        data_root: Dataset root.
            データセットのルート。
        num_documents: Required number of documents.
            必要な文書数。

    Returns:
        Integer word matrix with shape ``(D, L)``.
        ``(D, L)`` の整数単語行列。
    """
    json_path = data_root / "uni_words.json"
    if json_path.exists():
        words = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        csv_path = data_root / "agentU" / "testss(001-040)" / "testss_words.csv"
        with csv_path.open(encoding="utf-8") as stream:
            words = [row for row in csv.reader(stream) if row]
    array = np.asarray(words[:num_documents])
    if array.ndim != 2 or array.shape[0] != num_documents:
        raise ValueError("Uni word input must be a rectangular D x L matrix")
    try:
        return array.astype(np.int64)
    except ValueError:
        vocabulary = {word: i for i, word in enumerate(sorted(set(array.flat)))}
        return np.vectorize(vocabulary.__getitem__, otypes=[np.int64])(array)


def write_compact_fixture(
    data_root: Path,
    *,
    num_documents: int = 8,
    word_length: int = 4,
    vocabulary_size: int = 8,
    seed: int = 7,
) -> None:
    """Create a deterministic, distributable synthetic fixture.

    決定論的で配布可能な合成 fixture を作成する。

    Args:
        data_root: Destination directory.
            出力先ディレクトリ。
        num_documents: Number of documents per agent.
            エージェントごとの文書数。
        word_length: Words per document.
            文書ごとの単語数。
        vocabulary_size: Vocabulary size.
            語彙数。
        seed: Random seed.
            乱数シード。
    """
    rng = np.random.default_rng(seed)
    data_root.mkdir(parents=True, exist_ok=True)
    latent = rng.integers(0, 3, size=(num_documents, len(ATTRIBUTES)))
    dims = {"a": 3, "p": 2, "o": 4, "c": 3}
    for agent_id, agent in enumerate(("U", "A", "B")):
        payload: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] = {}
        for attr_id, attr in enumerate(ATTRIBUTES):
            means = rng.normal(size=(3, dims[attr]))
            payload[attr] = means[latent[:, attr_id]] + rng.normal(
                scale=0.08 + 0.02 * agent_id,
                size=(num_documents, dims[attr]),
            )
            payload[f"z_{attr}"] = latent[:, attr_id]
        np.savez(data_root / f"agent{agent}.npz", **payload)
    latent_word_sources = latent[:, np.arange(word_length) % len(ATTRIBUTES)]
    words = np.mod(
        latent_word_sources + np.arange(word_length), vocabulary_size
    ).tolist()
    (data_root / "uni_words.json").write_text(
        json.dumps(words, indent=2), encoding="utf-8"
    )


def _load_compact_dataset(path: Path) -> ObservationDataset:
    with np.load(path, allow_pickle=False) as archive:
        observations = {
            attr: np.asarray(archive[attr], dtype=np.float64) for attr in ATTRIBUTES
        }
        truth = {
            attr: np.asarray(archive[f"z_{attr}"], dtype=np.int64)
            for attr in ATTRIBUTES
            if f"z_{attr}" in archive
        }
    return ObservationDataset(observations, truth or None)


def _load_legacy_dataset(
    agent_root: Path, data_root: Path, agent: str
) -> ObservationDataset:
    doc_dirs = sorted(path for path in agent_root.glob("testss[0-9][0-9][0-9]") if path.is_dir())
    if agent in {"A", "B"} and len(doc_dirs) >= 40:
        doc_dirs = doc_dirs[-30:]
    observations = {attr: [] for attr in ATTRIBUTES}
    attended_ids: list[int] = []
    for doc_dir in doc_dirs:
        row = _read_legacy_document(doc_dir)
        attended_ids.append(
            int(_numeric_rows(doc_dir / "target_object.txt")[0][0]) - 1
        )
        for attr in ATTRIBUTES:
            observations[attr].append(row[attr])
    matrices = {
        attr: np.asarray(values, dtype=np.float64)
        for attr, values in observations.items()
    }
    truth = _load_legacy_truth(data_root, agent, attended_ids)
    return ObservationDataset(matrices, truth)


def _read_legacy_document(doc_dir: Path) -> dict[str, npt.NDArray[np.float64]]:
    object_center = _numeric_rows(doc_dir / "object_center.txt")
    target = _numeric_rows(doc_dir / "target_object.txt")
    num_objects = int(object_center[0][0])
    attended = int(target[0][0]) - 1
    positions = np.asarray([row[1:3] for row in object_center[2 : 2 + num_objects]])
    action_parts = _read_action(doc_dir / "action.csv")
    relative = np.asarray(target[2][:3]) - np.asarray(target[1][:3])
    random_move = np.asarray([target[4][0] if len(target) > 4 else 0.0])
    normalized_joints = []
    for key in ("head", "right_arm", "torso"):
        value = action_parts[key]
        normalized_joints.append(
            (value - _JOINT_MINS[key]) / (_JOINT_MAXS[key] - _JOINT_MINS[key])
        )
    action = np.concatenate([relative, random_move, *normalized_joints])[:29]
    object_id = attended + 1
    object_feature = np.asarray(
        _numeric_rows(doc_dir / "image" / f"object_CNN_PCA_40_{object_id:02}.csv")[0],
        dtype=np.float64,
    )
    color_feature = np.asarray(
        _numeric_rows(doc_dir / "image" / f"object_RGB_BoF_40_{object_id:02}.csv")[0],
        dtype=np.float64,
    )
    return {"a": action, "p": positions[attended], "o": object_feature, "c": color_feature}


def _numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            values = [value.strip() for value in row if value.strip()]
            if values:
                rows.append([float(value) for value in values])
    return rows


def _read_action(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result: dict[str, npt.NDArray[np.float64]] = {}
    index = 0
    while index < len(lines):
        if not lines[index] or "," not in lines[index]:
            index += 1
            continue
        name = lines[index].split(",", maxsplit=1)[0]
        index += 1
        while index < len(lines) and not lines[index]:
            index += 1
        if index < len(lines):
            result[name] = np.asarray(
                [float(value) for value in lines[index].split(",") if value.strip()],
                dtype=np.float64,
            )
        index += 1
    return result


def _load_legacy_truth(
    data_root: Path, agent: str, attended_ids: list[int]
) -> Mapping[str, npt.NDArray[np.int64]] | None:
    roots = [data_root / f"agent{agent}" / "testss(001-040)", data_root / "testss(001-040)"]
    for root in roots:
        files = [root / f"z{attr}_C.csv" for attr in ATTRIBUTES]
        if not all(path.exists() for path in files):
            continue
        truth: dict[str, npt.NDArray[np.int64]] = {}
        for attr, path in zip(ATTRIBUTES, files):
            rows = _numeric_rows(path)
            if len(rows) == 1:
                values = rows[0][-len(attended_ids) :]
            else:
                selected_rows = rows[-len(attended_ids) :]
                values = [
                    row[attended_id]
                    for row, attended_id in zip(selected_rows, attended_ids)
                ]
            truth[attr] = np.asarray(values, dtype=np.int64)
        return truth
    return None
