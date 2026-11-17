"""Input adapters for processed legacy CSL observations.

処理済み legacy CSL 観測の入力アダプター。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeAlias

import numpy as np
import numpy.typing as npt

ATTRIBUTES = ("a", "p", "o", "c")
OBJECT_ATTRIBUTES = ("p", "o", "c")
LEGACY_TRUTH_DIRECTORY = "testss(001-040)"

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
ObjectObservations: TypeAlias = tuple[FloatArray, ...]
ObjectCategories: TypeAlias = tuple[IntArray, ...]

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
    """Store action and all object observations for one agent.

    1エージェントのActionと全物体の観測を保持する。

    Args:
        observations: ``a`` is a ``(D, dimension)`` matrix. Each other value is
            a tuple of ``(M_d, dimension)`` matrices.
            ``a`` は ``(D, dimension)`` 行列、その他は文書ごとの
            ``(M_d, dimension)`` 行列のtuple。
        attentions: Attended object index for every document.
            各文書の注視物体番号。
        true_categories: Optional full category assignments. Action is a
            ``(D,)`` vector; object modalities use ragged tuples.
            任意の全カテゴリ割当て。Actionは ``(D,)``、物体属性は
            ragged tuple。
    """

    observations: Mapping[str, FloatArray | ObjectObservations]
    attentions: IntArray
    true_categories: Mapping[str, IntArray | ObjectCategories] | None = None

    def __post_init__(self) -> None:
        """Validate shapes and retain read-only owned copies.

        形状を検証し、所有する読取専用コピーを保持する。

        Raises:
            ValueError: If document, object, or feature shapes are inconsistent.
                文書数、物体数、特徴形状が整合しない場合。
        """
        missing = set(ATTRIBUTES) - set(self.observations)
        if missing:
            raise ValueError(f"missing observation attributes: {sorted(missing)}")

        action = _readonly_float_array(self.observations["a"], ndim=2, name="a")
        num_documents = action.shape[0]
        objects: dict[str, ObjectObservations] = {}
        for attr in OBJECT_ATTRIBUTES:
            raw_docs = self.observations[attr]
            if not isinstance(raw_docs, (tuple, list)) or len(raw_docs) != num_documents:
                raise ValueError(
                    f"observations[{attr!r}] must contain one matrix per document"
                )
            docs = tuple(
                _readonly_float_array(value, ndim=2, name=f"{attr}[{doc_id}]")
                for doc_id, value in enumerate(raw_docs)
            )
            if docs and len({value.shape[1] for value in docs}) != 1:
                raise ValueError(f"observations[{attr!r}] feature dimensions differ")
            objects[attr] = docs

        num_objects = np.asarray(
            [value.shape[0] for value in objects["p"]], dtype=np.int64
        )
        for attr in ("o", "c"):
            if not np.array_equal(
                num_objects,
                np.asarray([value.shape[0] for value in objects[attr]]),
            ):
                raise ValueError("p, o, and c must have identical object counts")
        if np.any(num_objects <= 0):
            raise ValueError("every document must contain at least one object")

        attentions = np.asarray(self.attentions, dtype=np.int64)
        if attentions.shape != (num_documents,):
            raise ValueError(f"attentions must have shape ({num_documents},)")
        if np.any(attentions < 0) or np.any(attentions >= num_objects):
            raise ValueError("attention index is outside its document's objects")
        attentions = _readonly_copy(attentions)

        observations: dict[str, FloatArray | ObjectObservations] = {
            "a": action,
            **objects,
        }
        object.__setattr__(self, "observations", MappingProxyType(observations))
        object.__setattr__(self, "attentions", attentions)
        if self.true_categories is not None:
            truth = _validated_truth(
                self.true_categories,
                num_documents=num_documents,
                num_objects=num_objects,
            )
            object.__setattr__(self, "true_categories", MappingProxyType(truth))

    @property
    def num_documents(self) -> int:
        """Return the number of documents.

        文書数を返す。

        Returns:
            Number of documents.
            文書数。
        """
        return int(np.asarray(self.observations["a"]).shape[0])

    @property
    def num_objects(self) -> IntArray:
        """Return a copy of the number of objects in each document.

        各文書の物体数のコピーを返す。

        Returns:
            Per-document object counts.
            文書ごとの物体数。
        """
        docs = self.observations["p"]
        assert isinstance(docs, tuple)
        return np.asarray([value.shape[0] for value in docs], dtype=np.int64)

    def flatten(self, attribute: str) -> FloatArray:
        """Return observations flattened over documents and objects.

        文書と物体を平坦化した観測を返す。

        Args:
            attribute: One of ``a``, ``p``, ``o``, or ``c``.
                ``a``、``p``、``o``、``c`` のいずれか。

        Returns:
            Independent ``(N, dimension)`` observation matrix.
            独立した ``(N, dimension)`` 観測行列。

        Raises:
            KeyError: If the attribute is unknown.
                属性が未知の場合。
        """
        if attribute == "a":
            return np.array(self.observations["a"], dtype=np.float64, copy=True)
        if attribute not in OBJECT_ATTRIBUTES:
            raise KeyError(attribute)
        docs = self.observations[attribute]
        assert isinstance(docs, tuple)
        return np.concatenate(docs, axis=0)

    def document_arrays(self, attribute: str) -> tuple[FloatArray, ...]:
        """Return one observation matrix per document.

        文書ごとに1つの観測行列を返す。

        Args:
            attribute: One of ``a``, ``p``, ``o``, or ``c``.
                ``a``、``p``、``o``、``c`` のいずれか。

        Returns:
            Tuple of ``(M_d, dimension)`` matrices. Action uses ``M_d=1``.
            ``(M_d, dimension)`` 行列のtuple。Actionは ``M_d=1``。
        """
        if attribute == "a":
            action = np.asarray(self.observations["a"], dtype=np.float64)
            return tuple(action[doc_id : doc_id + 1] for doc_id in range(len(action)))
        if attribute not in OBJECT_ATTRIBUTES:
            raise KeyError(attribute)
        docs = self.observations[attribute]
        assert isinstance(docs, tuple)
        return tuple(np.array(value, copy=True) for value in docs)

    def attended(self, attribute: str) -> FloatArray:
        """Return one attended observation per document.

        各文書から1つの注視対象観測を返す。

        Args:
            attribute: One of ``a``, ``p``, ``o``, or ``c``.
                ``a``、``p``、``o``、``c`` のいずれか。

        Returns:
            Independent ``(D, dimension)`` matrix.
            独立した ``(D, dimension)`` 行列。
        """
        if attribute == "a":
            return np.array(self.observations["a"], dtype=np.float64, copy=True)
        if attribute not in OBJECT_ATTRIBUTES:
            raise KeyError(attribute)
        docs = self.observations[attribute]
        assert isinstance(docs, tuple)
        return np.asarray(
            [value[int(attention)] for value, attention in zip(docs, self.attentions)],
            dtype=np.float64,
        )

    def attended_categories(self, attribute: str) -> IntArray | None:
        """Return attended ground-truth categories when available.

        利用可能な場合に注視対象の正解カテゴリを返す。

        Args:
            attribute: One of ``a``, ``p``, ``o``, or ``c``.
                ``a``、``p``、``o``、``c`` のいずれか。

        Returns:
            Category vector, or ``None`` when truth is unavailable.
            カテゴリベクトル。正解が無い場合は ``None``。
        """
        if self.true_categories is None or attribute not in self.true_categories:
            return None
        values = self.true_categories[attribute]
        if attribute == "a":
            return np.array(values, dtype=np.int64, copy=True)
        assert isinstance(values, tuple)
        return np.asarray(
            [value[int(attention)] for value, attention in zip(values, self.attentions)],
            dtype=np.int64,
        )


@dataclass(frozen=True, slots=True)
class WordCorpus:
    """Store ragged encoded utterances and their first-seen vocabulary.

    raggedな符号化済み発話と初出順語彙を保持する。

    Args:
        encoded: One integer word-id vector per document.
            文書ごとの整数単語IDベクトル。
        vocabulary: Original tokens in first-encounter order.
            初出順の元トークン。
    """

    encoded: tuple[IntArray, ...]
    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate encoded ids and retain read-only copies.

        符号化IDを検証し、読取専用コピーを保持する。
        """
        if not self.encoded:
            raise ValueError("word corpus must contain at least one document")
        encoded: list[IntArray] = []
        for doc_id, value in enumerate(self.encoded):
            array = np.asarray(value, dtype=np.int64)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"encoded[{doc_id}] must be a non-empty vector")
            if np.any(array < 0) or np.any(array >= len(self.vocabulary)):
                raise ValueError(f"encoded[{doc_id}] contains an unknown word id")
            encoded.append(_readonly_copy(array))
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must not contain duplicates")
        object.__setattr__(self, "encoded", tuple(encoded))
        object.__setattr__(self, "vocabulary", tuple(map(str, self.vocabulary)))

    @property
    def lengths(self) -> IntArray:
        """Return the number of words per document.

        文書ごとの単語数を返す。
        """
        return np.asarray([len(value) for value in self.encoded], dtype=np.int64)

    def rectangular(self) -> IntArray:
        """Return a rectangular copy when all utterances have equal length.

        全発話が同じ長さの場合に長方形コピーを返す。

        Returns:
            ``(D, L)`` encoded word matrix.
            ``(D, L)`` 符号化単語行列。

        Raises:
            ValueError: If utterance lengths differ.
                発話長が異なる場合。
        """
        if len(set(self.lengths.tolist())) != 1:
            raise ValueError("word corpus is ragged")
        return np.stack(self.encoded)


def load_dataset(data_root: Path, agent: str) -> ObservationDataset:
    """Load an agent dataset from compact NPZ or legacy datadump files.

    小型 NPZ または legacy datadump からエージェントデータを読む。
    """
    compact_path = data_root / f"agent{agent}.npz"
    if compact_path.exists():
        return _load_compact_dataset(compact_path)
    agent_root = data_root / f"agent{agent}"
    if agent_root.exists():
        return _load_legacy_dataset(agent_root, data_root, agent)
    raise FileNotFoundError(f"expected {compact_path} or legacy directory {agent_root}")


def load_datasets(
    data_root: Path,
    agents: tuple[str, ...],
    *,
    execution_profile: str = "current",
) -> tuple[ObservationDataset, ...]:
    """Select the same processed observations for every CLI workflow.

    Original INTER requires raw legacy inputs because both actions use B's
    raw relative-coordinate range. Compact files cannot recover that range.
    """
    if execution_profile != "original" or agents != ("A", "B"):
        return tuple(load_dataset(data_root, agent) for agent in agents)
    for agent in agents:
        root = data_root / f"agent{agent}"
        if not root.is_dir() or not _legacy_document_dirs(root, agent):
            raise ValueError(
                "original INTER preprocessing requires raw legacy datadump "
                f"documents for agent {agent}; compact NPZ inputs do not retain "
                "the raw action coordinates needed for B-range normalization"
            )
    datasets = tuple(
        _load_legacy_dataset(
            data_root / f"agent{agent}", data_root, agent, normalize_action=False
        )
        for agent in agents
    )
    raw_b = np.asarray(datasets[1].observations["a"])[:, :3]
    minimum = raw_b.min(axis=0)
    extent = raw_b.max(axis=0) - minimum
    if np.any(extent <= 0.0):
        columns = np.flatnonzero(extent <= 0.0).tolist()
        raise ValueError(
            "original INTER B action normalization has constant columns "
            f"{columns}; the original reader division is undefined"
        )
    normalized: list[ObservationDataset] = []
    for dataset in datasets:
        action = np.array(dataset.observations["a"], copy=True)
        action[:, :3] = (action[:, :3] - minimum) / extent
        normalized.append(
            ObservationDataset(
                {**dataset.observations, "a": action},
                dataset.attentions,
                dataset.true_categories,
            )
        )
    return tuple(normalized)


def dataset_input_files(
    data_root: Path, agent: str, *, execution_profile: str = "current"
) -> tuple[Path, ...]:
    """Return every source file consumed by the selected observation reader."""
    compact = data_root / f"agent{agent}.npz"
    if compact.exists() and not (
        execution_profile == "original" and agent in {"A", "B"}
    ):
        return (compact,)
    agent_root = data_root / f"agent{agent}"
    paths: list[Path] = []
    for document in _legacy_document_dirs(agent_root, agent):
        paths.extend(document / name for name in (
            "object_center.txt", "target_object.txt", "action.csv"
        ))
        count = int(_numeric_rows(document / "object_center.txt")[0][0])
        paths.extend(
            document / "image" / f"{prefix}_{object_id:02}.csv"
            for object_id in range(1, count + 1)
            for prefix in ("object_CNN_PCA_40", "object_RGB_BoF_40")
        )
    paths.extend(
        agent_root / LEGACY_TRUTH_DIRECTORY / f"z{attr}_C.csv"
        for attr in ATTRIBUTES
        if (agent_root / LEGACY_TRUTH_DIRECTORY / f"z{attr}_C.csv").is_file()
    )
    return tuple(paths)


def single_words_input_file(data_root: Path) -> Path:
    """Return the corpus source selected by the existing SINGLE reader."""
    for name in ("single_words.json", "uni_words.json"):
        path = data_root / name
        if path.exists():
            return path
    return data_root / "agentU" / LEGACY_TRUTH_DIRECTORY / "testss_words.csv"


def validate_legacy_truth_layout(data_root: Path, agent: str) -> None:
    """Require every truth CSV in the canonical per-agent directory.

    canonicalなagent別ディレクトリ内の全正解CSVを必須検証する。

    Args:
        data_root: Parent of all agent directories.
            全agentディレクトリの親。
        agent: Agent identifier such as ``U``, ``A``, or ``B``.
            ``U``、``A``、``B`` などのagent識別子。

    Raises:
        ValueError: If the truth directory or any required CSV is missing.
            正解ディレクトリまたは必須CSVが不足している場合。
    """
    truth_root = data_root / f"agent{agent}" / LEGACY_TRUTH_DIRECTORY
    required = tuple(f"z{attr}_C.csv" for attr in ATTRIBUTES)
    missing = [name for name in required if not (truth_root / name).is_file()]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"legacy ground truth for agent {agent} must be under "
            f"{truth_root}; missing: {missing_text}"
        )


def load_single_words(data_root: Path, num_documents: int) -> WordCorpus:
    """Load and first-seen encode fixed Single-CSL utterances.

    Single-CSLの固定発話を読み、初出順で符号化する。

    Args:
        data_root: Dataset root.
            データセットのルート。
        num_documents: Required number of documents.
            必要な文書数。

    Returns:
        Ragged encoded corpus and vocabulary.
        raggedな符号化済みコーパスと語彙。
    """
    json_path = data_root / "single_words.json"
    legacy_json_path = data_root / "uni_words.json"
    if json_path.exists():
        raw_words = json.loads(json_path.read_text(encoding="utf-8"))
    elif legacy_json_path.exists():
        raw_words = json.loads(legacy_json_path.read_text(encoding="utf-8"))
    else:
        csv_path = data_root / "agentU" / "testss(001-040)" / "testss_words.csv"
        with csv_path.open(encoding="utf-8") as stream:
            raw_words = [
                [token for token in row if token != ""]
                for row in csv.reader(stream)
                if any(token != "" for token in row)
            ]
    if not isinstance(raw_words, list) or len(raw_words) < num_documents:
        raise ValueError(f"Single-CSL words must contain {num_documents} documents")

    vocabulary: list[str] = []
    indexes: dict[str, int] = {}
    encoded: list[IntArray] = []
    for doc_id, raw_sequence in enumerate(raw_words[:num_documents]):
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError(f"Single-CSL words[{doc_id}] must be a non-empty sequence")
        sequence: list[int] = []
        for raw_word in raw_sequence:
            word = str(raw_word)
            if word not in indexes:
                indexes[word] = len(vocabulary)
                vocabulary.append(word)
            sequence.append(indexes[word])
        encoded.append(np.asarray(sequence, dtype=np.int64))
    return WordCorpus(tuple(encoded), tuple(vocabulary))


def write_compact_fixture(
    data_root: Path,
    *,
    num_documents: int = 12,
    word_length: int = 4,
    vocabulary_size: int = 8,
    seed: int = 7,
) -> None:
    """Create a deterministic multi-object compact fixture.

    決定論的な複数物体compact fixtureを作成する。

    The fixture embeds truth in each NPZ and also exports the same truth under
    every agent's canonical legacy directory.

    fixtureは各NPZに正解を埋め込み、同じ正解を各agentのcanonicalなlegacy
    ディレクトリにも出力する。

    Args:
        data_root: Destination containing agent inputs and per-agent truth.
            agent入力とagent別正解を含む出力先。
        num_documents: Number of mock documents.
            mock文書数。
        word_length: Number of words in each Single-CSL utterance.
            Single-CSLの各発話の単語数。
        vocabulary_size: Number of mock word types.
            mock単語type数。
        seed: Random seed for fixture generation.
            fixture生成用乱数seed。
    """
    rng = np.random.default_rng(seed)
    data_root.mkdir(parents=True, exist_ok=True)
    num_objects = 2 + np.arange(num_documents, dtype=np.int64) % 2
    attentions = np.arange(num_documents, dtype=np.int64) % num_objects
    latent_a = rng.integers(0, 3, size=num_documents)
    latent_objects = {
        attr: tuple(rng.integers(0, 3, size=int(count)) for count in num_objects)
        for attr in OBJECT_ATTRIBUTES
    }
    dims = {"a": 3, "p": 2, "o": 4, "c": 3}
    for agent_id, agent in enumerate(("U", "A", "B")):
        payload: dict[str, npt.NDArray[np.generic]] = {
            "num_objects": num_objects,
            "attentions": attentions,
            "z_a": latent_a,
        }
        action_means = rng.normal(size=(3, dims["a"]))
        payload["a"] = action_means[latent_a] + rng.normal(
            scale=0.08 + 0.02 * agent_id,
            size=(num_documents, dims["a"]),
        )
        for attr in OBJECT_ATTRIBUTES:
            means = rng.normal(size=(3, dims[attr]))
            flat_z = np.concatenate(latent_objects[attr])
            payload[attr] = means[flat_z] + rng.normal(
                scale=0.08 + 0.02 * agent_id,
                size=(int(num_objects.sum()), dims[attr]),
            )
            payload[f"z_{attr}"] = flat_z
        np.savez(data_root / f"agent{agent}.npz", **payload)

    attended_latent = np.column_stack(
        [
            latent_a,
            *(
                np.asarray(
                    [
                        values[int(attention)]
                        for values, attention in zip(
                            latent_objects[attr], attentions
                        )
                    ]
                )
                for attr in OBJECT_ATTRIBUTES
            ),
        ]
    )
    words = np.mod(
        attended_latent[:, np.arange(word_length) % len(ATTRIBUTES)]
        + np.arange(word_length),
        vocabulary_size,
    ).tolist()
    (data_root / "single_words.json").write_text(
        json.dumps(words, indent=2), encoding="utf-8"
    )
    _write_agent_truth_fixtures(data_root, latent_a, latent_objects)


def _load_compact_dataset(path: Path) -> ObservationDataset:
    with np.load(path, allow_pickle=False) as archive:
        action = np.asarray(archive["a"], dtype=np.float64)
        num_documents = action.shape[0]
        if "num_objects" in archive:
            num_objects = np.asarray(archive["num_objects"], dtype=np.int64)
            if num_objects.shape != (num_documents,) or np.any(num_objects <= 0):
                raise ValueError("compact num_objects is invalid")
        else:
            num_objects = np.ones(num_documents, dtype=np.int64)
        attentions = (
            np.asarray(archive["attentions"], dtype=np.int64)
            if "attentions" in archive
            else np.zeros(num_documents, dtype=np.int64)
        )
        observations: dict[str, FloatArray | ObjectObservations] = {"a": action}
        for attr in OBJECT_ATTRIBUTES:
            observations[attr] = _split_compact_rows(archive[attr], num_objects, attr)

        truth: dict[str, IntArray | ObjectCategories] = {}
        if "z_a" in archive:
            truth["a"] = np.asarray(archive["z_a"], dtype=np.int64)
        for attr in OBJECT_ATTRIBUTES:
            key = f"z_{attr}"
            if key in archive:
                truth[attr] = tuple(
                    value.reshape(-1)
                    for value in _split_compact_rows(archive[key], num_objects, key)
                )
    return ObservationDataset(observations, attentions, truth or None)


def _split_compact_rows(
    raw: npt.ArrayLike, num_objects: IntArray, name: str
) -> tuple[npt.NDArray[np.generic], ...]:
    array = np.asarray(raw)
    num_documents = len(num_objects)
    if (
        array.ndim >= 3
        and np.all(num_objects == num_objects[0])
        and array.shape[:2] == (num_documents, int(num_objects[0]))
    ):
        return tuple(array[doc_id] for doc_id in range(num_documents))
    if array.ndim >= 1 and array.shape[0] == int(num_objects.sum()):
        boundaries = np.cumsum(num_objects)[:-1]
        return tuple(np.split(array, boundaries, axis=0))
    if array.ndim >= 1 and array.shape[0] == num_documents and np.all(num_objects == 1):
        return tuple(array[doc_id : doc_id + 1] for doc_id in range(num_documents))
    raise ValueError(f"compact {name} shape does not match num_objects")


def _legacy_document_dirs(agent_root: Path, agent: str) -> list[Path]:
    documents = sorted(
        path for path in agent_root.glob("testss[0-9][0-9][0-9]") if path.is_dir()
    )
    if agent in {"A", "B"} and len(documents) >= 40:
        documents = documents[-30:]
    return documents


def _load_legacy_dataset(
    agent_root: Path, data_root: Path, agent: str, *, normalize_action: bool = True
) -> ObservationDataset:
    doc_dirs = _legacy_document_dirs(agent_root, agent)
    if not doc_dirs:
        raise FileNotFoundError(f"no legacy documents under {agent_root}")

    actions: list[FloatArray] = []
    objects: dict[str, list[FloatArray]] = {attr: [] for attr in OBJECT_ATTRIBUTES}
    attentions: list[int] = []
    for doc_dir in doc_dirs:
        action, document_objects, attention = _read_legacy_document(
            doc_dir, include_tactile=agent == "U"
        )
        actions.append(action)
        attentions.append(attention)
        for attr in OBJECT_ATTRIBUTES:
            objects[attr].append(document_objects[attr])

    action_matrix = np.asarray(actions, dtype=np.float64)
    if normalize_action:
        action_matrix[:, :3] = _minmax_columns(action_matrix[:, :3])
    observations: dict[str, FloatArray | ObjectObservations] = {
        "a": action_matrix,
        **{attr: tuple(values) for attr, values in objects.items()},
    }
    num_objects = np.asarray([len(value) for value in objects["p"]], dtype=np.int64)
    truth = _load_legacy_truth(
        data_root,
        agent,
        num_objects=num_objects,
        num_documents=len(doc_dirs),
    )
    return ObservationDataset(
        observations,
        np.asarray(attentions, dtype=np.int64),
        truth,
    )


def _read_legacy_document(
    doc_dir: Path, *, include_tactile: bool
) -> tuple[FloatArray, dict[str, FloatArray], int]:
    object_center = _numeric_rows(doc_dir / "object_center.txt")
    target = _numeric_rows(doc_dir / "target_object.txt")
    num_objects = int(object_center[0][0])
    attention = int(target[0][0]) - 1
    positions = np.full((num_objects, 2), np.nan, dtype=np.float64)
    for row in object_center[2:]:
        if len(row) < 3:
            continue
        object_index = int(row[0]) - 1
        if 0 <= object_index < num_objects:
            positions[object_index] = row[1:3]
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"invalid object_center data in {doc_dir}")

    action_parts = _read_action(doc_dir / "action.csv")
    relative = np.asarray(target[2][:3], dtype=np.float64) - np.asarray(
        target[1][:3], dtype=np.float64
    )
    random_move = np.asarray([target[4][0] if len(target) > 4 else 0.0])
    normalized_joints = [
        (action_parts[key] - _JOINT_MINS[key]) / (_JOINT_MAXS[key] - _JOINT_MINS[key])
        for key in ("head", "right_arm", "torso")
    ]
    action_values = [relative, random_move, *normalized_joints]
    if include_tactile:
        tactile = _find_tactile(action_parts)
        if len(tactile) < 108:
            raise ValueError(f"Single-CSL tactile data must have at least 108 values: {doc_dir}")
        action_values.append(
            np.asarray(
                [np.mean(tactile[index : index + 12]) / 255.0 for index in range(0, 108, 12)],
                dtype=np.float64,
            )
        )
    action = np.concatenate(action_values)
    expected_dimension = 38 if include_tactile else 29
    if action.shape != (expected_dimension,):
        raise ValueError(f"action in {doc_dir} must have dimension {expected_dimension}")

    object_features: list[FloatArray] = []
    color_features: list[FloatArray] = []
    for object_id in range(1, num_objects + 1):
        object_features.append(
            np.asarray(
                _numeric_rows(
                    doc_dir / "image" / f"object_CNN_PCA_40_{object_id:02}.csv"
                )[0],
                dtype=np.float64,
            )
        )
        color = np.asarray(
            _numeric_rows(
                doc_dir / "image" / f"object_RGB_BoF_40_{object_id:02}.csv"
            )[0],
            dtype=np.float64,
        )
        # The paper normalizes the vector-quantized RGB histogram.  Preserve
        # the original CSL-PGM preprocessing by making its total mass one.
        total = color.sum()
        color_features.append(color / total if total > 0.0 else color)
    return (
        action,
        {
            "p": positions,
            "o": np.asarray(object_features, dtype=np.float64),
            "c": np.asarray(color_features, dtype=np.float64),
        },
        attention,
    )


def _numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            values = [value.strip() for value in row if value.strip()]
            if values:
                rows.append([float(value) for value in values])
    return rows


def _read_action(path: Path) -> dict[str, FloatArray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result: dict[str, FloatArray] = {}
    index = 0
    while index < len(lines):
        if not lines[index] or "," not in lines[index]:
            index += 1
            continue
        name = lines[index].split(",", maxsplit=1)[0].strip()
        index += 1
        while index < len(lines) and not lines[index]:
            index += 1
        if index < len(lines):
            result[name] = np.asarray(
                [float(value) for value in lines[index].split(",") if value.strip()],
                dtype=np.float64,
            )
        index += 1
    for key, expected in _JOINT_MINS.items():
        if key not in result or result[key].shape != expected.shape:
            raise ValueError(f"action file {path} has invalid {key!r} values")
    return result


def _find_tactile(action_parts: Mapping[str, FloatArray]) -> FloatArray:
    for name, values in action_parts.items():
        if "tact" in name.lower() or "skin" in name.lower():
            return values
    candidates = [
        value
        for key, value in action_parts.items()
        if key not in {"head", "left_arm", "right_arm", "torso"}
    ]
    if candidates:
        return max(candidates, key=len)
    return np.asarray([], dtype=np.float64)


def _minmax_columns(values: FloatArray) -> FloatArray:
    minimum = values.min(axis=0)
    extent = values.max(axis=0) - minimum
    result = np.full_like(values, 0.5, dtype=np.float64)
    np.divide(values - minimum, extent, out=result, where=extent > 0.0)
    return result


def _load_legacy_truth(
    data_root: Path,
    agent: str,
    *,
    num_objects: IntArray,
    num_documents: int,
) -> Mapping[str, IntArray | ObjectCategories] | None:
    """Load truth only from the canonical per-agent directory.

    canonicalなagent別ディレクトリからだけ正解を読み込む。

    Args:
        data_root: Parent of all agent directories.
            全agentディレクトリの親。
        agent: Agent identifier such as ``U``, ``A``, or ``B``.
            ``U``、``A``、``B`` などのagent識別子。
        num_objects: Per-document object counts for loaded observations.
            読み込んだ観測の文書別物体数。
        num_documents: Number of trailing truth documents to load.
            読み込む末尾正解文書数。

    Returns:
        Full truth mapping, or ``None`` when any required file is absent.
            全正解mapping。必須fileが1つでも無い場合は ``None``。

    Raises:
        ValueError: If truth rows do not match observations.
            正解行が観測と一致しない場合。
    """
    root = data_root / f"agent{agent}" / LEGACY_TRUTH_DIRECTORY
    files = {attr: root / f"z{attr}_C.csv" for attr in ATTRIBUTES}
    if not all(path.exists() for path in files.values()):
        return None
    action_rows = _numeric_rows(files["a"])
    action_values = np.asarray(
        [value for row in action_rows for value in row][-num_documents:],
        dtype=np.int64,
    )
    if action_values.shape != (num_documents,):
        raise ValueError("legacy action truth does not match document count")
    truth: dict[str, IntArray | ObjectCategories] = {"a": action_values}
    for attr in OBJECT_ATTRIBUTES:
        rows = _numeric_rows(files[attr])
        if len(rows) < num_documents:
            raise ValueError(f"legacy {attr} truth does not match document count")
        selected = rows[-num_documents:]
        values: list[IntArray] = []
        for doc_id, (row, count) in enumerate(zip(selected, num_objects)):
            if len(row) < count:
                raise ValueError(
                    f"legacy {attr} truth row {doc_id} has fewer than {count} objects"
                )
            values.append(np.asarray(row[:count], dtype=np.int64))
        truth[attr] = tuple(values)
    return truth


def _write_agent_truth_fixtures(
    data_root: Path,
    action: IntArray,
    objects: Mapping[str, ObjectCategories],
) -> None:
    """Write mock truth under every agent in legacy CSV form.

    mock正解を各agent配下へlegacy CSV形式で書き出す。

    Args:
        data_root: Parent containing all agent inputs.
            全agent入力を含む親。
        action: One action category per document.
            文書ごとのactionカテゴリ。
        objects: Ragged object categories for each object attribute.
            物体属性ごとのragged物体カテゴリ。
    """
    for agent in ("U", "A", "B"):
        truth_root = data_root / f"agent{agent}" / LEGACY_TRUTH_DIRECTORY
        truth_root.mkdir(parents=True, exist_ok=True)
        _write_csv_rows(truth_root / "za_C.csv", [action.tolist()])
        for attr in OBJECT_ATTRIBUTES:
            _write_csv_rows(
                truth_root / f"z{attr}_C.csv",
                [values.tolist() for values in objects[attr]],
            )


def _write_csv_rows(path: Path, rows: list[list[int]]) -> None:
    """Write integer rows as a legacy-compatible CSV file.

    整数行をlegacy互換CSVとして書き出す。

    Args:
        path: Destination CSV path.
            CSV出力先。
        rows: Integer rows to write.
            書き出す整数行。
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def _validated_truth(
    raw_truth: Mapping[str, IntArray | ObjectCategories],
    *,
    num_documents: int,
    num_objects: IntArray,
) -> dict[str, IntArray | ObjectCategories]:
    missing = set(ATTRIBUTES) - set(raw_truth)
    if missing:
        raise ValueError(
            f"true_categories must contain every attribute; missing {sorted(missing)}"
        )
    truth: dict[str, IntArray | ObjectCategories] = {}
    if "a" in raw_truth:
        action = np.asarray(raw_truth["a"], dtype=np.int64)
        if action.shape != (num_documents,):
            raise ValueError(f"true_categories['a'] must have shape ({num_documents},)")
        truth["a"] = _readonly_copy(action)
    for attr in OBJECT_ATTRIBUTES:
        if attr not in raw_truth:
            continue
        raw_docs = raw_truth[attr]
        if not isinstance(raw_docs, (tuple, list)) or len(raw_docs) != num_documents:
            raise ValueError(f"true_categories[{attr!r}] must be ragged by document")
        docs: list[IntArray] = []
        for doc_id, (value, count) in enumerate(zip(raw_docs, num_objects)):
            array = np.asarray(value, dtype=np.int64)
            if array.shape != (count,):
                raise ValueError(
                    f"true_categories[{attr!r}][{doc_id}] must have shape ({count},)"
                )
            docs.append(_readonly_copy(array))
        truth[attr] = tuple(docs)
    return truth


def _readonly_float_array(raw: object, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return _readonly_copy(array)


def _readonly_copy(array: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result
