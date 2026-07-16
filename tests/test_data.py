"""Tests for ragged legacy-compatible input adapters.

legacy互換のragged入力アダプタのテスト。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from legacy_csl.data import (
    LEGACY_TRUTH_DIRECTORY,
    ObservationDataset,
    load_dataset,
    load_single_words,
    validate_legacy_truth_layout,
    write_compact_fixture,
)


def test_observation_dataset_preserves_all_objects_and_attended_views() -> None:
    """The dataset keeps ragged objects and derives flat/attended matrices.

    データセットがragged物体を保持し、平坦・注視行列を導出する。
    """
    observations = {
        "a": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "p": (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])),
        "o": (np.array([[1.0], [2.0]]), np.array([[3.0]])),
        "c": (np.array([[4.0], [5.0]]), np.array([[6.0]])),
    }
    truth = {
        "a": np.array([7, 8]),
        "p": (np.array([1, 2]), np.array([3])),
        "o": (np.array([4, 5]), np.array([6])),
        "c": (np.array([7, 8]), np.array([9])),
    }
    dataset = ObservationDataset(observations, np.array([1, 0]), truth)

    np.testing.assert_array_equal(dataset.num_objects, [2, 1])
    np.testing.assert_allclose(dataset.flatten("p"), [[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(dataset.attended("p"), [[3, 4], [5, 6]])
    np.testing.assert_array_equal(dataset.attended_categories("p"), [2, 3])
    np.testing.assert_array_equal(dataset.attended_categories("a"), [7, 8])
    assert not np.asarray(dataset.observations["a"]).flags.writeable
    assert not dataset.observations["p"][0].flags.writeable  # type: ignore[index]


def test_observation_dataset_rejects_inconsistent_object_counts() -> None:
    """Object modalities must agree on each document's object count.

    物体属性は文書ごとの物体数が一致しなければならない。
    """
    observations = {
        "a": np.ones((1, 2)),
        "p": (np.ones((2, 2)),),
        "o": (np.ones((1, 2)),),
        "c": (np.ones((2, 2)),),
    }
    with pytest.raises(ValueError, match="identical object counts"):
        ObservationDataset(observations, np.array([0]))


def test_observation_dataset_rejects_partial_truth() -> None:
    """Ground truth is either absent or complete for every modality.

    正解カテゴリは無し、または全モダリティ分を必要とする。
    """
    observations = {
        "a": np.ones((1, 2)),
        "p": (np.ones((1, 2)),),
        "o": (np.ones((1, 2)),),
        "c": (np.ones((1, 2)),),
    }
    with pytest.raises(ValueError, match="every attribute"):
        ObservationDataset(
            observations,
            np.array([0]),
            {"a": np.array([0])},
        )


def test_old_compact_two_dimensional_data_wraps_each_document(tmp_path: Path) -> None:
    """Old compact matrices remain readable as one-object documents.

    旧compact行列は1物体文書として読み込める。
    """
    np.savez(
        tmp_path / "agentU.npz",
        a=np.arange(6).reshape(2, 3),
        p=np.arange(4).reshape(2, 2),
        o=np.arange(8).reshape(2, 4),
        c=np.arange(6).reshape(2, 3),
        z_a=np.array([0, 1]),
        z_p=np.array([1, 2]),
        z_o=np.array([2, 0]),
        z_c=np.array([0, 2]),
    )

    dataset = load_dataset(tmp_path, "U")

    np.testing.assert_array_equal(dataset.num_objects, [1, 1])
    np.testing.assert_array_equal(dataset.attentions, [0, 0])
    assert dataset.observations["p"][0].shape == (1, 2)  # type: ignore[index]
    assert dataset.true_categories is not None
    assert dataset.true_categories["p"][0].shape == (1,)  # type: ignore[index]


def test_compact_fixture_contains_multiple_objects_and_full_truth(tmp_path: Path) -> None:
    """The distributable fixture exercises ragged multi-object inference.

    配布fixtureがraggedな複数物体推論を通る。
    """
    write_compact_fixture(tmp_path, num_documents=5, seed=11)

    dataset = load_dataset(tmp_path, "U")

    np.testing.assert_array_equal(dataset.num_objects, [2, 3, 2, 3, 2])
    assert dataset.flatten("o").shape[0] == 12
    assert dataset.attended("c").shape == (5, 3)
    assert dataset.true_categories is not None
    assert [
        len(values) for values in dataset.true_categories["o"]  # type: ignore[union-attr]
    ] == [2, 3, 2, 3, 2]
    for agent in ("U", "A", "B"):
        truth_root = tmp_path / f"agent{agent}" / LEGACY_TRUTH_DIRECTORY
        assert {path.name for path in truth_root.iterdir()} == {
            "za_C.csv",
            "zp_C.csv",
            "zo_C.csv",
            "zc_C.csv",
        }


def test_single_words_are_ragged_and_encoded_in_first_seen_order(tmp_path: Path) -> None:
    """Words retain variable lengths and use encounter-order vocabulary ids.

    単語は可変長を保持し、出現順語彙IDを使う。
    """
    (tmp_path / "single_words.json").write_text(
        json.dumps([["z", "a", "z"], ["b"], ["a", "c"]]), encoding="utf-8"
    )

    corpus = load_single_words(tmp_path, 3)

    assert corpus.vocabulary == ("z", "a", "b", "c")
    assert [value.tolist() for value in corpus.encoded] == [[0, 1, 0], [2], [1, 3]]
    np.testing.assert_array_equal(corpus.lengths, [3, 1, 2])
    with pytest.raises(ValueError, match="ragged"):
        corpus.rectangular()


def test_legacy_loader_keeps_objects_and_applies_original_preprocessing(
    tmp_path: Path,
) -> None:
    """Legacy Single data keeps tactile/object values and old normalization.

    legacy Singleデータが触覚・全物体と旧正規化を保持する。
    """
    agent_root = tmp_path / "agentU"
    _write_legacy_document(agent_root / "testss001", relative_scale=1.0)
    _write_legacy_document(agent_root / "testss002", relative_scale=2.0)
    truth_root = agent_root / LEGACY_TRUTH_DIRECTORY
    truth_root.mkdir(parents=True)
    (truth_root / "za_C.csv").write_text("7,8\n", encoding="utf-8")
    for attr, rows in {
        "p": "1,2\n3,4\n",
        "o": "5,6\n7,8\n",
        "c": "9,10\n11,12\n",
    }.items():
        (truth_root / f"z{attr}_C.csv").write_text(rows, encoding="utf-8")

    dataset = load_dataset(tmp_path, "U")

    assert np.asarray(dataset.observations["a"]).shape == (2, 38)
    np.testing.assert_allclose(np.asarray(dataset.observations["a"])[:, :3], [[0] * 3, [1] * 3])
    np.testing.assert_allclose(
        np.asarray(dataset.observations["a"])[0, 29:], np.arange(1, 10) / 255.0
    )
    np.testing.assert_allclose(
        dataset.observations["p"][0], [[1, 2], [3, 4]]  # type: ignore[index]
    )
    np.testing.assert_allclose(
        dataset.observations["o"][0], [[1, 2], [2, 3]]  # type: ignore[index]
    )
    np.testing.assert_allclose(
        dataset.observations["c"][0],  # type: ignore[index]
        np.asarray([[3, 4], [5, 12]]) / np.asarray([[7], [17]]),
    )
    assert dataset.true_categories is not None
    assert dataset.true_categories["p"][1].tolist() == [3, 4]  # type: ignore[index]
    np.testing.assert_array_equal(dataset.attended_categories("p"), [2, 4])


def test_legacy_truth_is_not_loaded_from_shared_sibling(tmp_path: Path) -> None:
    """Truth is resolved only from inside the requested agent directory.

    正解は指定agentディレクトリの内側からのみ解決する。
    """
    agent_root = tmp_path / "agentU"
    _write_legacy_document(agent_root / "testss001", relative_scale=1.0)
    shared_truth = tmp_path / LEGACY_TRUTH_DIRECTORY
    shared_truth.mkdir(parents=True)
    for attr in ("a", "p", "o", "c"):
        (shared_truth / f"z{attr}_C.csv").write_text("0,0\n", encoding="utf-8")

    dataset = load_dataset(tmp_path, "U")

    assert dataset.true_categories is None


def test_legacy_truth_layout_validation_lists_missing_agent_files(
    tmp_path: Path,
) -> None:
    """Validation requires all four files under the requested agent.

    validationは指定agent配下の4fileすべてを要求する。
    """
    truth_root = tmp_path / "agentA" / LEGACY_TRUTH_DIRECTORY
    truth_root.mkdir(parents=True)
    for attr in ("a", "p", "o"):
        (truth_root / f"z{attr}_C.csv").write_text("0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"agent A.*zc_C\.csv"):
        validate_legacy_truth_layout(tmp_path, "A")

    (truth_root / "zc_C.csv").write_text("0\n", encoding="utf-8")
    validate_legacy_truth_layout(tmp_path, "A")


def test_legacy_inter_action_excludes_tactile(tmp_path: Path) -> None:
    """Legacy Inter agents retain their original 29-dimensional action.

    legacy Interエージェントは元の29次元Actionを保持する。
    """
    _write_legacy_document(tmp_path / "agentA" / "testss001", relative_scale=1.0)

    dataset = load_dataset(tmp_path, "A")

    assert np.asarray(dataset.observations["a"]).shape == (1, 29)
    np.testing.assert_allclose(np.asarray(dataset.observations["a"])[0, :3], [0.5] * 3)


def _write_legacy_document(path: Path, *, relative_scale: float) -> None:
    path.mkdir(parents=True)
    (path / "image").mkdir()
    (path / "object_center.txt").write_text(
        "2\n0\n2,3,4\n1,1,2\n", encoding="utf-8"
    )
    (path / "target_object.txt").write_text(
        "2\n1,2,3\n"
        f"{1 + relative_scale},{2 + 2 * relative_scale},{3 + 3 * relative_scale}\n"
        "0\n0.25\n",
        encoding="utf-8",
    )
    tactile = np.repeat(np.arange(1, 10, dtype=np.float64), 12)
    sections = {
        "head": np.zeros(6),
        "left_arm": np.zeros(16),
        "right_arm": np.zeros(16),
        "torso": np.zeros(3),
        "tactile": tactile,
    }
    action_lines: list[str] = []
    for name, values in sections.items():
        action_lines.extend(
            [f"{name},{len(values)}", ",".join(map(str, values.tolist())), ""]
        )
    (path / "action.csv").write_text("\n".join(action_lines), encoding="utf-8")
    for object_id, (object_values, color_values) in enumerate(
        [([1, 2], [3, 4]), ([2, 3], [5, 12])], start=1
    ):
        (path / "image" / f"object_CNN_PCA_40_{object_id:02}.csv").write_text(
            ",".join(map(str, object_values)), encoding="utf-8"
        )
        (path / "image" / f"object_RGB_BoF_40_{object_id:02}.csv").write_text(
            ",".join(map(str, color_values)), encoding="utf-8"
        )
