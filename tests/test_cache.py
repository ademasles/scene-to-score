import json
from pathlib import Path

import numpy as np
import pytest

from scene_to_score.features.cache import (
    Features,
    read_features,
    read_meta,
    read_targets,
    write_features,
    write_targets,
)

META = {"encoder": "dinov2-large", "frames": 16}


def _features(n: int = 4, length: int = 3, d: int = 8) -> Features:
    rng = np.random.default_rng(0)
    mask = np.ones((n, length), dtype=bool)
    mask[0, 1:] = False
    return Features(
        ids=[f"1_{i}_0_1000@0" for i in range(n)],
        pooled=rng.normal(size=(n, d)).astype(np.float32),
        tokens=rng.normal(size=(n, length, d)).astype(np.float32),
        mask=mask,
    )


def _unit(n: int, d: int = 8) -> np.ndarray:
    z = np.random.default_rng(1).normal(size=(n, d))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def test_features_round_trip_as_float16(tmp_path: Path) -> None:
    features = _features()
    write_features(tmp_path, "train", features, META)
    back = read_features(tmp_path, "train")
    assert back.ids == features.ids
    assert back.pooled.dtype == np.float16
    np.testing.assert_allclose(back.pooled, features.pooled, atol=1e-2)
    assert back.tokens is not None and back.mask is not None
    assert back.tokens.shape == (4, 3, 8)
    np.testing.assert_array_equal(back.mask, features.mask)


def test_pooled_only_features_have_no_tokens(tmp_path: Path) -> None:
    features = _features()
    write_features(tmp_path, "movies", Features(features.ids, features.pooled), META)
    back = read_features(tmp_path, "movies", mmap=False)
    assert back.tokens is None and back.mask is None


def test_meta_records_each_split_and_refuses_conflicts(tmp_path: Path) -> None:
    write_features(tmp_path, "train", _features(), META)
    write_features(tmp_path, "test", _features(n=2), META)
    meta = read_meta(tmp_path)
    assert meta["encoder"] == "dinov2-large"
    assert {split: info["n"] for split, info in meta["splits"].items()} == {"train": 4, "test": 2}
    with pytest.raises(ValueError, match="encoder"):
        write_features(tmp_path, "test", _features(), {"encoder": "vjepa2-large"})


def test_meta_accepts_tuples_written_as_lists(tmp_path: Path) -> None:
    write_features(tmp_path, "train", _features(), {"size": (224, 224)})
    write_features(tmp_path, "test", _features(), {"size": (224, 224)})
    assert json.loads((tmp_path / "meta.json").read_text())["size"] == [224, 224]


def test_float16_overflow_is_refused(tmp_path: Path) -> None:
    features = _features()
    pooled = features.pooled.copy()
    pooled[0, 0] = 1e6
    with pytest.raises(ValueError, match="inf"):
        write_features(tmp_path, "train", Features(features.ids, pooled), META)
    assert not (tmp_path / "train_ids.txt").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"ids": ["a", "a", "b", "c"]}, "unique"),
        ({"ids": ["a b", "c", "d", "e"]}, "whitespace"),
        ({"pooled": np.zeros((3, 8))}, "pooled"),
        ({"tokens": None}, "together"),
        ({"mask": np.ones((4, 3), dtype=np.int8)}, "bool"),
        ({"mask": np.zeros((4, 3), dtype=bool)}, "real token"),
    ],
)
def test_invalid_features_are_rejected(change: dict, message: str) -> None:
    base = _features()
    fields = {"ids": base.ids, "pooled": base.pooled, "tokens": base.tokens, "mask": base.mask}
    with pytest.raises(ValueError, match=message):
        Features(**(fields | change))


def test_targets_round_trip_in_float32(tmp_path: Path) -> None:
    ids = ["a", "b", "c"]
    write_targets(tmp_path, "train", ids, _unit(3), {"encoder": "clap-audio"})
    back_ids, z = read_targets(tmp_path, "train")
    assert back_ids == ids
    assert z.dtype == np.float32


def test_targets_must_be_unit_norm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="normalised"):
        write_targets(tmp_path, "train", ["a", "b"], 2 * _unit(2), {})
