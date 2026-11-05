from pathlib import Path

import numpy as np
import pytest

from scene_to_score.predictions import (
    Predictions,
    load_predictions,
    prediction_path,
    save_predictions,
)


def _unit(*shape: int) -> np.ndarray:
    z = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def test_prediction_path_follows_the_contract() -> None:
    path = prediction_path(Path("outputs/preds"), "cross_attn", 2, "test", "video_only")
    assert path == Path("outputs/preds/cross_attn/seed2/test_video_only.npz")


@pytest.mark.parametrize(
    ("method", "seed", "split", "condition"),
    [
        ("CrossAttn", 0, "test", "both"),
        ("late", 0, "train", "both"),
        ("late", -1, "val", "both"),
        ("late", 0, "val", "no year"),
    ],
)
def test_prediction_path_rejects_bad_names(
    method: str, seed: int, split: str, condition: str
) -> None:
    with pytest.raises(ValueError):
        prediction_path(Path("p"), method, seed, split, condition)


@pytest.mark.parametrize("shape", [(3, 8), (3, 4, 8)])
def test_predictions_round_trip(tmp_path: Path, shape: tuple[int, ...]) -> None:
    predictions = Predictions(["1_0_0_1000@0", "1_1_0_1000@0", "2_0_0_1000@0"], _unit(*shape))
    path = tmp_path / "late" / "seed0" / "val_both.npz"
    save_predictions(path, predictions)
    back = load_predictions(path)
    assert back.segment_ids == predictions.segment_ids
    np.testing.assert_array_equal(back.z_hat, predictions.z_hat)
    assert back.n_hypotheses == (1 if len(shape) == 2 else 4)


@pytest.mark.parametrize(
    ("z_hat", "message"),
    [
        (_unit(2, 8), "z_hat must be"),
        (_unit(3, 8).astype(np.float64), "float32"),
        (2 * _unit(3, 8), "normalised"),
    ],
)
def test_invalid_predictions_are_rejected(z_hat: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Predictions(["a", "b", "c"], z_hat)
