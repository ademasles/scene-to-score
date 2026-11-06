"""On-disk format of model predictions, the only input of the evaluation.

``outputs/preds/<method>/seed<k>/<split>_<condition>.npz`` holds ``segment_ids`` and ``z_hat``
(float32, N x d, or N x K x d for a multi-hypothesis head), L2-normalised along the last axis.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scene_to_score.features.cache import check_ids, check_unit_norm

SPLITS = ("val", "test")
# Conditions named in the plan; ablations may add others with the same spelling rules.
CONDITIONS = ("both", "video_only", "text_only", "text_shuffled", "gray")
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class Predictions:
    segment_ids: list[str]
    z_hat: np.ndarray

    def __post_init__(self) -> None:
        check_ids(self.segment_ids)
        if self.z_hat.ndim not in (2, 3) or self.z_hat.shape[0] != len(self.segment_ids):
            raise ValueError(
                f"z_hat must be ({len(self.segment_ids)}, d) or (N, K, d), got {self.z_hat.shape}"
            )
        if self.z_hat.dtype != np.float32:
            raise ValueError(f"z_hat must be float32, got {self.z_hat.dtype}")
        check_unit_norm(self.z_hat, "z_hat")

    @property
    def n_hypotheses(self) -> int:
        return 1 if self.z_hat.ndim == 2 else self.z_hat.shape[1]


def prediction_path(root: Path, method: str, seed: int, split: str, condition: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    for name in (method, condition):
        if not _NAME.fullmatch(name):
            raise ValueError(f"{name!r}: use lowercase letters, digits, '_' and '-'")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return root / method / f"seed{seed}" / f"{split}_{condition}.npz"


def save_predictions(path: Path, predictions: Predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, segment_ids=np.array(predictions.segment_ids), z_hat=predictions.z_hat)


def load_predictions(path: Path) -> Predictions:
    with np.load(path, allow_pickle=False) as data:
        return Predictions(segment_ids=data["segment_ids"].tolist(), z_hat=data["z_hat"])
