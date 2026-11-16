"""On-disk cache of frozen encoder outputs.

One directory per modality and encoder, ``<features_dir>/<modality>/<encoder>/``. For each split
it holds ``<split>_ids.txt`` (one id per line) and either

- inputs: ``<split>_pooled.npy`` (float16, N x d), optionally ``<split>_tokens.npy``
  (float16, N x L x d) with ``<split>_mask.npy`` (bool, N x L, True for a real token); or
- targets: ``<split>_z.npy`` (float32, N x d, unit norm).

Ids are segment ids, except for film-level text, which uses movie ids and the split ``movies``.
A single ``meta.json`` describes the encoder; each write records its split under ``splits``.
"""

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

META_FILE = "meta.json"
UNIT_NORM_ATOL = 1e-3


@dataclass(frozen=True)
class Features:
    """Encoder outputs for one split, row-aligned with ``ids``."""

    ids: list[str]
    pooled: np.ndarray
    tokens: np.ndarray | None = None
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        check_ids(self.ids)
        n = len(self.ids)
        if self.pooled.ndim != 2 or self.pooled.shape[0] != n:
            raise ValueError(f"pooled must be ({n}, d), got {self.pooled.shape}")
        if (self.tokens is None) != (self.mask is None):
            raise ValueError("tokens and mask go together")
        if self.tokens is not None and self.mask is not None:
            if self.tokens.ndim != 3 or self.tokens.shape[0] != n:
                raise ValueError(f"tokens must be ({n}, L, d), got {self.tokens.shape}")
            if self.mask.dtype != np.bool_ or self.mask.shape != self.tokens.shape[:2]:
                raise ValueError(
                    f"mask must be bool {self.tokens.shape[:2]}, got {self.mask.dtype} "
                    f"{self.mask.shape}"
                )
            if not self.mask.any(axis=1).all():
                raise ValueError("every row needs at least one real token")


def check_ids(ids: Sequence[str]) -> None:
    if not ids:
        raise ValueError("no ids")
    if any(not i or i != i.strip() or any(c.isspace() for c in i) for i in ids):
        raise ValueError("ids must be non-empty and contain no whitespace")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")


def check_unit_norm(z: np.ndarray, name: str = "z") -> None:
    norms = np.linalg.norm(z.astype(np.float32), axis=-1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=UNIT_NORM_ATOL):
        raise ValueError(f"{name} must be L2-normalised along its last axis")


def write_features(
    directory: Path,
    split: str,
    features: Features,
    meta: Mapping[str, Any],
    *,
    dtype: np.dtype | type = np.float16,
    split_info: Mapping[str, Any] | None = None,
) -> None:
    """Write one split of encoder inputs, cast to ``dtype`` (float16 by default).

    ``meta`` describes the encoder and must agree across splits; ``split_info`` (timings,
    counts) is recorded for this split only.
    """
    with np.errstate(over="ignore"):  # overflow becomes inf, refused just below
        arrays = {"pooled": features.pooled.astype(dtype)}
        if features.tokens is not None and features.mask is not None:
            arrays["tokens"] = features.tokens.astype(dtype)
            arrays["mask"] = features.mask
    for name, array in arrays.items():
        if name != "mask" and not np.isfinite(array).all():
            raise ValueError(f"{name} has NaN or inf after casting to {np.dtype(dtype).name}")
    directory.mkdir(parents=True, exist_ok=True)
    _write_ids(directory, split, features.ids)
    for name, array in arrays.items():
        np.save(directory / f"{split}_{name}.npy", array)
    _update_meta(directory, split, len(features.ids), meta, split_info)


def read_features(directory: Path, split: str, *, mmap: bool = True) -> Features:
    """Read one split; with ``mmap`` the arrays stay on disk until sliced."""
    mode = "r" if mmap else None
    tokens_path = directory / f"{split}_tokens.npy"
    has_tokens = tokens_path.exists()
    return Features(
        ids=_read_ids(directory, split),
        pooled=np.load(directory / f"{split}_pooled.npy", mmap_mode=mode),
        tokens=np.load(tokens_path, mmap_mode=mode) if has_tokens else None,
        mask=np.load(directory / f"{split}_mask.npy", mmap_mode=mode) if has_tokens else None,
    )


def write_targets(
    directory: Path,
    split: str,
    ids: Sequence[str],
    z: np.ndarray,
    meta: Mapping[str, Any],
    *,
    split_info: Mapping[str, Any] | None = None,
) -> None:
    """Write one split of unit-norm target vectors in float32."""
    check_ids(ids)
    if z.ndim != 2 or z.shape[0] != len(ids):
        raise ValueError(f"z must be ({len(ids)}, d), got {z.shape}")
    z = z.astype(np.float32)
    check_unit_norm(z)
    directory.mkdir(parents=True, exist_ok=True)
    _write_ids(directory, split, ids)
    np.save(directory / f"{split}_z.npy", z)
    _update_meta(directory, split, len(ids), meta, split_info)


def read_targets(directory: Path, split: str) -> tuple[list[str], np.ndarray]:
    ids = _read_ids(directory, split)
    z = np.load(directory / f"{split}_z.npy")
    if z.shape[0] != len(ids):
        raise ValueError(f"{directory}: {len(ids)} ids but {z.shape[0]} targets")
    return ids, z


def read_meta(directory: Path) -> dict[str, Any]:
    return json.loads((directory / META_FILE).read_text())


def git_commit() -> str | None:
    """Commit of the code that produced a cache, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_ids(directory: Path, split: str, ids: Sequence[str]) -> None:
    (directory / f"{split}_ids.txt").write_text("\n".join(ids) + "\n")


def _read_ids(directory: Path, split: str) -> list[str]:
    return (directory / f"{split}_ids.txt").read_text().split()


def _update_meta(
    directory: Path,
    split: str,
    n: int,
    meta: Mapping[str, Any],
    split_info: Mapping[str, Any] | None = None,
) -> None:
    """Merge ``meta`` into meta.json; a key already set to another value is an error.

    This stops two encoders, or two preprocessing settings, from sharing a directory.
    """
    path = directory / META_FILE
    current: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    for key, value in json.loads(json.dumps(dict(meta))).items():  # tuples -> lists, as on disk
        if key == "splits":
            raise ValueError("'splits' is written by the cache itself")
        if key in current and current[key] != value:
            raise ValueError(f"{path}: {key!r} is {current[key]!r}, refusing {value!r}")
        current[key] = value
    info = json.loads(json.dumps(dict(split_info or {})))
    current.setdefault("splits", {})[split] = info | {"n": n, "git_commit": git_commit()}
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
