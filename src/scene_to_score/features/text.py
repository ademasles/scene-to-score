"""B1: CLAP text features of the films, one cache folder per text variant.

Each film gets a short text built from ``metadata.csv`` with a fixed template (no title):
``Genres: {genres}. Year: {year}. Synopsis: {imdb_description}``. A missing field disappears
from the text. Four variants (``full``, ``noyear``, ``genres``, ``synopsis``) are encoded with the
text tower of MusicLDM's CLAP, the same model as the music target, so text and music vectors live
in one space.

Per variant, ``${paths.features_dir}/text/clap-<variant>/`` holds (format 1.3, one row per film):

- ``<split>_ids.txt``: one ``movie_id`` per line, ``<split>`` being ``all``;
- ``<split>_pooled.npy``: the final CLAP vector, float16 (N, 512), L2-normalised;
- ``<split>_tokens.npy``: hidden states of the text tower, float16 (N, L, H), before projection;
- ``<split>_mask.npy``: bool (N, L), True for a real token;
- ``texts.jsonl``: the exact text encoded, with its token count and whether it was truncated;
- ``meta.json``: written last, so its presence means the folder is complete.
"""

import ast
import csv
import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, ClapModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from scene_to_score.features.cache import check_unit_norm
from scene_to_score.generation import MODEL_ID, _projected
from scene_to_score.seed import pick_device, seed_everything

log = logging.getLogger(__name__)

TEMPLATE = "Genres: {genres}. Year: {year}. Synopsis: {imdb_description}"
# Fields kept by each variant, in template order.
VARIANTS: dict[str, tuple[str, ...]] = {
    "full": ("genres", "year", "synopsis"),
    "noyear": ("genres", "synopsis"),
    "genres": ("genres",),
    "synopsis": ("synopsis",),
}
_PARTS = {"genres": "Genres: {}.", "year": "Year: {}.", "synopsis": "Synopsis: {}"}
# Values read as "no value" (compared in lower case), never written into a text.
_MISSING = frozenset({"", "none", "nan", "null", "n/a"})
_HIDDEN_STATES_ACCESS = (
    "pooled = ClapModel.get_text_features(...).pooler_output (transformers 5 returns an output "
    "object whose pooler_output is the projected, L2-normalised vector, as in generation.py); "
    "tokens = ClapModel.text_model(...).last_hidden_state, the text tower's states before the "
    "projection"
)


@dataclass(frozen=True)
class Film:
    movie_id: str
    genres: str | None
    year: str | None
    synopsis: str | None


def _text(value: object) -> str | None:
    """Single-spaced string, or None when the value means 'missing'."""
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return None if cleaned.lower() in _MISSING else cleaned


def _genres(value: object) -> str | None:
    """Genres as ``a, b``. Accepts a list literal (``['a', 'b']``) or text kept as it is."""
    cleaned = _text(value)
    if cleaned is None or not cleaned.startswith("["):
        return cleaned
    try:
        items = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return cleaned
    if not isinstance(items, list | tuple):
        return cleaned
    return ", ".join(name for item in items if (name := _text(item))) or None


def _year(value: object) -> str | None:
    """Year as an integer string: ``1977.0`` becomes ``1977``; anything else is kept."""
    cleaned = _text(value)
    if cleaned is None:
        return None
    try:
        return str(int(float(cleaned)))
    except (ValueError, OverflowError):
        return cleaned


def read_films(path: Path, columns: Mapping[str, str]) -> list[Film]:
    """One :class:`Film` per row of ``metadata.csv``, sorted by ``movie_id``.

    ``columns`` maps ``movie_id``, ``genres``, ``year`` and ``synopsis`` to the CSV headers.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [name for name in columns.values() if name not in header]
        if missing:
            raise KeyError(f"{path}: columns {missing} not found; available: {header}")
        rows = list(reader)
    films = []
    for row in rows:
        movie_id = _text(row[columns["movie_id"]])
        if movie_id is None:
            raise ValueError(f"{path}: a row has no {columns['movie_id']!r}")
        films.append(
            Film(
                movie_id=movie_id,
                genres=_genres(row[columns["genres"]]),
                year=_year(row[columns["year"]]),
                synopsis=_text(row[columns["synopsis"]]),
            )
        )
    ids = [film.movie_id for film in films]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate movie ids")
    if all(movie_id.isdigit() for movie_id in ids):
        return sorted(films, key=lambda film: int(film.movie_id))
    return sorted(films, key=lambda film: film.movie_id)


def build_text(film: Film, variant: str) -> str:
    """The text of a film for a variant; a missing field is left out, never written as None."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {sorted(VARIANTS)}")
    parts = []
    for field in VARIANTS[variant]:
        value = getattr(film, field)
        if value is not None:
            parts.append(_PARTS[field].format(value))
    return " ".join(parts)


def text_dir(features_dir: Path, variant: str) -> Path:
    return features_dir / "text" / f"clap-{variant}"


def load_text_encoder(model_id: str, device: torch.device) -> tuple[ClapModel, Any]:
    """CLAP and its tokenizer, exactly the ``text_encoder`` and ``tokenizer`` of MusicLDM."""
    model = ClapModel.from_pretrained(model_id, subfolder="text_encoder", dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    cast(torch.nn.Module, model).to(device).eval()
    return model, tokenizer


def resolve_max_length(tokenizer: Any, max_length: int | None) -> int:
    """``max_length`` if given, else the tokenizer's own limit (which must be a real number)."""
    limit = max_length if max_length is not None else int(tokenizer.model_max_length)
    if not 2 <= limit <= 100_000:
        raise ValueError(f"max_length {limit} is not usable; set max_length explicitly")
    return limit


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> np.ndarray:
    """Token count of each text with special tokens and no truncation."""
    ids = tokenizer(list(texts), truncation=False, verbose=False)["input_ids"]
    return np.array([len(item) for item in ids], dtype=np.int64)


def _apply(model: torch.nn.Module, name: str, **kwargs: Any) -> object:
    method = getattr(model, name, None)
    if not callable(method):
        raise TypeError(f"{type(model).__name__} has no callable {name!r}")
    return method(**kwargs)


@torch.inference_mode()
def encode_batch(
    model: torch.nn.Module, tokenizer: Any, texts: Sequence[str], max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CLAP on a batch: pooled (B, d) float32 unit-norm, hidden (B, Lb, H) float32, mask (B, Lb).

    ``Lb`` is the longest text of the batch (at most ``max_length``); padding is masked out.
    """
    device = next(model.parameters()).device
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pooled = _projected(
        _apply(model, "get_text_features", input_ids=input_ids, attention_mask=attention_mask)
    )
    tower = _apply(model, "text_model", input_ids=input_ids, attention_mask=attention_mask)
    if not isinstance(tower, BaseModelOutputWithPooling) or tower.last_hidden_state is None:
        raise TypeError(f"text_model returned {type(tower).__name__} without last_hidden_state")
    return pooled.cpu(), tower.last_hidden_state.float().cpu(), attention_mask.cpu().bool()


def write_cache(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    ids: Sequence[str],
    out_dir: Path,
    *,
    split: str,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Encode ``texts`` and write ids, pooled, tokens and mask (format 1.3) into ``out_dir``.

    Tokens go straight to a memory-mapped file, so the full (N, L, H) array is never in memory.
    Returns the statistics for ``meta.json``.
    """
    n = len(texts)
    if n == 0 or n != len(ids) or len(set(ids)) != n:
        raise ValueError(f"need as many texts as distinct ids, got {n} texts and {len(ids)} ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    full = token_lengths(tokenizer, texts)
    length = int(min(full.max(), max_length))
    mask_all = np.zeros((n, length), dtype=bool)
    pooled_all: np.ndarray | None = None
    tokens: np.ndarray | None = None
    for start in tqdm(range(0, n, batch_size), desc=out_dir.name, unit="batch"):
        stop = min(start + batch_size, n)
        pooled, hidden, mask = encode_batch(model, tokenizer, texts[start:stop], max_length)
        if pooled_all is None or tokens is None:
            pooled_all = np.zeros((n, pooled.shape[1]), dtype=np.float32)
            tokens = np.lib.format.open_memmap(
                out_dir / f"{split}_tokens.npy",
                mode="w+",
                dtype=np.float16,
                shape=(n, length, hidden.shape[2]),
            )
        width = hidden.shape[1]
        block = hidden.numpy().astype(np.float16)
        if not np.isfinite(block).all():
            raise ValueError(f"hidden states overflow float16 in texts {start}:{stop}")
        pooled_all[start:stop] = pooled.numpy()
        tokens[start:stop, :width] = block  # the rest stays zero: padding
        mask_all[start:stop, :width] = mask.numpy()
    assert pooled_all is not None and tokens is not None
    if not (mask_all.sum(axis=1) == np.minimum(full, max_length)).all():
        raise ValueError("mask lengths disagree with the tokenizer: wrong padding or truncation")
    check_unit_norm(pooled_all, "pooled")
    tokens.flush()
    hidden_size = int(tokens.shape[2])
    del tokens
    np.save(out_dir / f"{split}_pooled.npy", pooled_all.astype(np.float16))
    np.save(out_dir / f"{split}_mask.npy", mask_all)
    (out_dir / f"{split}_ids.txt").write_text("".join(f"{item}\n" for item in ids))
    truncated = full > max_length
    return {
        "n_films": n,
        "pooled_dim": int(pooled_all.shape[1]),
        "hidden_size": hidden_size,
        "L": length,
        "max_length": max_length,
        "padding_side": str(getattr(tokenizer, "padding_side", "unknown")),
        "n_truncated": int(truncated.sum()),
        "share_truncated": float(truncated.mean()),
        "tokens_untruncated_mean": float(full.mean()),
        "tokens_untruncated_max": int(full.max()),
        "truncated": truncated.tolist(),
        "token_counts": np.minimum(full, max_length).tolist(),
    }


def _git_commit() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()


def extract_variant(
    model: torch.nn.Module,
    tokenizer: Any,
    films: Sequence[Film],
    variant: str,
    features_dir: Path,
    *,
    split: str = "all",
    max_length: int = 512,
    batch_size: int = 32,
    model_id: str = MODEL_ID,
    complete: bool = True,
) -> Path:
    """Encode one variant of every film into ``features_dir/text/clap-<variant>/``."""
    texts = [build_text(film, variant) for film in films]
    ids = [film.movie_id for film in films]
    out_dir = text_dir(features_dir, variant)
    start = time.perf_counter()
    stats = write_cache(
        model,
        tokenizer,
        texts,
        ids,
        out_dir,
        split=split,
        max_length=max_length,
        batch_size=batch_size,
    )
    seconds = time.perf_counter() - start
    counts = stats.pop("token_counts")
    truncated = stats.pop("truncated")
    with (out_dir / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for movie_id, text, count, cut in zip(ids, texts, counts, truncated, strict=True):
            record = {"movie_id": movie_id, "text": text, "n_tokens": count, "truncated": cut}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    config = getattr(model, "config", None)
    meta = {
        "variant": variant,
        "split": split,
        "complete": complete,
        "encoder": {
            "model_id": model_id,
            "subfolders": ["text_encoder", "tokenizer"],
            "commit_hash": getattr(config, "_commit_hash", None),
        },
        "template": TEMPLATE,
        "fields": list(VARIANTS[variant]),
        "missing_fields_dropped": True,
        "n_missing": {
            field: sum(getattr(film, field) is None for film in films)
            for field in ("genres", "year", "synopsis")
        },
        "n_empty_texts": sum(not text for text in texts),
        "hidden_states_access": _HIDDEN_STATES_ACCESS,
        "dtype": {"pooled": "float16", "tokens": "float16", "mask": "bool"},
        **stats,
        "seconds_total": seconds,
        "seconds_per_film": seconds / len(films),
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    log.info(
        "%s: %d films, L=%d, %d truncated (%.1f %%), %.1f s",
        variant,
        len(films),
        stats["L"],
        stats["n_truncated"],
        100 * stats["share_truncated"],
        seconds,
    )
    return out_dir


@hydra.main(version_base=None, config_path="../../../configs", config_name="features/text")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    variants = [str(name) for name in cfg.variants]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}, expected some of {sorted(VARIANTS)}")
    columns = {str(key): str(value) for key, value in cfg.columns.items()}
    films = read_films(Path(cfg.metadata_path), columns)
    if cfg.max_films is not None:
        films = films[: cfg.max_films]
    log.info("%d films from %s", len(films), cfg.metadata_path)
    device = pick_device(cfg.device)
    model, tokenizer = load_text_encoder(cfg.model_id, device)
    max_length = resolve_max_length(tokenizer, cfg.max_length)
    for variant in variants:
        out_dir = extract_variant(
            model,
            tokenizer,
            films,
            variant,
            Path(cfg.paths.features_dir),
            split=cfg.split_name,
            max_length=max_length,
            batch_size=cfg.batch_size,
            model_id=cfg.model_id,
            complete=cfg.max_films is None,
        )
        log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
