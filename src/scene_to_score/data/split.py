"""Create and audit a reproducible movie-level validation split for OSSL-v2."""

import csv
import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
from huggingface_hub import snapshot_download
from omegaconf import DictConfig

from scene_to_score.data.ossl import Clip, read_split, segment_clip

log = logging.getLogger(__name__)

VALIDATION_FRACTION = 0.1
# Not 0: the movie subset (data.subset_seed = 0) draws from the same sorted list with
# random.Random(0), and the two draws then share most of their movies (157 of the 166
# validation movies fell inside the 400-movie subset). Seed 1 gives the expected ~10 %.
VALIDATION_SEED = 1
INDEX_PATTERNS = ["metadata.csv", "splits/train.txt", "splits/test.txt"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_MOVIES_PATH = PROJECT_ROOT / "splits" / "val_movies.txt"


def create_splits(
    root: Path, seed: int = VALIDATION_SEED, validation_fraction: float = VALIDATION_FRACTION
) -> tuple[dict[str, list[Clip]], list[int]]:
    """Split official training movies into train/validation and retain the official test split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    official = {split: read_split(root, split) for split in ("train", "test")}
    movie_ids = {split: {clip.movie_id for clip in clips} for split, clips in official.items()}
    overlap = movie_ids["train"] & movie_ids["test"]
    if overlap:
        raise ValueError(f"official train and test splits share movies: {sorted(overlap)}")

    train_movies = sorted(movie_ids["train"])
    n_validation_movies = round(len(train_movies) * validation_fraction)
    if n_validation_movies == 0:
        raise ValueError("training split is too small to select validation movies")
    validation_movies = sorted(random.Random(seed).sample(train_movies, n_validation_movies))
    validation_set = set(validation_movies)

    splits = {
        "train": [clip for clip in official["train"] if clip.movie_id not in validation_set],
        "val": [clip for clip in official["train"] if clip.movie_id in validation_set],
        "test": official["test"],
    }
    return splits, validation_movies


def write_validation_movies(movie_ids: Sequence[int], path: Path = VALIDATION_MOVIES_PATH) -> None:
    """Write one sorted movie ID per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{movie_id}\n" for movie_id in sorted(movie_ids)))


def _read_metadata(path: Path) -> dict[int, dict[str, str]]:
    required = {"movie_id", "split", "year", "genres", "imdb_description"}
    optional = {"youtube_description"}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            movie_id = int(row["movie_id"])
            if movie_id in rows:
                raise ValueError(f"duplicate movie_id in {path}: {movie_id}")
            rows[movie_id] = {key: row.get(key) or "" for key in required | optional}
    return rows


def _validate_metadata_splits(
    clips: Mapping[str, Sequence[Clip]], metadata: Mapping[int, Mapping[str, str]]
) -> None:
    for split in ("train", "test"):
        clip_movies = {clip.movie_id for clip in clips[split]}
        if split == "train" and "val" in clips:
            clip_movies.update(clip.movie_id for clip in clips["val"])
        metadata_movies = {movie_id for movie_id, row in metadata.items() if row["split"] == split}
        if clip_movies != metadata_movies:
            raise ValueError(
                f"{split} split movies disagree between split files and metadata.csv "
                f"(only in split files: {sorted(clip_movies - metadata_movies)[:10]}, "
                f"only in metadata: {sorted(metadata_movies - clip_movies)[:10]})"
            )


def audit_report(
    splits: Mapping[str, Sequence[Clip]],
    metadata: Mapping[int, Mapping[str, str]],
    segment_s: float,
    hop_s: float,
) -> str:
    """Return a concise Markdown audit note suitable for the C1 pull request."""
    _validate_metadata_splits(splits, metadata)

    missing_description = sum(not row["imdb_description"].strip() for row in metadata.values())
    no_synopsis = sum(
        not (row["imdb_description"].strip() or row.get("youtube_description", "").strip())
        for row in metadata.values()
    )
    missing_genres = sum(not row["genres"].strip() for row in metadata.values())
    movie_count = len(metadata)
    if not movie_count:
        raise ValueError("metadata.csv contains no movies")

    years: Counter[int] = Counter()
    missing_years = 0
    for row in metadata.values():
        value = row["year"].strip()
        try:
            years[int(value)] += 1
        except ValueError:
            missing_years += 1

    year_range = f"{min(years)}-{max(years)}" if years else "unavailable"
    year_1977 = years[1977]
    histogram: Counter[int] = Counter()
    for year, count in years.items():
        histogram[(year // 10) * 10] += count
    histogram_text = ", ".join(f"{decade}s: {histogram[decade]}" for decade in sorted(histogram))
    year_note = (
        f"1977 is in range ({year_1977} films)."
        if years and min(years) <= 1977 <= max(years)
        else f"1977 is outside the observed range ({year_range}; {year_1977} films)."
    )

    lines = [
        "### OSSL-v2 split audit",
        "",
        f"- Films without `imdb_description`: {missing_description}/{movie_count} "
        f"({missing_description / movie_count:.1%}); "
        f"without any synopsis once `youtube_description` fills the gaps: {no_synopsis}.",
        f"- Films without `genres`: {missing_genres}/{movie_count} "
        f"({missing_genres / movie_count:.1%}).",
        f"- Release-year range: {year_range}; missing/unparseable year: {missing_years}.",
        f"- Year histogram by decade: {histogram_text}.",
        f"- {year_note}",
        "",
        "| Split | Films | Clips | Segments |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        clips = splits[split]
        films = len({clip.movie_id for clip in clips})
        segments = sum(len(segment_clip(clip, segment_s, hop_s)) for clip in clips)
        lines.append(f"| {split} | {films} | {len(clips)} | {segments} |")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../../configs", config_name="download")
def main(cfg: DictConfig) -> None:
    """Fetch only the index files, write the fixed split, and print its audit."""
    root = Path(cfg.paths.ossl_dir)
    snapshot_download(
        repo_id=cfg.data.repo_id,
        repo_type="dataset",
        revision=cfg.data.revision,
        local_dir=root,
        allow_patterns=INDEX_PATTERNS,
        token=False,
    )
    splits, validation_movies = create_splits(root)
    metadata = _read_metadata(root / "metadata.csv")
    # The audit checks the splits against metadata.csv, so run it before writing anything.
    report = audit_report(splits, metadata, segment_s=cfg.data.segment_s, hop_s=cfg.data.hop_s)
    write_validation_movies(validation_movies)
    print(report)
    log.info("wrote %d validation movie IDs to %s", len(validation_movies), VALIDATION_MOVIES_PATH)


if __name__ == "__main__":
    main()
