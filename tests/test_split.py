from pathlib import Path

import pytest

from scene_to_score.data.ossl import parse_clip_id
from scene_to_score.data.split import audit_report, create_splits, write_validation_movies


def _write_official_splits(root: Path, train: list[str], test: list[str]) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("\n".join(train))
    (split_dir / "test.txt").write_text("\n".join(test))


def test_create_splits_is_reproducible_and_movie_disjoint(tmp_path: Path) -> None:
    _write_official_splits(
        tmp_path,
        [f"{movie_id}_{clip}_0_2000" for movie_id in range(10) for clip in range(2)],
        ["20_0_0_2000"],
    )

    splits, validation_movies = create_splits(tmp_path, seed=0)
    repeated_splits, repeated_validation_movies = create_splits(tmp_path, seed=0)

    assert validation_movies == repeated_validation_movies
    assert validation_movies == sorted({clip.movie_id for clip in splits["val"]})
    assert splits == repeated_splits
    assert {clip.movie_id for clip in splits["train"]}.isdisjoint(validation_movies)
    assert {clip.movie_id for clip in splits["test"]}.isdisjoint(validation_movies)
    assert len(validation_movies) == 1


def test_create_splits_rejects_official_train_test_overlap(tmp_path: Path) -> None:
    _write_official_splits(tmp_path, ["1_0_0_2000"], ["1_1_0_2000"])

    with pytest.raises(ValueError, match="share movies"):
        create_splits(tmp_path)


def test_write_validation_movies_is_sorted(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "val_movies.txt"

    write_validation_movies([12, 4], path)

    assert path.read_text() == "4\n12\n"


def test_audit_reports_movie_clip_and_segment_counts() -> None:
    clips = {
        "train": [parse_clip_id("1_0_0_2000", "train")],
        "val": [parse_clip_id("2_0_0_3000", "train")],
        "test": [parse_clip_id("3_0_0_1000", "test")],
    }
    metadata = {
        movie_id: {
            "split": "test" if movie_id == 3 else "train",
            "year": {1: "1977", 2: "1980", 3: "2000"}[movie_id],
            "genres": "Drama" if movie_id != 1 else "",
            "imdb_description": "" if movie_id == 2 else "Description",
        }
        for movie_id in (1, 2, 3)
    }

    report = audit_report(clips, metadata, segment_s=10.0, hop_s=10.0)

    assert "| train | 1 | 1 | 2 |" in report
    assert "| val | 1 | 1 | 3 |" in report
    assert "| test | 1 | 1 | 1 |" in report
    assert "1977 is in range" in report
    assert "1/3 (33.3%)" in report


def test_create_splits_rejects_invalid_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        create_splits(tmp_path, validation_fraction=1.0)
