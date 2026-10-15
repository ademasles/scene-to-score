from pathlib import Path

import pytest

from scene_to_score.data.ossl import (
    media_patterns,
    parse_clip_id,
    segment_clip,
    select_movies,
    subset,
)


def test_parse_clip_id_reads_centiseconds() -> None:
    clip = parse_clip_id("746_0_4192_5407", "test")
    assert (clip.movie_id, clip.index) == (746, 0)
    assert clip.duration_s == pytest.approx(12.15)
    assert clip.audio_path(Path("root")) == Path("root/test/audio/746/746_0_4192_5407.wav")
    assert clip.video_path(Path("root")) == Path("root/test/video/746/746_0_4192_5407.mp4")


@pytest.mark.parametrize("bad", ["746_0_4192", "a_0_1_2", "1_0_500_400"])
def test_parse_clip_id_rejects_malformed_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_clip_id(bad, "train")


@pytest.mark.parametrize(
    ("end_cs", "hop_s", "expected"),
    [(999, 10.0, 0), (1000, 10.0, 1), (1215, 10.0, 1), (2000, 10.0, 2), (2000, 5.0, 3)],
)
def test_segment_clip_keeps_full_windows_only(end_cs: int, hop_s: float, expected: int) -> None:
    clip = parse_clip_id(f"1_0_0_{end_cs}", "train")
    segments = segment_clip(clip, 10.0, hop_s)
    assert len(segments) == expected
    assert all(s.end_s <= clip.duration_s + 1e-6 for s in segments)


def test_segment_ids_are_unique_within_a_clip() -> None:
    segments = segment_clip(parse_clip_id("7_2_0_3000", "train"), 10.0, 5.0)
    assert [s.segment_id for s in segments] == [
        "7_2_0_3000@0",
        "7_2_0_3000@500",
        "7_2_0_3000@1000",
        "7_2_0_3000@1500",
        "7_2_0_3000@2000",
    ]


def test_select_movies_is_seeded_and_keeps_whole_movies() -> None:
    clips = [parse_clip_id(f"{m}_{i}_0_1500", "train") for m in range(10) for i in range(3)]
    picked = select_movies(clips, 4, seed=0)
    assert picked == select_movies(clips, 4, seed=0)
    assert len({c.movie_id for c in picked}) == 4
    assert len(picked) == 12
    assert select_movies(clips, None, seed=0) == clips


def test_subset_reads_split_files(tmp_path: Path) -> None:
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "train.txt").write_text("1_0_0_1500\n1_1_2000_3500\n2_0_0_1200\n")
    (tmp_path / "splits" / "test.txt").write_text("9_0_0_1100\n")
    clips = subset(tmp_path, {"train": 1, "test": None}, seed=0)
    assert len({c.movie_id for c in clips["train"]}) == 1
    assert [c.clip_id for c in clips["test"]] == ["9_0_0_1100"]
    assert {c.split for c in clips["test"]} == {"test"}


def test_media_patterns_cover_video_and_audio_once_per_movie() -> None:
    clips = [parse_clip_id("3_0_0_1500", "train"), parse_clip_id("3_1_0_1500", "train")]
    assert media_patterns(clips) == ["train/audio/3/*", "train/video/3/*"]
