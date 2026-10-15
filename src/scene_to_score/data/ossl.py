"""OSSL-v2 index: clip ids, splits, movie subsets and fixed-length segments.

Clip ids look like ``<movie_id>_<clip_idx>_<start>_<end>``. Start and end are in hundredths of
a second: clip ``746_0_4192_5407`` spans 12.15 s and its wav file lasts 12.2 s.

Media files sit at ``<split>/{video,audio}/<movie_id>/<clip_id>.{mp4,wav}``.
"""

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "test")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    movie_id: int
    index: int
    start_cs: int
    end_cs: int
    split: str

    @property
    def duration_s(self) -> float:
        return (self.end_cs - self.start_cs) / 100

    def video_path(self, root: Path) -> Path:
        return root / self.split / "video" / str(self.movie_id) / f"{self.clip_id}.mp4"

    def audio_path(self, root: Path) -> Path:
        return root / self.split / "audio" / str(self.movie_id) / f"{self.clip_id}.wav"


@dataclass(frozen=True)
class Segment:
    """A fixed-length window inside a clip; times are relative to the clip start."""

    clip: Clip
    start_s: float
    end_s: float

    @property
    def segment_id(self) -> str:
        return f"{self.clip.clip_id}@{round(self.start_s * 100)}"


def parse_clip_id(clip_id: str, split: str) -> Clip:
    parts = clip_id.split("_")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        raise ValueError(f"not an OSSL-v2 clip id: {clip_id!r}")
    movie_id, index, start_cs, end_cs = (int(part) for part in parts)
    if end_cs <= start_cs:
        raise ValueError(f"clip {clip_id!r} ends before it starts")
    return Clip(clip_id, movie_id, index, start_cs, end_cs, split)


def read_split(root: Path, split: str) -> list[Clip]:
    """Read ``splits/<split>.txt`` (one clip id per line)."""
    clip_ids = (root / "splits" / f"{split}.txt").read_text().split()
    return [parse_clip_id(clip_id, split) for clip_id in clip_ids]


def select_movies(clips: Iterable[Clip], n_movies: int | None, seed: int) -> list[Clip]:
    """Keep every clip of ``n_movies`` movies drawn with ``seed``; ``None`` keeps all movies."""
    clips = list(clips)
    movies = sorted({clip.movie_id for clip in clips})
    if n_movies is not None and n_movies < len(movies):
        movies = random.Random(seed).sample(movies, n_movies)
    keep = set(movies)
    return [clip for clip in clips if clip.movie_id in keep]


def subset(
    root: Path, n_movies: Mapping[str, int | None], seed: int, splits: Iterable[str] = SPLITS
) -> dict[str, list[Clip]]:
    """The clips used by the project, per split.

    Every step calls this with the same settings, so that all of them see the same clips.
    """
    return {
        split: select_movies(read_split(root, split), n_movies[split], seed) for split in splits
    }


def segment_clip(clip: Clip, segment_s: float, hop_s: float) -> list[Segment]:
    """Cut ``clip`` into windows of ``segment_s`` seconds every ``hop_s`` seconds.

    Windows that would run past the end of the clip are dropped.
    """
    if clip.duration_s + 1e-6 < segment_s:
        return []
    count = int((clip.duration_s - segment_s + 1e-6) // hop_s) + 1
    return [Segment(clip, i * hop_s, i * hop_s + segment_s) for i in range(count)]


def media_patterns(clips: Iterable[Clip]) -> list[str]:
    """``allow_patterns`` that download the video and audio folders of the given clips' movies."""
    folders = {(clip.split, clip.movie_id) for clip in clips}
    return sorted(
        f"{split}/{kind}/{movie_id}/*" for split, movie_id in folders for kind in ("video", "audio")
    )
