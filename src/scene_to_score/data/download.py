"""Download the OSSL-v2 index, then the media of a seeded subset of movies.

Run with ``just download``; take every movie with
``just download data.n_train_movies=null data.n_test_movies=null`` (180 GB).
"""

import logging
from pathlib import Path

import hydra
from huggingface_hub import snapshot_download
from omegaconf import DictConfig

from scene_to_score.data.ossl import Clip, media_patterns, subset

log = logging.getLogger(__name__)

INDEX_PATTERNS = ["README.md", "metadata.csv", "splits/*.txt"]


def download(cfg: DictConfig) -> dict[str, list[Clip]]:
    root = Path(cfg.paths.ossl_dir)
    common = {
        "repo_id": cfg.data.repo_id,
        "repo_type": "dataset",
        "revision": cfg.data.revision,
        "local_dir": root,
    }
    snapshot_download(**common, allow_patterns=INDEX_PATTERNS)
    clips = subset(
        root,
        {"train": cfg.data.n_train_movies, "test": cfg.data.n_test_movies},
        cfg.data.subset_seed,
    )
    for split, split_clips in clips.items():
        movies = {clip.movie_id for clip in split_clips}
        log.info("%s: %d clips from %d movies", split, len(split_clips), len(movies))
    patterns = media_patterns(clip for split_clips in clips.values() for clip in split_clips)
    snapshot_download(**common, allow_patterns=patterns, max_workers=cfg.data.download_workers)
    log.info("media downloaded to %s", root)
    return clips


@hydra.main(version_base=None, config_path="../../../configs", config_name="download")
def main(cfg: DictConfig) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HTTP request otherwise
    download(cfg)


if __name__ == "__main__":
    main()
