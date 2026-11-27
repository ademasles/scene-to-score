"""C2: CLAP audio vectors of the music, the regression targets.

Each 10 s segment of a clip's wav file is embedded with MusicLDM's own CLAP (its
``text_encoder``, a full ClapModel, and its ``feature_extractor``), so the targets live in the
space the generator is conditioned on (see :mod:`scene_to_score.generation`). Segments are cut
exactly as for the video (:func:`scene_to_score.features.video.segments_of`), so ids match row
for row. Written to ``<features_dir>/music/clap/<split>_z.npy`` (float32, unit norm).
"""

import logging
import time
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import ClapFeatureExtractor, ClapModel

from scene_to_score.audio import load_audio, resample
from scene_to_score.data.ossl import Segment, subset
from scene_to_score.features.cache import write_targets
from scene_to_score.features.video import segments_of
from scene_to_score.generation import audio_embedding
from scene_to_score.seed import pick_device, seed_everything

log = logging.getLogger(__name__)


def load_audio_encoder(
    model_id: str, device: torch.device
) -> tuple[ClapModel, ClapFeatureExtractor]:
    """MusicLDM's CLAP in float32 and its feature extractor."""
    model = ClapModel.from_pretrained(model_id, subfolder="text_encoder", dtype=torch.float32)
    extractor = ClapFeatureExtractor.from_pretrained(model_id, subfolder="feature_extractor")
    cast(torch.nn.Module, model).to(device).eval()
    return model, extractor


def segment_waves(root: Path, segments: list[Segment], rate_out: int) -> list[np.ndarray]:
    """The audio of each segment of one clip, mono at ``rate_out``; the wav is decoded once."""
    path = segments[0].clip.audio_path(root)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `just download` with the same subset")
    samples, rate = load_audio(path)
    waves = []
    for segment in segments:
        start, end = round(segment.start_s * rate), round(segment.end_s * rate)
        if end > len(samples):
            raise ValueError(
                f"{path}: segment {segment.segment_id} ends after the audio "
                f"({len(samples) / rate:.2f} s)"
            )
        waves.append(resample(samples[start:end], rate, rate_out))
    return waves


def extract(
    model: ClapModel,
    extractor: ClapFeatureExtractor,
    root: Path,
    segments: Iterable[Segment],
    *,
    batch_size: int,
) -> tuple[list[str], np.ndarray, float]:
    """Embed every segment; returns ids, unit-norm vectors (N x d) and seconds per segment."""
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    start = time.perf_counter()

    def flush() -> None:
        if batch:
            vectors.append(audio_embedding(model, extractor, batch).cpu().numpy())
            batch.clear()

    for _, group in groupby(segments, key=lambda s: s.clip.clip_id):
        clip_segments = list(group)
        for segment, wave in zip(
            clip_segments,
            segment_waves(root, clip_segments, extractor.sampling_rate),
            strict=True,
        ):
            batch.append(wave)
            ids.append(segment.segment_id)
            if len(batch) == batch_size:
                flush()
    flush()
    if not ids:
        raise ValueError("no segments to embed")
    seconds = (time.perf_counter() - start) / len(ids)
    return ids, np.concatenate(vectors).astype(np.float32), seconds


@hydra.main(version_base=None, config_path="../../../configs", config_name="features/music")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    root = Path(cfg.paths.ossl_dir)
    clips = subset(
        root,
        {"train": cfg.data.n_train_movies, "test": cfg.data.n_test_movies},
        cfg.data.subset_seed,
        splits=cfg.splits,
    )
    device = pick_device(cfg.device)
    model, extractor = load_audio_encoder(cfg.model_id, device)
    # Longer inputs are cropped or fused at random by the extractor: the targets would not be
    # reproducible, so refuse them.
    if round(cfg.data.segment_s * extractor.sampling_rate) > extractor.nb_max_samples:
        raise ValueError(
            f"segments of {cfg.data.segment_s} s exceed the extractor's "
            f"{extractor.nb_max_samples / extractor.sampling_rate} s"
        )
    out_dir = Path(cfg.paths.features_dir) / "music" / "clap"
    meta = {
        "encoder": "clap",
        "model_id": cfg.model_id,
        "subfolders": ["text_encoder", "feature_extractor"],
        "sampling_rate": extractor.sampling_rate,
        "truncation": extractor.truncation,
        "max_length_s": extractor.nb_max_samples / extractor.sampling_rate,
        "segment_s": cfg.data.segment_s,
        "hop_s": cfg.data.hop_s,
        "dtype": "torch.float32",
    }
    log.info("embedding music with %s on %s", cfg.model_id, device)
    for split, split_clips in clips.items():
        segments = segments_of(split_clips, cfg.data.segment_s, cfg.data.hop_s)
        if cfg.max_segments is not None:
            segments = segments[: cfg.max_segments]
        ids, z, seconds = extract(
            model,
            extractor,
            root,
            tqdm(segments, desc=f"clap {split}", unit="seg"),
            batch_size=cfg.batch_size,
        )
        info = {
            "seconds_per_segment": seconds,
            "device": str(device),
            "partial": cfg.max_segments is not None,
        }
        write_targets(out_dir, split, ids, z, meta, split_info=info)
        log.info("clap %s: %d segments, %.3f s/segment", split, len(ids), seconds)


if __name__ == "__main__":
    main()
