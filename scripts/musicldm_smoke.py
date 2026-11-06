"""A2 smoke test: MusicLDM conditioned on CLAP audio vectors through ``prompt_embeds``.

For a few test clips, embed the real music (middle segment) with MusicLDM's own CLAP, generate
10 s from that vector, and also from the mean of ``n_mean`` vectors (off the data manifold, like
a model's prediction). Writes the real and generated audio for listening, plus ``summary.json``:
seconds per generation and a round trip, CLAP(generated) against the conditioning vectors.
"""

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from scene_to_score.audio import load_audio, resample, write_wav
from scene_to_score.data.ossl import evaluation_segment, segment_clip, subset
from scene_to_score.generation import (
    audio_embedding,
    cosine_matrix,
    empty_text_embedding,
    generate,
    load_pipeline,
    round_trip_report,
)
from scene_to_score.seed import pick_device, seed_everything

log = logging.getLogger(__name__)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@hydra.main(version_base=None, config_path="../configs", config_name="musicldm_smoke")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    root = Path(cfg.paths.ossl_dir)
    clips = subset(
        root,
        {"train": cfg.data.n_train_movies, "test": cfg.data.n_test_movies},
        cfg.data.subset_seed,
    )[cfg.split]
    segments = [
        evaluation_segment(s)
        for clip in sorted(clips, key=lambda c: c.clip_id)
        if (s := segment_clip(clip, cfg.data.segment_s, cfg.data.hop_s))
    ][: cfg.n_mean]
    if len(segments) < max(cfg.n_generate, 2):
        raise ValueError(f"only {len(segments)} usable clips in {root}")

    device = pick_device()
    pipe = load_pipeline(cfg.model_id, device, half=cfg.half)
    clap, extractor = pipe.text_encoder, pipe.feature_extractor
    clap_rate, vocoder_rate = extractor.sampling_rate, pipe.vocoder.config.sampling_rate

    waves = []
    for segment in segments:
        samples, rate = load_audio(
            segment.clip.audio_path(root), segment.start_s, cfg.data.segment_s
        )
        waves.append(resample(samples, rate, clap_rate))
    z_real = audio_embedding(clap, extractor, waves)
    mean = torch.nn.functional.normalize(z_real.mean(0, keepdim=True), dim=-1)
    names = [s.segment_id for s in segments[: cfg.n_generate]] + [f"mean{len(segments)}"]
    z_cond = torch.cat([z_real[: cfg.n_generate], mean])
    negative = empty_text_embedding(clap, pipe.tokenizer)

    generated, seconds = [], []
    for i, name in enumerate(names):
        sync(device)
        start = time.perf_counter()
        audio = generate(
            pipe,
            z_cond[i : i + 1],
            negative,
            seconds=cfg.data.segment_s,
            steps=cfg.steps,
            guidance=cfg.guidance,
            seed=cfg.seed,
        )[0]
        sync(device)
        seconds.append(time.perf_counter() - start)
        generated.append(resample(audio, vocoder_rate, clap_rate))
        write_wav(out_dir / f"{i:02d}_{name}_generated.wav", audio, vocoder_rate)
        if i < cfg.n_generate:
            write_wav(out_dir / f"{i:02d}_{name}_real.wav", waves[i], clap_rate)
        log.info("%s: %.1f s", name, seconds[-1])

    similarity = cosine_matrix(audio_embedding(clap, extractor, generated), z_cond)
    real = cosine_matrix(z_real, z_real)
    summary = {
        "conditions": names,
        "device": str(device),
        "dtype": str(clap.dtype),
        "steps": cfg.steps,
        "guidance": cfg.guidance,
        "seconds_per_generation": seconds,
        "real_pairwise_cosine": float(real[~np.eye(len(real), dtype=bool)].mean()),
        "round_trip": round_trip_report(similarity),
        "round_trip_matrix": similarity.round(3).tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("round trip: %s", summary["round_trip"])
    log.info("listen in %s", out_dir)


if __name__ == "__main__":
    main()
