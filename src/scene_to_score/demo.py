"""End-to-end demo: an unseen video in, the same video with generated music out.

The video is cut into ``segment_s`` windows (the last one is moved back so that it ends with the
video); each window is encoded by the frozen video encoder, the trained predictor maps it to a
CLAP vector, and MusicLDM turns each vector into ``segment_s`` seconds of music. The pieces are
joined with a short crossfade and muxed over the picture with ffmpeg (the original sound track is
dropped). With ``retrieve_k > 0`` the real training music closest to each prediction is also cut
out, as a reference to listen to.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import av
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from scene_to_score.audio import load_audio, write_wav
from scene_to_score.data.ossl import parse_clip_id
from scene_to_score.features.cache import read_targets
from scene_to_score.features.video import VideoEncoder, load_encoder, sample_frames
from scene_to_score.generation import empty_text_embedding, generate, load_pipeline
from scene_to_score.seed import pick_device, seed_everything
from scene_to_score.train import Predictor, build

log = logging.getLogger(__name__)


def video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return container.duration / av.time_base
    raise ValueError(f"{path}: unknown duration")


def windows(duration_s: float, segment_s: float) -> list[tuple[float, float]]:
    """Windows of ``segment_s`` covering the video; the last ends with it (it may overlap).

    A video shorter than one window gives a single, shorter window.
    """
    if duration_s <= 0:
        raise ValueError("empty video")
    if duration_s <= segment_s:
        return [(0.0, duration_s)]
    n = int(np.ceil(duration_s / segment_s - 1e-6))
    starts = [i * segment_s for i in range(n - 1)] + [duration_s - segment_s]
    return [(start, start + segment_s) for start in starts]


def window_frames(path: Path, window: tuple[float, float], n_frames: int) -> np.ndarray:
    start, end = window
    step = (end - start) / n_frames
    return sample_frames(path, [start + (i + 0.5) * step for i in range(n_frames)])


@torch.no_grad()
def encode_windows(
    encoder: VideoEncoder, path: Path, spans: list[tuple[float, float]], n_frames: int
) -> torch.Tensor:
    """Video tokens of each window (W x L x d), as in the feature cache."""
    return torch.cat([encoder([window_frames(path, w, n_frames)]) for w in spans]).float()


@torch.no_grad()
def predict(model: Predictor, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """One unit-norm CLAP vector per window (W x 512); hypothesis 0 if the head has several."""
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
    model.eval()
    z = model(tokens.to(device), mask)
    if z.ndim == 3:
        z = z[:, 0]
    return torch.nn.functional.normalize(z.float(), dim=-1)


def join(
    pieces: list[np.ndarray], spans: list[tuple[float, float]], rate: int, fade_s: float
) -> np.ndarray:
    """Place each piece at its window start; overlaps are crossfaded linearly."""
    total = round(spans[-1][1] * rate)
    out = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    fade = max(1, round(fade_s * rate))
    for piece, (start, end) in zip(pieces, spans, strict=True):
        a = round(start * rate)
        n = min(len(piece), round((end - start) * rate), total - a)
        ramp = np.ones(n, dtype=np.float32)
        k = min(fade, n // 2)
        ramp[:k] = np.linspace(0, 1, k, endpoint=False)
        ramp[n - k :] = np.linspace(1, 0, k, endpoint=False)
        out[a : a + n] += piece[:n] * ramp
        weight[a : a + n] += ramp
    return out / np.maximum(weight, 1e-3)


def nearest_training_music(
    z: torch.Tensor, target_dir: Path, k: int
) -> list[list[tuple[str, float]]]:
    """The ``k`` training segments whose CLAP vector is closest to each prediction."""
    ids, bank = read_targets(target_dir, "train")
    sims = z.cpu().float() @ torch.from_numpy(bank).float().T
    top = sims.topk(k, dim=1)
    return [
        [(ids[int(j)], float(s)) for s, j in zip(row_s, row_j, strict=True)]
        for row_s, row_j in zip(top.values, top.indices, strict=True)
    ]


def cut_segment(ossl_dir: Path, segment_id: str, segment_s: float) -> tuple[np.ndarray, int]:
    clip_id, start_cs = segment_id.split("@")
    clip = parse_clip_id(clip_id, "train")
    return load_audio(clip.audio_path(ossl_dir), int(start_cs) / 100, segment_s)


def mux(video: Path, music: Path, out: Path) -> bool:
    """Replace the sound of ``video`` by ``music``; False if ffmpeg is missing."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("ffmpeg not found: only the wav file is written")
        return False
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-i", str(music),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    return True


def load_predictor(checkpoint: Path, d_video: int, device: torch.device) -> tuple[Predictor, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in state:
        raise ValueError(f"{checkpoint} has no config: retrain with the current train.py")
    cfg = OmegaConf.create(state["config"])
    model = build(cast(DictConfig, cfg), d_video)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), cfg


@hydra.main(version_base=None, config_path="../../configs", config_name="demo")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    video = Path(cfg.input)
    device = pick_device(cfg.device)
    duration = video_duration(video)
    spans = windows(duration, cfg.data.segment_s)
    log.info("%s: %.1f s, %d windows", video, duration, len(spans))

    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    encoder_name = state["config"]["video_encoder"]
    encoder = load_encoder(encoder_name, device, half=cfg.half)
    tokens = encode_windows(encoder, video, spans, cfg.n_frames)
    model, train_cfg = load_predictor(Path(cfg.checkpoint), tokens.shape[-1], device)
    z = predict(model, tokens, device)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.save(out_dir / "z.npy", z.cpu().numpy())

    report: dict[str, Any] = {
        "input": str(video),
        "duration_s": duration,
        "windows": spans,
        "checkpoint": str(cfg.checkpoint),
        "method": train_cfg.method,
        "video_encoder": encoder_name,
    }
    if cfg.retrieve_k > 0:
        target_dir = Path(cfg.paths.features_dir) / "music" / "clap"
        nearest = nearest_training_music(z, target_dir, cfg.retrieve_k)
        report["nearest_training_music"] = nearest
        for w, row in enumerate(nearest):
            samples, rate = cut_segment(Path(cfg.paths.ossl_dir), row[0][0], cfg.data.segment_s)
            write_wav(out_dir / f"nearest_{w:02d}.wav", samples, rate)

    pipe = load_pipeline(cfg.model_id, device, half=cfg.half)
    negative = empty_text_embedding(pipe.text_encoder, pipe.tokenizer)
    rate = pipe.vocoder.config.sampling_rate
    pieces = [
        generate(pipe, z[w : w + 1], negative, seconds=cfg.data.segment_s, steps=cfg.steps,
                 guidance=cfg.guidance, seed=cfg.seed + w)[0]
        for w in range(len(spans))
    ]  # fmt: skip
    music = join(pieces, spans, rate, cfg.crossfade_s)
    write_wav(out_dir / "music.wav", music, rate)
    if mux(video, out_dir / "music.wav", out_dir / f"{video.stem}_scored.mp4"):
        report["output"] = str(out_dir / f"{video.stem}_scored.mp4")
    (out_dir / "demo.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("done: %s", out_dir)


if __name__ == "__main__":
    main()
