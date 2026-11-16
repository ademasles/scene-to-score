"""Frozen video encoders: DINOv2 frame by frame, and V-JEPA 2 on the whole segment.

Both see the same ``n_frames`` frames per 10 s segment, taken at the centres of equal time
bins, so that the comparison is about the encoder and not about what it sees. Stored tokens:

- DINOv2: one token per frame (the layer-normed CLS, ``pooler_output``), T x d;
- V-JEPA 2: patch tokens averaged over space, one per tubelet, T / tubelet_size x d.

``pooled`` is the mean of the tokens. The audio track of the video is never read: the music
would leak into the input.

Preprocessing is done here in torch (resize the short side, centre crop, ImageNet
normalisation) rather than by the Hugging Face processors, because the V-JEPA 2 one needs
torchvision. Sizes follow those processors: 256 then 224 for DINOv2, 292 then 256 for V-JEPA 2.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Literal

import av
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from transformers import VJEPA2Model

from scene_to_score.data.ossl import Clip, Segment, segment_clip, subset
from scene_to_score.features.cache import Features, write_features
from scene_to_score.seed import pick_device

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Kind = Literal["dinov2", "vjepa2"]


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    kind: Kind
    resize: int
    crop: int


ENCODERS = {
    "dinov2-large": EncoderSpec("facebook/dinov2-large", "dinov2", resize=256, crop=224),
    "vjepa2-large": EncoderSpec("facebook/vjepa2-vitl-fpc64-256", "vjepa2", resize=292, crop=256),
}


def frame_times(segment: Segment, n_frames: int) -> list[float]:
    """Clip-relative times (s) of the ``n_frames`` frames of a segment, at bin centres."""
    step = (segment.end_s - segment.start_s) / n_frames
    return [segment.start_s + (i + 0.5) * step for i in range(n_frames)]


def sample_frames(path: Path, times: Sequence[float], *, gray: bool = False) -> np.ndarray:
    """RGB frames (len(times) x H x W x 3, uint8) nearest to each time, in one decoding pass.

    ``times`` must be sorted. With ``gray`` the frames are converted to luma and repeated over
    the three channels (the colour ablation).
    """
    if list(times) != sorted(times):
        raise ValueError("times must be sorted")
    picked: list[np.ndarray] = []
    previous: tuple[float, av.VideoFrame] | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            while len(picked) < len(times) and frame.time >= times[len(picked)]:
                target = times[len(picked)]
                nearest = frame
                if previous is not None and target - previous[0] < frame.time - target:
                    nearest = previous[1]
                picked.append(_to_rgb(nearest, gray))
            if len(picked) == len(times):
                break
            previous = (frame.time, frame)
        if len(picked) < len(times) and previous is not None:
            # times past the last frame (rounding at the clip end): repeat the last frame
            if times[-1] - previous[0] > 0.5:
                raise ValueError(f"{path}: no frame near {times[len(picked)]:.2f} s")
            picked += [_to_rgb(previous[1], gray)] * (len(times) - len(picked))
    if len(picked) != len(times):
        raise ValueError(f"{path}: decoded {len(picked)} of {len(times)} frames")
    return np.stack(picked)


def _to_rgb(frame: av.VideoFrame, gray: bool) -> np.ndarray:
    if gray:
        luma = frame.to_ndarray(format="gray")
        return np.repeat(luma[..., None], 3, axis=-1)
    return frame.to_ndarray(format="rgb24")


def preprocess(frames: torch.Tensor, resize: int, crop: int) -> torch.Tensor:
    """uint8 (..., H, W, 3) to normalised float (..., 3, crop, crop)."""
    lead = frames.shape[:-3]
    x = frames.reshape(-1, *frames.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    scale = resize / min(h, w)
    size = (max(crop, round(h * scale)), max(crop, round(w * scale)))
    x = F.interpolate(x, size=size, mode="bicubic", antialias=True, align_corners=False)
    top, left = (size[0] - crop) // 2, (size[1] - crop) // 2
    x = x[..., top : top + crop, left : left + crop].clamp(0.0, 1.0)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(*lead, 3, crop, crop)


class VideoEncoder:
    """Turns a batch of frame stacks (B x T x H x W x 3, uint8) into tokens (B x T' x d)."""

    def __init__(self, model: nn.Module, spec: EncoderSpec, device: torch.device) -> None:
        self.model = model.eval().to(device)
        self.spec = spec
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.tubelet_size = int(getattr(getattr(model, "config", None), "tubelet_size", 1))

    @torch.inference_mode()
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frames).to(self.device)
        x = preprocess(x, self.spec.resize, self.spec.crop).to(self.dtype)
        b, t = x.shape[:2]
        if self.spec.kind == "dinov2":
            out = self.model(pixel_values=x.flatten(0, 1))
            return out.pooler_output.reshape(b, t, -1).float()
        if not isinstance(self.model, VJEPA2Model):
            raise TypeError(f"expected VJEPA2Model, got {type(self.model).__name__}")
        tokens = self.model.get_vision_features(x)  # B x (T' S) x d, time outermost
        tubelets = t // self.tubelet_size
        return tokens.reshape(b, tubelets, -1, tokens.shape[-1]).mean(2).float()


def load_encoder(name: str, device: torch.device, half: bool = True) -> VideoEncoder:
    from transformers import AutoModel

    spec = ENCODERS[name]
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    return VideoEncoder(AutoModel.from_pretrained(spec.model_id, dtype=dtype), spec, device)


def extract(
    encoder: VideoEncoder,
    root: Path,
    segments: Iterable[Segment],
    *,
    n_frames: int,
    batch_size: int,
    gray: bool = False,
) -> tuple[Features, float]:
    """Encode every segment; returns the features and the seconds spent per segment.

    Segments are grouped by clip so that each video is decoded once.
    """
    ids: list[str] = []
    tokens: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    start = time.perf_counter()

    def flush() -> None:
        if batch:
            tokens.append(encoder(np.stack(batch)).cpu().numpy())
            batch.clear()

    for clip_id, group in groupby(segments, key=lambda s: s.clip.clip_id):
        clip_segments = list(group)
        times = np.array([t for s in clip_segments for t in frame_times(s, n_frames)])
        # overlapping segments (hop < length) share times: decode each time once
        unique, index = np.unique(times, return_inverse=True)
        frames = sample_frames(clip_segments[0].clip.video_path(root), unique.tolist(), gray=gray)
        frames = frames[index]
        for i, segment in enumerate(clip_segments):
            batch.append(frames[i * n_frames : (i + 1) * n_frames])
            ids.append(segment.segment_id)
            if len(batch) == batch_size:
                flush()
        log.debug("%s: %d segments", clip_id, len(clip_segments))
    flush()
    if not ids:
        raise ValueError("no segments to encode")
    seconds = (time.perf_counter() - start) / len(ids)
    stacked = np.concatenate(tokens)
    features = Features(
        ids=ids,
        pooled=stacked.mean(1),
        tokens=stacked,
        mask=np.ones(stacked.shape[:2], dtype=bool),
    )
    return features, seconds


def segments_of(clips: Iterable[Clip], segment_s: float, hop_s: float) -> list[Segment]:
    return [s for clip in clips for s in segment_clip(clip, segment_s, hop_s)]


@hydra.main(version_base=None, config_path="../../../configs", config_name="extract_video")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.ossl_dir)
    clips = subset(
        root,
        {"train": cfg.data.n_train_movies, "test": cfg.data.n_test_movies},
        cfg.data.subset_seed,
        splits=cfg.splits,
    )
    device = pick_device()
    encoder = load_encoder(cfg.encoder, device, half=cfg.half)
    name = f"{cfg.encoder}-gray" if cfg.gray else cfg.encoder
    out_dir = Path(cfg.paths.features_dir) / "video" / name
    spec = encoder.spec
    meta = {
        "encoder": cfg.encoder,
        "model_id": spec.model_id,
        "n_frames": cfg.n_frames,
        "frame_sampling": "bin centres over each segment",
        "segment_s": cfg.data.segment_s,
        "hop_s": cfg.data.hop_s,
        "resize": spec.resize,
        "crop": spec.crop,
        "gray": cfg.gray,
        "dtype": str(encoder.dtype),
    }
    for split, split_clips in clips.items():
        segments = segments_of(split_clips, cfg.data.segment_s, cfg.data.hop_s)
        if cfg.max_segments is not None:
            segments = segments[: cfg.max_segments]
        features, seconds = extract(
            encoder,
            root,
            segments,
            n_frames=cfg.n_frames,
            batch_size=cfg.batch_size,
            gray=cfg.gray,
        )
        info = {
            "seconds_per_segment": seconds,
            "device": str(device),
            "partial": cfg.max_segments is not None,
        }
        write_features(out_dir, split, features, meta, split_info=info)
        log.info(
            "%s %s: %d segments, tokens %s, %.3f s/segment",
            name,
            split,
            len(features.ids),
            None if features.tokens is None else features.tokens.shape[1:],
            seconds,
        )


if __name__ == "__main__":
    main()
