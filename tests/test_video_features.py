from pathlib import Path

import av
import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model, VJEPA2Config, VJEPA2Model

from scene_to_score.data.ossl import parse_clip_id, segment_clip
from scene_to_score.features.cache import read_features, write_features
from scene_to_score.features.video import (
    EncoderSpec,
    VideoEncoder,
    extract,
    frame_times,
    preprocess,
    sample_frames,
)

FPS = 10


def _write_video(path: Path, seconds: float) -> None:
    """Frame i is a flat grey image of level 10 i (mod 260), so frames can be identified."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.options = {"qscale": "1"}
        for i in range(round(seconds * FPS)):
            image = np.full((48, 64, 3), (i * 10) % 260, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def _frame_index(frame: np.ndarray) -> int:
    level = float(frame.mean())
    return int(np.argmin([abs(level - 10 * i) for i in range(26)]))


def test_frame_times_are_bin_centres() -> None:
    segment = segment_clip(parse_clip_id("1_0_0_2000", "train"), 10.0, 10.0)[1]
    assert frame_times(segment, 4) == [11.25, 13.75, 16.25, 18.75]


def test_sample_frames_picks_the_nearest_frame(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_video(path, 2.5)
    frames = sample_frames(path, [0.0, 0.31, 0.74, 1.5, 2.49])
    assert frames.shape == (5, 48, 64, 3) and frames.dtype == np.uint8
    assert [_frame_index(f) for f in frames] == [0, 3, 7, 15, 24]


def test_sample_frames_gray_repeats_luma_and_rejects_unsorted(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_video(path, 1.0)
    gray = sample_frames(path, [0.5], gray=True)
    assert (gray[..., 0] == gray[..., 1]).all() and (gray[..., 1] == gray[..., 2]).all()
    with pytest.raises(ValueError, match="sorted"):
        sample_frames(path, [0.5, 0.1])


def test_preprocess_crops_and_normalises() -> None:
    frames = torch.full((2, 3, 48, 64, 3), 255, dtype=torch.uint8)
    x = preprocess(frames, resize=32, crop=28)
    assert x.shape == (2, 3, 3, 28, 28)
    expected = (1 - 0.485) / 0.229
    assert x[0, 0, 0].mean().item() == pytest.approx(expected, rel=1e-3)


def _dinov2() -> VideoEncoder:
    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=2,
        image_size=28,
        patch_size=14,
    )
    spec = EncoderSpec("tiny", "dinov2", resize=32, crop=28)
    return VideoEncoder(Dinov2Model(config), spec, torch.device("cpu"))


def _vjepa2() -> VideoEncoder:
    torch.manual_seed(0)
    config = VJEPA2Config(
        patch_size=8,
        crop_size=32,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=2.0,
        pred_hidden_size=16,
        pred_num_hidden_layers=1,
        pred_num_attention_heads=2,
        pred_mlp_ratio=2.0,
    )
    spec = EncoderSpec("tiny", "vjepa2", resize=36, crop=32)
    return VideoEncoder(VJEPA2Model(config), spec, torch.device("cpu"))


@pytest.mark.parametrize(("make", "n_tokens"), [(_dinov2, 4), (_vjepa2, 2)])
def test_encoders_return_one_token_per_frame_or_tubelet(make, n_tokens: int) -> None:
    encoder = make()
    frames = np.random.default_rng(0).integers(0, 255, (3, 4, 48, 64, 3), dtype=np.uint8)
    tokens = encoder(frames)
    assert tokens.shape == (3, n_tokens, 32)
    assert tokens.dtype == torch.float32 and torch.isfinite(tokens).all()


def test_extract_writes_the_cache_format(tmp_path: Path) -> None:
    clip = parse_clip_id("5_0_0_2500", "train")
    video = clip.video_path(tmp_path)
    video.parent.mkdir(parents=True)
    _write_video(video, 25.0)
    segments = segment_clip(clip, 10.0, 5.0)  # 4 overlapping segments, one decoding pass
    features, seconds = extract(_dinov2(), tmp_path, segments, n_frames=4, batch_size=2)
    assert features.ids == [s.segment_id for s in segments]
    assert features.tokens is not None and features.tokens.shape == (4, 4, 32)
    np.testing.assert_allclose(features.pooled, features.tokens.mean(1), rtol=1e-5)
    assert seconds > 0
    write_features(tmp_path / "cache", "train", features, {"encoder": "tiny"})
    assert read_features(tmp_path / "cache", "train").ids == features.ids


def test_extract_video_config_composes() -> None:
    from hydra import compose, initialize_config_dir

    from scene_to_score.features.video import ENCODERS

    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="extract_video", overrides=["encoder=vjepa2-large"])
    assert cfg.encoder in ENCODERS
    assert cfg.n_frames % 2 == 0  # V-JEPA 2 tubelets of 2 frames
