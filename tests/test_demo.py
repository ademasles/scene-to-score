from pathlib import Path

import av
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scene_to_score.audio import write_wav
from scene_to_score.demo import (
    encode_windows,
    join,
    load_predictor,
    mux,
    nearest_training_music,
    predict,
    video_duration,
    windows,
)
from scene_to_score.features.cache import write_targets
from scene_to_score.train import build

TRAIN_CFG = {
    "method": "video_only-test",
    "video_encoder": "dinov2-large",
    "fusion": "video_mean",
    "d_h": 8,
    "hidden": 16,
    "dropout": 0.0,
    "n_hypotheses": 1,
}


def _write_video(path: Path, seconds: float, fps: int = 10) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(round(seconds * fps)):
            image = np.full((24, 32, 3), (i * 7) % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode(None))


def test_windows_cover_the_video_and_end_with_it() -> None:
    assert windows(25.0, 10.0) == [(0.0, 10.0), (10.0, 20.0), (15.0, 25.0)]
    assert windows(20.0, 10.0) == [(0.0, 10.0), (10.0, 20.0)]
    assert windows(4.0, 10.0) == [(0.0, 4.0)]
    with pytest.raises(ValueError):
        windows(0.0, 10.0)


def test_join_crossfades_overlaps_and_keeps_levels() -> None:
    rate = 10
    spans = [(0.0, 10.0), (5.0, 15.0)]
    out = join([np.ones(100, np.float32), 3 * np.ones(100, np.float32)], spans, rate, 1.0)
    assert out.shape == (150,)
    np.testing.assert_allclose(out[20:45], 1.0, atol=1e-6)  # first piece alone
    np.testing.assert_allclose(out[105:140], 3.0, atol=1e-6)  # second piece alone
    assert 1.0 < out[95] < 3.0  # crossfade inside the overlap


def test_checkpoint_round_trip_and_prediction(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build(OmegaConf.create(TRAIN_CFG), 4)
    ckpt = tmp_path / "seed0.pt"
    torch.save({"state_dict": model.state_dict(), "val_r10": 0.1, "config": TRAIN_CFG}, ckpt)
    loaded, cfg = load_predictor(ckpt, 4, torch.device("cpu"))
    assert cfg.method == "video_only-test"

    video = tmp_path / "clip.mp4"
    _write_video(video, 3.0)
    calls: list[int] = []

    def encoder(frames: list[np.ndarray]) -> torch.Tensor:
        calls.append(len(frames[0]))
        return torch.from_numpy(np.stack([f.reshape(len(f), -1)[:, :4] for f in frames])).float()

    spans = windows(video_duration(video), 1.0)
    tokens = encode_windows(encoder, video, spans, 4)  # type: ignore[arg-type]
    z = predict(loaded, tokens, torch.device("cpu"))
    assert z.shape == (len(spans), 512) and calls == [4] * len(spans)
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(len(spans)))


def test_checkpoint_without_config_is_refused(tmp_path: Path) -> None:
    ckpt = tmp_path / "old.pt"
    torch.save({"state_dict": {}, "val_r10": 0.1}, ckpt)
    with pytest.raises(ValueError, match="no config"):
        load_predictor(ckpt, 4, torch.device("cpu"))


def test_nearest_training_music_ranks_by_cosine(tmp_path: Path) -> None:
    bank = np.eye(3, 512, dtype=np.float32)
    write_targets(tmp_path, "train", ["1_0_0_1000@0", "2_0_0_1000@0", "3_0_0_1000@0"], bank, {})
    z = torch.from_numpy(bank[[2, 0]])
    nearest = nearest_training_music(z, tmp_path, 2)
    assert [row[0][0] for row in nearest] == ["3_0_0_1000@0", "1_0_0_1000@0"]
    assert nearest[0][0][1] == pytest.approx(1.0)


def test_mux_replaces_the_sound(tmp_path: Path) -> None:
    video, music, out = tmp_path / "v.mp4", tmp_path / "m.wav", tmp_path / "o.mp4"
    _write_video(video, 2.0)
    write_wav(music, 0.1 * np.sin(np.arange(32_000) / 5), 16_000)
    if not mux(video, music, out):
        pytest.skip("ffmpeg not installed")
    with av.open(str(out)) as container:
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 1
