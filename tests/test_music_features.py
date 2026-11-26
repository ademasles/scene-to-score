from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import ClapConfig, ClapFeatureExtractor, ClapModel

from scene_to_score.audio import write_wav
from scene_to_score.data.ossl import parse_clip_id, segment_clip
from scene_to_score.features.cache import read_targets, write_targets
from scene_to_score.features.music import extract, segment_waves

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RATE = 16_000


@pytest.fixture(scope="module")
def clap() -> ClapModel:
    torch.manual_seed(0)
    config = ClapConfig(
        text_config={
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 37,
            "max_position_embeddings": 64,
            "projection_dim": 16,
        },
        audio_config={
            "depths": [1, 1],
            "num_attention_heads": [2, 2],
            "patch_embeds_hidden_size": 16,
            "hidden_size": 32,
            "projection_dim": 16,
            "num_mel_bins": 64,
            "spec_size": 128,
            "window_size": 4,
            "patch_stride": (4, 4),
            "patch_size": 4,
        },
        projection_dim=16,
    )
    return ClapModel(config).eval()


@pytest.fixture
def extractor() -> ClapFeatureExtractor:
    # segments last exactly max_length_s, so rand_trunc never crops: no randomness
    return ClapFeatureExtractor(
        feature_size=64, sampling_rate=RATE, max_length_s=1, truncation="rand_trunc"
    )


def _write_clip(root: Path, clip_id: str, frequency: float) -> None:
    clip = parse_clip_id(clip_id, "train")
    t = np.arange(round(clip.duration_s * RATE)) / RATE
    write_wav(clip.audio_path(root), 0.5 * np.sin(2 * np.pi * frequency * t), RATE)


def test_segment_waves_cut_each_window(tmp_path: Path) -> None:
    clip = parse_clip_id("1_0_0_250", "train")  # 2.5 s: two 1 s segments
    t = np.arange(round(2.5 * RATE)) / RATE
    write_wav(clip.audio_path(tmp_path), np.where(t < 1.0, 0.25, -0.25), RATE)

    waves = segment_waves(tmp_path, segment_clip(clip, 1.0, 1.0), RATE)

    assert [len(w) for w in waves] == [RATE, RATE]
    assert waves[0].mean() == pytest.approx(0.25, abs=1e-3)
    assert waves[1].mean() == pytest.approx(-0.25, abs=1e-3)


def test_segment_waves_resamples(tmp_path: Path) -> None:
    clip = parse_clip_id("1_0_0_100", "train")
    write_wav(clip.audio_path(tmp_path), np.zeros(RATE), RATE)

    (wave,) = segment_waves(tmp_path, segment_clip(clip, 1.0, 1.0), 48_000)

    assert len(wave) == 48_000 and wave.dtype == np.float32


def test_segment_waves_reports_a_missing_file(tmp_path: Path) -> None:
    clip = parse_clip_id("1_0_0_100", "train")
    with pytest.raises(FileNotFoundError, match="just download"):
        segment_waves(tmp_path, segment_clip(clip, 1.0, 1.0), RATE)


def test_extract_writes_aligned_unit_norm_targets(
    tmp_path: Path, clap: ClapModel, extractor: ClapFeatureExtractor
) -> None:
    root = tmp_path / "ossl"
    clip_ids = ["1_0_0_200", "1_1_500_600", "2_0_0_300"]
    for clip_id, frequency in zip(clip_ids, (220.0, 440.0, 880.0), strict=True):
        _write_clip(root, clip_id, frequency)
    segments = [
        s for clip_id in clip_ids for s in segment_clip(parse_clip_id(clip_id, "train"), 1.0, 1.0)
    ]

    ids, z, seconds = extract(clap, extractor, root, segments, batch_size=4)

    assert ids == [s.segment_id for s in segments]
    assert ids[:2] == ["1_0_0_200@0", "1_0_0_200@100"]
    assert z.shape == (6, 16) and z.dtype == np.float32 and seconds > 0
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    # the same sine in two segments of a clip embeds alike, different sines do not
    assert z[0] @ z[1] > z[0] @ z[-1]
    # batching does not change the vectors
    _, z_single, _ = extract(clap, extractor, root, segments, batch_size=1)
    np.testing.assert_allclose(z, z_single, atol=1e-4)

    out = tmp_path / "music" / "clap"
    write_targets(out, "train", ids, z, {"encoder": "clap"})
    read_ids, read_z = read_targets(out, "train")
    assert read_ids == ids
    np.testing.assert_array_equal(read_z, z)


def test_config_composes() -> None:
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("features/music", overrides=["data.n_train_movies=400"])
    assert cfg.model_id == "ucsd-reach/musicldm"
    assert cfg.data.n_train_movies == 400 and cfg.data.segment_s == 10.0
    assert cfg.paths.features_dir
