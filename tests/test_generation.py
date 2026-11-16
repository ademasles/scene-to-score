import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput
from transformers import ClapConfig, ClapFeatureExtractor, ClapModel, RobertaTokenizer

from scene_to_score.generation import (
    audio_embedding,
    cosine_matrix,
    empty_text_embedding,
    generate,
    round_trip_report,
)


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


def test_audio_embedding_is_unit_norm_per_wave(clap: ClapModel) -> None:
    extractor = ClapFeatureExtractor(feature_size=64, max_length_s=1, truncation="rand_trunc")
    rng = np.random.default_rng(0)
    waves = [0.1 * rng.normal(size=n).astype(np.float32) for n in (48_000, 30_000)]
    z = audio_embedding(clap, extractor, waves)
    assert z.shape == (2, 16) and z.dtype == torch.float32
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(2))


def test_empty_text_embedding_is_one_unit_vector(clap: ClapModel, tmp_path: Path) -> None:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "a": 4}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = RobertaTokenizer(
        vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt")
    )
    z = empty_text_embedding(clap, tokenizer)
    assert z.shape == (1, 16)
    assert z.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_generate_passes_embeddings_and_never_a_prompt() -> None:
    calls = {}

    class FakePipe:
        device = torch.device("cpu")
        text_encoder = SimpleNamespace(dtype=torch.float32)

        def __call__(self, **kwargs: object) -> AudioPipelineOutput:
            calls.update(kwargs)
            n = kwargs["prompt_embeds"].shape[0]  # type: ignore[attr-defined]
            return AudioPipelineOutput(audios=np.zeros((n, 160)))

    pipe = FakePipe()
    z = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    audio = generate(pipe, z, torch.zeros(1, 8), seconds=5.0, steps=4, seed=1)  # type: ignore[arg-type]
    assert audio.shape == (3, 160) and audio.dtype == np.float32
    assert "prompt" not in calls
    assert calls["negative_prompt_embeds"].shape == (3, 8)
    assert (calls["audio_length_in_s"], calls["num_inference_steps"]) == (5.0, 4)


def test_generate_rejects_mismatched_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        generate(object(), torch.zeros(2, 8), torch.zeros(2, 8))  # type: ignore[arg-type]


def test_round_trip_report_ranks_the_own_vector() -> None:
    similarity = np.array([[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.5, 0.6, 0.4]])
    report = round_trip_report(similarity)
    assert report["own_cosine"] == pytest.approx(0.7)
    assert report["mean_rank"] == pytest.approx(5 / 3)
    assert report["top1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        round_trip_report(np.ones((2, 3)))


def test_cosine_matrix_normalises_rows() -> None:
    a = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    np.testing.assert_allclose(cosine_matrix(a, a), np.eye(2), atol=1e-6)


def test_smoke_config_composes() -> None:
    from hydra import compose, initialize_config_dir

    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="musicldm_smoke")
    assert cfg.n_mean >= cfg.n_generate >= 1
    assert cfg.data.segment_s == 10.0
