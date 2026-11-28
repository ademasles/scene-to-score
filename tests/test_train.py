from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scene_to_score.features.cache import Features, write_features, write_targets
from scene_to_score.heads import Head
from scene_to_score.losses import InfoNCE, cosine_loss, winners
from scene_to_score.predictions import load_predictions, prediction_path
from scene_to_score.train import evaluation_rows, recall_at_k, run, val_movies


def _unit(*shape: int, seed: int = 0) -> torch.Tensor:
    return F.normalize(torch.randn(*shape, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_head_returns_unit_vectors_per_hypothesis() -> None:
    h = torch.randn(3, 16)
    assert Head(16, 8, hidden=32)(h).shape == (3, 8)
    z = Head(16, 8, hidden=32, n_hypotheses=4)(h)
    assert z.shape == (3, 4, 8)
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(3, 4))


def test_winners_picks_the_hypothesis_closest_to_the_target() -> None:
    z = _unit(2, 8)
    z_hat = torch.stack([_unit(2, 8, seed=1), z, _unit(2, 8, seed=2)], dim=1)
    torch.testing.assert_close(winners(z_hat, z), z)
    assert cosine_loss(z_hat, z).item() == pytest.approx(0.0, abs=1e-6)


def test_infonce_prefers_aligned_predictions() -> None:
    z = _unit(16, 8)
    loss = InfoNCE()
    assert loss(z, z).item() < loss(_unit(16, 8, seed=3), z).item()


def test_infonce_ignores_negatives_from_the_same_clip() -> None:
    z = _unit(4, 8)
    z[1] = z[0]  # two segments of one clip share their music
    loss = InfoNCE()
    plain = loss(z, z).item()
    masked = loss(z, z, torch.tensor([0, 0, 1, 2])).item()
    assert masked < plain


def test_evaluation_rows_keep_the_middle_segment_of_each_clip() -> None:
    ids = ["1_0_0_3000@1000", "1_0_0_3000@0", "1_0_0_3000@2000", "2_0_0_1000@0"]
    assert evaluation_rows(ids).tolist() == [0, 3]


def test_recall_at_k_on_perfect_and_constant_predictions() -> None:
    z = _unit(20, 8)
    assert recall_at_k(z, z, 1) == 1.0
    constant = z[:1].expand(20, -1)
    assert recall_at_k(constant, z, 10) == pytest.approx(0.5)  # chance level
    tied = z.clone()
    tied[1] = tied[0]  # two identical musics: each one's twin ties with it
    assert recall_at_k(tied, tied, 1) == pytest.approx(18 / 20)  # ties count against


def test_val_movies_reads_the_committed_list_or_draws_ten_percent(tmp_path: Path) -> None:
    movies = [str(m) for m in range(40)]
    drawn = val_movies(tmp_path / "missing.txt", movies)
    assert len(drawn) == 4 and drawn == val_movies(tmp_path / "missing.txt", movies)
    (tmp_path / "val.txt").write_text("3\n7\n")
    assert val_movies(tmp_path / "val.txt", movies) == {"3", "7"}


def _cache(root: Path, split: str, movies: range, rng: np.random.Generator) -> None:
    ids = [f"{m}_{c}_0_3000@{s}" for m in movies for c in range(3) for s in (0, 1000, 2000)]
    z = rng.normal(size=(len(ids), 512)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    tokens = (z[:, None, :32] + 0.1 * rng.normal(size=(len(ids), 4, 32))).astype(np.float32)
    features = Features(ids, tokens.mean(1), tokens, np.ones((len(ids), 4), dtype=bool))
    write_features(root / "video" / "tiny", split, features, {"encoder": "tiny"})
    write_targets(root / "music" / "clap", split, ids, z, {"encoder": "clap"})


def test_run_trains_and_writes_contract_predictions(tmp_path: Path) -> None:
    from hydra import compose, initialize_config_dir

    rng = np.random.default_rng(0)
    _cache(tmp_path / "features", "train", range(10), rng)
    _cache(tmp_path / "features", "test", range(10, 13), rng)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"paths.features_dir={tmp_path / 'features'}",
                f"outputs_dir={tmp_path / 'out'}",
                f"val_movies_file={tmp_path / 'none.txt'}",
                "video_encoder=tiny",
                "device=cpu",
                "d_h=32",
                "hidden=32",
                "batch_size=16",
                "max_epochs=3",
                "n_hypotheses=2",
            ],
        )
    result = run(cfg)
    assert 0.0 <= result["val_r10"] <= 1.0
    for split, n in (("val", 9), ("test", 27)):
        path = prediction_path(tmp_path / "out" / "preds", cfg.method, 0, split, cfg.condition)
        predictions = load_predictions(path)
        assert len(predictions.segment_ids) == n and predictions.n_hypotheses == 2
    assert (tmp_path / "out" / "ckpt" / cfg.method / "seed0.pt").exists()
