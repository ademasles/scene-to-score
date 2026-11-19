"""Train a fusion + head on cached features and write predictions (contract 1.4 to 1.6).

Run with ``just train``. Until the fusions of ``feat/fusion`` land, the only fusion is
``video_mean``: the masked mean of the video tokens, which is the VideoOnly reference.
"""

import copy
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from scene_to_score.features.cache import read_features, read_targets
from scene_to_score.heads import Head
from scene_to_score.losses import InfoNCE, cosine_loss
from scene_to_score.predictions import Predictions, prediction_path, save_predictions
from scene_to_score.seed import pick_device, seed_everything

log = logging.getLogger(__name__)


def clip_of(segment_id: str) -> str:
    return segment_id.split("@", 1)[0]


def movie_of(segment_id: str) -> str:
    return segment_id.split("_", 1)[0]


def evaluation_rows(ids: Sequence[str]) -> np.ndarray:
    """Row of the middle segment of each clip (same rule as ``ossl.evaluation_segment``)."""
    by_clip: dict[str, list[tuple[int, int]]] = {}
    for row, segment_id in enumerate(ids):
        by_clip.setdefault(clip_of(segment_id), []).append((int(segment_id.rsplit("@", 1)[1]), row))
    rows = [sorted(starts)[len(starts) // 2][1] for starts in by_clip.values()]
    return np.array(sorted(rows))


def val_movies(path: Path, train_movies: Sequence[str], fraction: float = 0.1) -> set[str]:
    """Movies held out for validation: the committed list if any, else 10 % drawn with seed 0."""
    if path.exists():
        return set(path.read_text().split())
    movies = sorted(set(train_movies))
    log.warning("%s is missing: drawing %.0f %% of the train movies", path, 100 * fraction)
    return set(random.Random(0).sample(movies, max(1, round(fraction * len(movies)))))


class VideoMeanPool(nn.Module):
    """VideoOnly stand-in with the Fusion interface: masked mean of the video tokens."""

    def __init__(self, d_video: int, d_h: int = 1024) -> None:
        super().__init__()
        self.d_h = d_h
        self.proj = nn.Sequential(nn.LayerNorm(d_video), nn.Linear(d_video, d_h), nn.GELU())

    def forward(
        self,
        v_tok: torch.Tensor,
        v_mask: torch.Tensor,
        t_tok: torch.Tensor | None = None,
        t_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = v_mask.unsqueeze(-1).to(v_tok.dtype)
        pooled = (v_tok * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return self.proj(pooled)


FUSIONS: dict[str, Callable[..., nn.Module]] = {"video_mean": VideoMeanPool}


class Predictor(nn.Module):
    def __init__(self, fusion: nn.Module, head: Head) -> None:
        super().__init__()
        self.fusion = fusion
        self.head = head

    def forward(self, v_tok: torch.Tensor, v_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.fusion(v_tok, v_mask, None, None))


@dataclass
class Split:
    ids: list[str]
    v_tok: torch.Tensor  # N x L x d, float32
    v_mask: torch.Tensor  # N x L, bool
    z: torch.Tensor  # N x 512, unit norm
    groups: torch.Tensor  # N, one integer per clip

    def subset(self, rows: np.ndarray) -> "Split":
        index = torch.from_numpy(rows)
        return Split(
            [self.ids[i] for i in rows],
            self.v_tok[index],
            self.v_mask[index],
            self.z[index],
            self.groups[index],
        )


def load_split(video_dir: Path, target_dir: Path, split: str) -> Split:
    """Video tokens and targets of one cache split, aligned on segment ids."""
    features = read_features(video_dir, split, mmap=False)
    target_ids, z = read_targets(target_dir, split)
    target_row = {segment_id: row for row, segment_id in enumerate(target_ids)}
    rows = [row for row, segment_id in enumerate(features.ids) if segment_id in target_row]
    missing = len(features.ids) - len(rows)
    if missing:
        log.warning("%s: %d segments have video but no target", split, missing)
    if not rows:
        raise ValueError(f"{split}: no segment has both video features and a target")
    ids = [features.ids[row] for row in rows]
    if features.tokens is None or features.mask is None:
        tokens, mask = features.pooled[rows][:, None], np.ones((len(rows), 1), dtype=bool)
    else:
        tokens, mask = features.tokens[rows], features.mask[rows]
    clips = {clip: i for i, clip in enumerate(dict.fromkeys(clip_of(s) for s in ids))}
    return Split(
        ids=ids,
        v_tok=torch.from_numpy(np.asarray(tokens, dtype=np.float32)),
        v_mask=torch.from_numpy(np.asarray(mask)),
        z=torch.from_numpy(z[[target_row[s] for s in ids]].astype(np.float32)),
        groups=torch.tensor([clips[clip_of(s)] for s in ids]),
    )


@torch.no_grad()
def predict(model: Predictor, split: Split, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    out = [
        model(
            split.v_tok[i : i + batch_size].to(device), split.v_mask[i : i + batch_size].to(device)
        )
        .float()
        .cpu()
        for i in range(0, len(split.ids), batch_size)
    ]
    return torch.cat(out)


def recall_at_k(z_hat: torch.Tensor, z: torch.Tensor, k: int) -> float:
    """Cross-film R@k on the given rows; a K-hypothesis prediction scores a candidate by its
    best hypothesis (no oracle). Ties count against the true music."""
    sims = torch.einsum("nkd,md->nkm", z_hat.view(len(z_hat), -1, z.shape[1]), z).amax(1)
    true = sims.diagonal()
    rank = (sims > true[:, None]).sum(1) + (sims == true[:, None]).sum(1) - 1
    return float((rank < k).float().mean())


def fit(
    model: Predictor,
    train: Split,
    val: Split,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[Predictor, float]:
    """AdamW with early stopping on validation R@10; returns the best model and its R@10."""
    criterion = InfoNCE().to(device) if cfg.objective == "infonce" else None
    params = list(model.parameters()) + ([] if criterion is None else list(criterion.parameters()))
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    val_rows = evaluation_rows(val.ids)
    generator = torch.Generator().manual_seed(cfg.seed)
    best, best_state, stale = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(len(train.ids), generator=generator)
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            if len(rows) < 2:
                continue
            z_hat = model(train.v_tok[rows].to(device), train.v_mask[rows].to(device))
            z = train.z[rows].to(device)
            if criterion is None:
                loss = cosine_loss(z_hat, z)
            else:
                loss = criterion(z_hat, z, train.groups[rows].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        z_val = predict(model, val, device, cfg.batch_size)
        r10 = recall_at_k(z_val[val_rows], val.z[val_rows], 10)
        log.info("epoch %d: loss %.4f, val R@10 %.4f", epoch, float(np.mean(losses)), r10)
        if r10 > best:
            best, best_state, stale = r10, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best


def build(cfg: DictConfig, d_video: int) -> Predictor:
    fusion = FUSIONS[cfg.fusion](d_video, cfg.d_h)
    head = Head(cfg.d_h, hidden=cfg.hidden, n_hypotheses=cfg.n_hypotheses, dropout=cfg.dropout)
    return Predictor(fusion, head)


def run(cfg: DictConfig) -> dict[str, float]:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    features_dir = Path(cfg.paths.features_dir)
    video_dir = features_dir / "video" / cfg.video_encoder
    target_dir = features_dir / "music" / "clap"
    full_train = load_split(video_dir, target_dir, "train")
    held_out = val_movies(Path(cfg.val_movies_file), [movie_of(s) for s in full_train.ids])
    is_val = np.array([movie_of(s) in held_out for s in full_train.ids])
    train, val = (
        full_train.subset(np.flatnonzero(~is_val)),
        full_train.subset(np.flatnonzero(is_val)),
    )
    test = load_split(video_dir, target_dir, "test")
    log.info("segments: %d train, %d val, %d test", len(train.ids), len(val.ids), len(test.ids))

    model = build(cfg, train.v_tok.shape[-1]).to(device)
    model, best = fit(model, train, val, cfg, device)

    out = Path(cfg.outputs_dir)
    ckpt = out / "ckpt" / cfg.method / f"seed{cfg.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "val_r10": best}, ckpt)
    for name, split in (("val", val), ("test", test)):
        z_hat = predict(model, split, device, cfg.batch_size).numpy().astype(np.float32)
        path = prediction_path(out / "preds", cfg.method, cfg.seed, name, cfg.condition)
        save_predictions(path, Predictions(split.ids, z_hat))
    log.info("best val R@10 %.4f; predictions in %s", best, out / "preds" / cfg.method)
    return {"val_r10": best}


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
