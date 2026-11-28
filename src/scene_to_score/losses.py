"""Training objectives for the head that predicts CLAP music embeddings.

Every loss takes unit-norm predictions ``z_hat`` (B x d, or B x K x d with K hypotheses) and
unit-norm targets ``z`` (B x d). With K hypotheses only the hypothesis closest to its target
(winner-takes-all, chosen without gradient) is trained.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def winners(z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The hypothesis of each row closest to its target: B x K x d -> B x d."""
    if z_hat.ndim == 2:
        return z_hat
    with torch.no_grad():
        best = torch.einsum("bkd,bd->bk", z_hat, z).argmax(dim=1)
    return z_hat[torch.arange(len(z_hat), device=z_hat.device), best]


def cosine_loss(z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Mean of 1 - cos(z_hat, z); no negatives, so it can pull every prediction to the mean."""
    return (1.0 - (winners(z_hat, z) * z).sum(-1)).mean()


class InfoNCE(nn.Module):
    """Symmetric InfoNCE with a learned temperature.

    Pairs from the same clip share their music, so they are dropped from each other's negatives
    (``groups`` holds one integer per row, equal for rows of the same clip).
    """

    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_log_scale = math.log(max_scale)

    @property
    def temperature(self) -> float:
        return float(torch.exp(-self.log_scale.detach().clamp(max=self.max_log_scale)))

    def forward(
        self, z_hat: torch.Tensor, z: torch.Tensor, groups: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred = winners(z_hat, z)
        scale = self.log_scale.clamp(max=self.max_log_scale).exp()
        logits = scale * pred @ z.T  # B x B, positives on the diagonal
        if groups is not None:
            same = groups[:, None] == groups[None, :]
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, float("-inf"))
        labels = torch.arange(len(z), device=z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
