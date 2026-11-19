"""Prediction head (interface 1.4): fused features h (B x d_h) -> z_hat on the unit sphere."""

import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    """MLP from h to K unit-norm hypotheses; returns B x d for K = 1, else B x K x d."""

    def __init__(
        self,
        d_in: int,
        d_out: int = 512,
        *,
        hidden: int = 1024,
        n_hypotheses: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_hypotheses < 1:
            raise ValueError(f"n_hypotheses must be >= 1, got {n_hypotheses}")
        self.d_out = d_out
        self.n_hypotheses = n_hypotheses
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out * n_hypotheses),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.mlp(h)
        if self.n_hypotheses > 1:
            z = z.view(len(h), self.n_hypotheses, self.d_out)
        return F.normalize(z, dim=-1)
