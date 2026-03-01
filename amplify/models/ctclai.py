"""CTCLAI auxiliary predictors.

This module implements two lightweight heads used only during inference-time
reranking (and their supervised training):

  1) Action -> token logits over the motion-token vocabulary for the next
     `token_seq_len` steps.
  2) Action -> per-step risk logits over the action horizon.

The heads are intentionally small and *do not* modify any AMPLIFY staged
components (motion tokenizer / forward dynamics / inverse dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CTCLAIInferenceConfig:
    """Runtime knobs for CTCLAI reranking."""

    enabled: bool = False
    n_samples: int = 8

    lambda_tok: float = 1.0
    lambda_risk: float = 0.5
    lambda_prior: float = 0.1

    # Token-consistency robustness: downweight high-entropy token predictions.
    entropy_weighted_tok: bool = True

    # Discount applied to the per-step risk vector.
    risk_discount: float = 0.99


class CTCLAIHeads(nn.Module):
    """Two small predictors used for CTCLAI candidate reranking."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_dim: int,
        action_horizon: int,
        token_seq_len: int,
        codebook_size: int,
        action_mlp_hidden: Optional[int] = None,
        action_mlp_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.token_seq_len = int(token_seq_len)
        self.codebook_size = int(codebook_size)

        h = int(action_mlp_hidden or hidden_dim)

        # Per-timestep action encoder: (A) -> (H)
        a_layers = []
        in_dim = action_dim
        for _ in range(max(1, int(action_mlp_layers))):
            a_layers.append(nn.Linear(in_dim, h))
            a_layers.append(nn.GELU())
            if dropout > 0:
                a_layers.append(nn.Dropout(dropout))
            in_dim = h
        a_layers.append(nn.Linear(in_dim, hidden_dim))
        self.action_step_encoder = nn.Sequential(*a_layers)

        # Fuse (img_feat, proprio_feat) -> (H)
        self.state_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Per-step fusion: (state, action_step) -> hidden
        self.step_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Shared output projections
        self.token_out = nn.Linear(hidden_dim, codebook_size)
        self.risk_out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        img_feat: torch.Tensor,
        proprio_feat: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            img_feat: (B, D)
            proprio_feat: (B, D)
            action_chunk: (B, T, A)

        Returns:
            token_logits: (B, token_seq_len, codebook_size)
            risk_logits: (B, action_horizon)
        """
        if action_chunk.dim() != 3:
            raise ValueError(f"action_chunk must be (B,T,A); got shape {tuple(action_chunk.shape)}")
        # Encode actions per-timestep.
        a_step = self.action_step_encoder(action_chunk)  # (B, T, H)

        # Fuse state.
        s = self.state_fuser(torch.cat([img_feat, proprio_feat], dim=-1))  # (B, H)
        s_step = s.unsqueeze(1).expand(-1, a_step.size(1), -1)  # (B, T, H)

        step_h = self.step_fuser(torch.cat([s_step, a_step], dim=-1))  # (B, T, H)

        # Token logits are predicted for the first `token_seq_len` steps.
        token_h = step_h[:, : self.token_seq_len]  # (B, L, H)
        token_logits = self.token_out(token_h)  # (B, L, V)

        # Risk logits predicted for the full action horizon.
        risk_logits = self.risk_out(step_h).squeeze(-1)  # (B, T)
        return token_logits, risk_logits
