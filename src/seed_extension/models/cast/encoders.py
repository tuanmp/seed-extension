"""Pluggable hit context encoders."""

import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Pass-through encoder — no contextual processing of hits."""

    def forward(self, hit_emb: "torch.Tensor") -> "torch.Tensor":
        return hit_emb


class BinnedSelfAttentionEncoder(nn.Module):
    """Self-attention within spatial bins. Placeholder for future ablation."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError("BinnedSelfAttentionEncoder not yet implemented")
