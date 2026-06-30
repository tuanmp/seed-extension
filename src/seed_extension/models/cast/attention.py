import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, kv, return_attention=False):
        attn_out, attn_weights = self.attn(query=queries, key=kv, value=kv)
        queries = queries + self.dropout(attn_out)
        queries = self.norm1(queries)
        queries = queries + self.ffn(queries)
        queries = self.norm2(queries)
        if return_attention:
            return queries, attn_weights
        return queries


class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ff_dim=1024, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, seed_emb, hit_emb, return_attention=False):
        attn = None
        for layer in self.layers:
            if return_attention and layer is self.layers[-1]:
                seed_emb, attn = layer(seed_emb, hit_emb, return_attention=True)
            else:
                seed_emb = layer(seed_emb, hit_emb)
        if return_attention:
            return seed_emb, attn
        return seed_emb


class SeedSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, seed_emb):
        attn_out, _ = self.attn(query=seed_emb, key=seed_emb, value=seed_emb)
        seed_emb = seed_emb + attn_out
        seed_emb = self.norm(seed_emb)
        return seed_emb
