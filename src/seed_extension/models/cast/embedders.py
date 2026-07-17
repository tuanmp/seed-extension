import math

import torch
import torch.nn as nn

NORMALIZATION_XYZ = [1.05, 1.05, 3.05]

def map_layer_id_to_index(layer_id: torch.Tensor, dataset: str) -> torch.Tensor:
    """Map layer_id to a 0-based index for embedding.

    ColliderML layer_id is 2, 4, 6, ..., 16. We map it to 0, 1, 2, ..., 7.
    """
    if dataset == "colliderml":
        return (layer_id / 2 - 1).long()
    return layer_id.long()  # For other datasets, return as is (or implement other mappings as needed) 

class FourierEncode(nn.Module):
    def __init__(self, L: int = 8, input_dim: int = 3):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim * 2 * L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(self.L):
            freq = (2**i) * math.pi
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class HitEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        fourier_L: int = 8,
        use_detector_features: bool = True,
        use_cylindrical_pe: bool = True,
        dropout: float = 0.1,
        dataset: str = "colliderml",
    ):
        super().__init__()
        self.d_model = d_model
        self.use_detector_features = use_detector_features
        self.use_cylindrical_pe = use_cylindrical_pe
        self._dataset = dataset

        self.fourier = FourierEncode(L=fourier_L, input_dim=3)
        input_dim = self.fourier.output_dim

        if use_cylindrical_pe:
            self.cylindrical_fourier = FourierEncode(L=fourier_L, input_dim=2)
            input_dim += self.cylindrical_fourier.output_dim

        if use_detector_features:
            self.layer_embed = nn.Embedding(num_embeddings=8, embedding_dim=16)
            self.detector_embed = nn.Embedding(num_embeddings=9, embedding_dim=16)
            input_dim += 32

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hits: torch.Tensor, detector_info: torch.Tensor) -> torch.Tensor:
        xyz = hits[..., :3]

        norm_factor = torch.tensor(NORMALIZATION_XYZ, device=xyz.device, dtype=xyz.dtype).reshape(1, 3)

        xyz_norm = xyz / norm_factor * 0.5 + 0.5

        features = []

        if self.use_cylindrical_pe:
            r = torch.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2) 
            phi = torch.atan2(xyz[..., 1], xyz[..., 0]) / (2 * math.pi) + 0.5
            cylindrical = torch.stack([r, phi], dim=-1)
            cylindrical = torch.clamp(cylindrical, 0.0, 1.0)
            features.append(self.cylindrical_fourier(cylindrical))

        # Ensure the normalized coordinates are within [0, 1]
        xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)

        features.append(self.fourier(xyz_norm))

        if self.use_detector_features:
            # layer id in ColliderML is indexed as 2 to 16
            layer_id = map_layer_id_to_index(detector_info[..., 0], self._dataset)
            detector = detector_info[..., 1].long()
            features.append(self.layer_embed(layer_id))
            features.append(self.detector_embed(detector))

        x = torch.cat(features, dim=-1)
        return self.mlp(x)


class SeedEmbedder(nn.Module):
    """Encodes seed triplets (3 hits x 3 coords = 9 dims) into embeddings.

    Coordinates are normalized to [0,1] using the same detector bounds
    as HitEmbedder before the MLP.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hit_x: torch.Tensor, seed_hit_idx: torch.Tensor) -> torch.Tensor:

        """Gather the features of the seed hits from hit_x and encode them.
            Assumes hit_x is of shape (B, N_h, F_hit) and seed_hit_idx is of shape (B, N_s, 3).
            Returns a tensor of shape (B, N_s, d_model).
            Input:
            hit_x: (B, N_h, F_hit) features of all hits
            seed_hit_idx: (B, N_s, 3) indices of the seed hits
        """
        # seed_hit_idx: (B, N_s, 3) indices of the seed hits in the hit_x tensor
        # hit_x: (B, N_h, F_hit) features of all hits

        B, N_s, _ = seed_hit_idx.shape
        F_hit = hit_x.shape[-1]
        # Flatten seed indices: (B, N_s*3) — each index points to one hit row
        idx_flat = seed_hit_idx.reshape(B, -1)                          # (B, N_s*3)

        # Expand to cover all F_hit features (gather needs matching last dim)
        idx_gather = idx_flat[..., None].expand(-1, -1, F_hit)          # (B, N_s*3, F_hit)

        # Gather along dim=1 (hit dimension)
        seed_feats = torch.gather(hit_x, dim=1, index=idx_gather)       # (B, N_s*3, F_hit)

        # Reshape back: (B, N_s, 3, F_hit) → (B, N_s, 3, F_hit)
        seed_feats = seed_feats.reshape(B, N_s, 3, F_hit)

        # sum the seed hit features to get a single feature vector per seed
        seed_feats = seed_feats.sum(dim=2)  # (B, N_s, F_hit)

        return self.mlp(seed_feats)  # (B, N_s, d_model)

    
