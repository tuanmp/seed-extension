import math

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.d_model = d_model
        self.use_detector_features = use_detector_features
        self.use_cylindrical_pe = use_cylindrical_pe

        self.fourier = FourierEncode(L=fourier_L, input_dim=3)
        input_dim = self.fourier.output_dim

        if use_cylindrical_pe:
            self.cylindrical_fourier = FourierEncode(L=fourier_L, input_dim=3)
            input_dim += self.cylindrical_fourier.output_dim

        if use_detector_features:
            self.layer_embed = nn.Embedding(20, 16)
            self.volume_embed = nn.Embedding(14, 16)
            self.detector_embed = nn.Embedding(2, 16)
            input_dim += 48

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hits: torch.Tensor) -> torch.Tensor:
        xyz = hits[..., :3]

        xyz_norm = torch.stack(
            [
                xyz[..., 0] / 1100.0 * 0.5 + 0.5,
                xyz[..., 1] / 1100.0 * 0.5 + 0.5,
                xyz[..., 2] / 3000.0 * 0.5 + 0.5,
            ],
            dim=-1,
        )
        xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0)

        features = [self.fourier(xyz_norm)]

        if self.use_cylindrical_pe:
            r = torch.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
            phi = torch.atan2(xyz[..., 1], xyz[..., 0]) / (2 * math.pi) + 0.5
            z = xyz_norm[..., 2]
            cylindrical = torch.stack([r / 1100.0 * 0.5 + 0.5, phi, z], dim=-1)
            cylindrical = torch.clamp(cylindrical, 0.0, 1.0)
            features.append(self.cylindrical_fourier(cylindrical))

        if self.use_detector_features:
            layer_id = hits[..., 3].long()
            volume_id = hits[..., 4].long()
            detector = hits[..., 5].long()
            features.append(self.layer_embed(layer_id))
            features.append(self.volume_embed(volume_id))
            features.append(self.detector_embed(detector))

        x = torch.cat(features, dim=-1)
        return self.mlp(x)


class SeedEmbedder(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, seeds: torch.Tensor) -> torch.Tensor:
        return self.mlp(seeds)
