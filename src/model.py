"""
Flow Matching models:
  - FlowMatchingMLP: standard 6-layer MLP for flow matching
  - MeanFlowMLP: extended with horizon embedding for MeanFlow (Part 4)
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t in [0, 1] to a fixed 128-dim embedding.
    No trainable parameters.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        k = embed_dim // 2  # 64
        # ω_i = exp(-i * ln(10000) / (k-1)), i = 0, 1, ..., k-1
        freqs = torch.exp(-torch.arange(k, dtype=torch.float32) * math.log(10000) / (k - 1))
        # Register as buffer: not a parameter, but moves with device
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) scalar timesteps in [0, 1]
        Returns:
            (batch, 128) sinusoidal embedding
        """
        # t: (batch,) -> (batch, 1) for broadcasting with freqs (64,)
        args = t.unsqueeze(-1) * self.freqs  # (batch, 64)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, 128)


class FlowMatchingMLP(nn.Module):
    """
    6-layer MLP for flow matching.
    Input: (zt, t) -> output: D-dim prediction (v or x depending on training).

    Architecture:
        - Sinusoidal time embedding: t -> 128-dim
        - Concat [zt; et] -> D+128
        - 5 hidden layers: Linear -> ReLU (256 units each)
        - 1 output layer: Linear -> no activation (256 -> D)
    """

    def __init__(self, data_dim: int):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=128)

        self.net = nn.Sequential(
            # Hidden layer 1: D+128 -> 256
            nn.Linear(data_dim + 128, 256),
            nn.ReLU(),
            # Hidden layers 2-5: 256 -> 256
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # Output layer: 256 -> D, no activation
            nn.Linear(256, data_dim),
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zt: (batch, D) noisy sample
            t:  (batch,) timestep in [0, 1]
        Returns:
            (batch, D) prediction
        """
        et = self.time_embed(t)                       # (batch, 128)
        x = torch.cat([zt, et], dim=-1)               # (batch, D+128)
        return self.net(x)                             # (batch, D)


class MeanFlowMLP(nn.Module):
    """
    MeanFlow variant: 6-layer MLP with two sinusoidal embeddings.
    Input: (zt, t, h) where h = t - r is the horizon.
    Architecture:
        - Sinusoidal time embedding: t -> 128-dim
        - Sinusoidal horizon embedding: h -> 128-dim (separate parameters)
        - Concat [zt; et; eh] -> D+256
        - 5 hidden layers: Linear -> ReLU (256 units each)
        - 1 output layer: Linear -> no activation (256 -> D)
    Output is interpreted as velocity.
    """

    def __init__(self, data_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=128)
        self.horizon_embed = SinusoidalTimeEmbedding(embed_dim=128)

        layers = []
        layers.extend([nn.Linear(data_dim + 256, hidden_dim), nn.ReLU()])
        for _ in range(4):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        et = self.time_embed(t)          # (batch, 128)
        eh = self.horizon_embed(h)       # (batch, 128)
        x = torch.cat([zt, et, eh], dim=-1)  # (batch, D+256)
        return self.net(x)               # (batch, D)
