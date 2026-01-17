import math
from typing import Optional

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        time_factor: float = 1000.0,
    ) -> None:
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even.")

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        # cache freqs to avoid recomputing them every time
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)  # (half,) fp32
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match DiT-style init for the timestep MLP.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N,) or (N,1) tensor, int or float, can be fractional.
        returns: (N, frequency_embedding_size) in t.dtype
        """
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.reshape(-1)

        t = t * self.time_factor # scale 0-1 to original ddpm discrete scales as in Flux
        args = t.float()[:, None] * self.freqs[None, :]              # (N, half) fp32
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, 2*half)

        return emb.to(dtype=t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations. Also handles label dropout for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
