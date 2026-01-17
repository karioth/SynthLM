import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        """
        Feed-forward SwiGLU blocks
        """
        super().__init__()
        self.up_gate = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down_proj(up * self.act(gate))
