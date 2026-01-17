import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm with zero centered weights (1 + weight).
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()

        self.eps = eps
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        if self.weight is not None:
            output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
