import torch


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    del interleaved
    cos, sin = map(lambda t: torch.repeat_interleave(t, 2, dim=-1).unsqueeze(1), (cos, sin))
    return (x * cos) + (rotate_every_two(x) * sin)
