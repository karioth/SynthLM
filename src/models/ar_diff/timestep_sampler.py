import torch

def sample_monotone_anchor_times(
    L: int,
    scale: float= 1.0,
    bias: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Continuous timestep sampler for a monotone per-position diffusion time schedule.

    Returns:
      u: (L,) tensor with 0 <= u[0] <= ... <= u[L-1] <= 1
      k: anchor index (python int)
      a: anchor timestep value (scalar tensor)

    Main idea:
      1) Pick an anchor index k uniformly from {0..L-1}.
      2) Pick an anchor time by sampling logit-normal.
      3) Sample a random monotone prefix for positions [0..k-1], constrained to be <= a.
      4) Sample a random monotone suffix for positions [k+1..L-1], constrained to be >= a.
    """
    device = device or torch.device("cpu")

    # 1) Anchor position
    k_t = torch.randint(0, L, size=(), device=device)
    k = int(k_t.item())

    # 2) Anchor value
    a = torch.sigmoid(torch.randn((), device=device, dtype=dtype) * scale + bias) 

    u = torch.empty((L,), device=device, dtype=dtype)
    u[k] = a

    # 3) Prefix (<= a), then sort to enforce monotonicity
    if k > 0:
        prefix = torch.rand((k,), device=device, dtype=dtype) * a  # in [0, a]
        prefix, _ = torch.sort(prefix)
        u[:k] = prefix

    # 4) Suffix (>= a), then sort to enforce monotonicity
    n_after = L - (k + 1)
    if n_after > 0:
        suffix = a + torch.rand((n_after,), device=device, dtype=dtype) * (1.0 - a)  # in [a, 1]
        suffix, _ = torch.sort(suffix)
        u[k + 1:] = suffix

    return u, k, a
