import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding
from .norms import RMSNorm

class Attention(nn.Module):
    """
    Flash-attn based attention with optional KV cache and rotary embeddings.

    Expects q/k/v layout as (B, S, H, D). For rotary, this uses flash-attn's
    packed-QKV rotary path (Hq + 2*Hkv) and applies RoPE only to the new chunk.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layer_idx: int,
        num_kv_heads: int | None = None,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = False,
        rope_scale_base: float | None = None,
        is_gated: bool = True,
        is_causal: bool = True,
    ):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.head_size = hidden_size // num_heads
        self.layer_idx = int(layer_idx)

        self.is_causal = is_causal
        self.is_gated = is_gated
        self.qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        proj_out = self.qkv_size + (self.hidden_size if self.is_gated else 0)

        self.qkvg = nn.Linear(self.hidden_size, proj_out, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(hidden_size=self.head_size, elementwise_affine=False)
        self.k_norm = RMSNorm(hidden_size=self.head_size, elementwise_affine=False)

        self.rotary = RotaryEmbedding(
            dim=self.head_size,
            base=rope_theta,
            interleaved=rope_interleaved,
            scale_base=rope_scale_base,
        )

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_kv_heads,
            self.head_size,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv: torch.Tensor, inference_params):
        """
        kv: (B, L_new, 2, H_kv, Dh)
        returns: (B, L_total, 2, H_kv, Dh)
        """
        B, L_new = kv.shape[0], kv.shape[1]

        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                inference_params.max_batch_size,
                inference_params.max_seqlen,
                dtype=kv.dtype,
            )

        kv_cache = inference_params.key_value_memory_dict[self.layer_idx]  # (maxB, maxL, 2, Hkv, Dh)

        batch_start = int(getattr(inference_params, "batch_size_offset", 0))
        batch_end = batch_start + B
        seq_start = int(inference_params.seqlen_offset)
        seq_end = seq_start + L_new

        # (optional but nice) bounds checks
        assert batch_end <= kv_cache.shape[0]
        assert seq_end <= kv_cache.shape[1]

        kv_cache[batch_start:batch_end, seq_start:seq_end] = kv
        return kv_cache[batch_start:batch_end, :seq_end]

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        B, L, d_model = hidden_states.shape
        proj = self.qkvg(hidden_states)

        if self.is_gated:
            qkv_flat, gate = proj.split([self.qkv_size, self.hidden_size], dim=-1)
            gate = gate.sigmoid()
        else:
            qkv_flat, gate = proj, None

        qkv = qkv_flat.view(B, L, self.num_heads + 2 * self.num_kv_heads, self.head_size)
        q, k_new, v_new = torch.split(qkv, [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2)

        # qk-norm (per-head)
        q = self.q_norm(q)
        k_new = self.k_norm(k_new)

        # RoPE on just the new chunk using offset (flash packed-QKV path).
        start = 0 if inference_params is None else int(inference_params.seqlen_offset)
        # flash-attn 2.6.x expects (B, S, 3, H, D). Upgrade to use 4D packed QKV + num_heads_q.
        # qkv_new = torch.cat([q, k_new, v_new], dim=2)  # (B, L, Hq+2Hkv, Dh)
        # qkv_new = self.rotary(qkv_new, seqlen_offset=start, num_heads_q=self.num_heads)
        # q, k_new, v_new = torch.split(
        #     qkv_new,
        #     [self.num_heads, self.num_kv_heads, self.num_kv_heads],
        #     dim=2,
        # )
        qkv_new = torch.stack([q, k_new, v_new], dim=2)  # (B, L, 3, H, Dh)
        qkv_new = self.rotary(qkv_new, seqlen_offset=start)
        q, k_new, v_new = qkv_new.unbind(dim=2)

        if inference_params is not None:
            kv_new = torch.stack([k_new, v_new], dim=2)   # (B, L, 2, Hkv, Dh)
            kv = self._update_kv_cache(kv_new, inference_params)
            k, v = kv.unbind(dim=2)                       # (B, L_total, Hkv, Dh)
        else:
            k, v = k_new, v_new

        # Flash-attn expects q/k/v as (B, S, H, D).
        out = flash_attn_func(
            q, k, v,
            causal=self.is_causal,
        )
        out = out.reshape(B, L, d_model).contiguous()
        if gate is not None:
            out = out * gate
        return self.out_proj(out)
