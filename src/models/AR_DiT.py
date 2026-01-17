import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_func
    has_flash_attn2 = torch.cuda.get_device_properties(0).major >= 8
except ImportError:
    has_flash_attn2 = False
    print("flash_attn2 not found")

from .DiT import LabelEmbedder, TimestepEmbedder, FinalLayer, SwiGLU, modulate
from .rotary_torch import apply_rotary_pos_emb as apply_rotary_emb
from .RMSNorm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim + 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, start_pos, rel_pos, incremental_state=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q, k, v = torch.split(qkv, [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2)
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)
        if incremental_state is not None:
            incremental_state["key"][:B, start_pos : start_pos + N] = k
            incremental_state["value"][:B, start_pos : start_pos + N] = v
            k = incremental_state["key"][:B, :start_pos + N]
            v = incremental_state["value"][:B, :start_pos + N]
        if has_flash_attn2 and (x.dtype == torch.float16 or x.dtype == torch.bfloat16):
            x = flash_attn_func(q, k, v, causal=True, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q.transpose(1, 2)
            k = repeat_kv(k.transpose(1, 2), self.n_rep)
            v = repeat_kv(v.transpose(1, 2), self.n_rep)
            x = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=incremental_state is None,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2)

        x = self.proj(x.reshape(B, N, C))
        x = self.proj_drop(x)
        return x


class AR_DiTBlock(nn.Module):
    """
    A Autoregressive DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, mlp_ratio=4.0, proj_drop=0., attn_drop=0., **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, qkv_bias=False, proj_drop=proj_drop, attn_drop=attn_drop, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3 / 64) * 64
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=False)
        )

    def forward(self, x, c, start_pos, rel_pos, incremental_state=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), start_pos, rel_pos, incremental_state)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class AR_DiT(nn.Module):
    """
    Diffusion model with an Autoregressive Transformer backbone.
    """
    def __init__(
        self,
        seq_len=1024,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        num_kv_heads=None,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        drop=0.0,
        posi_scale=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.posi_scale = posi_scale
        self.seq_len = seq_len

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self._precomputed_freqs_cis = None

        self.blocks = nn.ModuleList([
            AR_DiTBlock(hidden_size, self.num_heads, self.num_kv_heads, mlp_ratio=mlp_ratio, proj_drop=drop, attn_drop=drop) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()
        
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def build_rel_pos(self, x, start_pos = 0):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / ((10000 * self.posi_scale) ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(x.size(1)).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))

        return rel_pos

    def forward(self, x_noise, t, y, **kwargs):
        """
        Forward pass of DiT.
        x_noise: (B, T, C) tensor of noisy latent tokens
        t: (B, T) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        assert t.dim() == 2, "AR_DiT expects tokenwise t with shape (B, T)"

        x = self.x_embedder(x_noise)             # (B, T, D)
        y = self.y_embedder(y, self.training)    # (B, D)

        t = t.contiguous()
        t_emb = self.t_embedder(t.view(-1)).view(x.size(0), x.size(1), -1)  # (B, T, D)

        # class token is not time‑modulated (AR‑Diffusion style)
        t_cls = torch.zeros(x.size(0), 1, t_emb.size(-1), device=t_emb.device, dtype=t_emb.dtype)
        c = torch.cat([t_cls, t_emb], dim=1)  # (B, T+1, D)

        x = torch.cat([y.unsqueeze(1), x], dim=1)  # (B, T+1, D)

        rel_pos = self.build_rel_pos(x)

        for block in self.blocks:
            x = block(x, c, start_pos=0, rel_pos=rel_pos) # (B, T, D)
        # remove conditioning token before final layer
        x = x[:, 1:, :] 
        c = c[:, 1:, :]
        
        x = self.final_layer(x, c) # (B, T, out_channels)

        return x
    
    def sample_with_cfg(self, y, cfg_scale, sample_func):
        bsz = y.shape[0]
        z = torch.randn(bsz, self.seq_len, self.in_channels, device=self.device, dtype=self.dtype)
        samples = sample_func(functools.partial(self.forward_with_cfg, y=y, cfg_scale=cfg_scale), z)
        samples, _ = samples.chunk(2, dim=0)
        return samples

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, y)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


#################################################################################
#                                  AR-DiT Configs                               #
#################################################################################

def AR_DiT_XL(**kwargs):
    return AR_DiT(depth=24, hidden_size=2048, num_heads=16, **kwargs)

def AR_DiT_Large(**kwargs):
    return AR_DiT(depth=24, hidden_size=1536, num_heads=12, **kwargs)

def AR_DiT_Medium(**kwargs):
    return AR_DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def AR_DiT_Base(**kwargs):
    return AR_DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

AR_DiT_models = {
    "AR-DiT-XL": AR_DiT_XL,
    "AR-DiT-Large": AR_DiT_Large,
    "AR-DiT-Medium": AR_DiT_Medium,
    "AR-DiT-Base": AR_DiT_Base,
}
