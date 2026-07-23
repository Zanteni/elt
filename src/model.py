"""
VAE-stage skeleton - Stage 1 of elt-baseline.
Two model variants share this file: RoPE-ViT-VAE (priority) and sincos-ViT-VAE.
Fill in top-down: patchify -> patch embed -> pos-encoding -> attention -> blocks -> VAE.

Ownership convention: VAEEncoder and VAEDecoder each build and own their own
rope_cache_2d (registered as a buffer in __init__), since they may operate on
different spatial grids. Neither forward() takes rope_cache_2d as an argument
from outside -- it's threaded internally to the backbone/blocks/attention.
"""

import torch
import torch.nn as nn
import  math


# ---------------------------------------------------------------------------
# 1. Patchify / Unpatchify
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Reused from MAE, adapted for latent channel count.
    x: (B, C, H, W) -> (B, N, patch_dim), N=(H/P)*(W/P), patch_dim=C*P*P
    """
    assert x.ndim ==3 or x.ndim==4,f"EXpected 3D or 4D tensor,got {x.ndim}."
    if x.ndim ==3:
        x = x.unsqueeze(0)
    B,C,H,W = x.shape
    assert H%patch_size == 0,f"H must be divided by the patch size ,got H:{H}, P:{patch_size}"
    assert W%patch_size == 0,f"H must be divided by the patch size ,got H:{H}, P:{patch_size}"
    N = (W//patch_size)*(H//patch_size)
    d_patch = patch_size**2*C
    x = x.reshape(B,C,H//patch_size,patch_size,W//patch_size,patch_size)
    x = x.permute(0,2,4,3,5,1)
    x=x.reshape(B,N,d_patch).contiguous()
    return x


def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int, h: int, w: int) -> torch.Tensor:
    """(B, N, patch_dim) -> (B, C, H, W). Inverse of patchify."""

    assert x.ndim ==3 or x.ndim==2,f"EXpected 2D or 3D tensor,got {x.ndim}."
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B,N,patch_dim = x.shape
    assert patch_dim%out_channels ==0,f"The patch_dim must be devided by the output_chanels,got C:{out_channels},patch_dim:{patch_dim}."
    assert h%patch_size == 0,f"H must be divided by the patch size ,got H:{h}, P:{patch_size}"
    assert w%patch_size == 0,f"H must be divided by the patch size ,got W:{w}, P:{patch_size}"
    assert (h//patch_size)*(w//patch_size) == N,f" the height,width and the patch  size should give the same number of token as the input ,gotH:{h},W:{w}, P:{patch_size} and N:{N} "
    x = x.reshape(B,h//patch_size,w//patch_size,patch_size,patch_size,out_channels)
    x = x.permute(0,5,1,3,2,4)
    x = x.reshape(B,out_channels,h,w).contiguous()
    return x



# ---------------------------------------------------------------------------
# 2. Patch Embedding / Patch Projection
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Linear: patch_dim -> d_model. Start of VAEEncoder, right after patchify."""
    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        self.embed = nn.Linear(in_features=patch_dim,out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class PatchProj(nn.Module):
    """
    Linear: d_model -> patch_dim. Inverse of PatchEmbed -- param order swapped
    (d_model first, patch_dim second) so the signature signals direction.
    End of VAEDecoder, right before unpatchify.
    """
    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_features=d_model,
                              out_features=patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    


# ---------------------------------------------------------------------------
# 3. Positional Encoding -- TWO variants (config-switched, not stacked)
# ---------------------------------------------------------------------------

def build_2d_sincos_pos_embed(d_model: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    (grid_h*grid_w, d_model). Added right after PatchEmbed, before first block,
    only when attention_type == 'mha'. Skipped entirely for 'rope'.
    """
    raise NotImplementedError

def build_rope_cache(dim: int, seq_len: int, base: float = 10000.0):
    """1D RoPE primitive, called per-axis inside build_rope_cache_2d.
    Returns: (cos, sin), each (seq_len, dim//2)."""
    assert dim % 2 == 0, "the dimension must be even."
    i = torch.arange(0, dim, 2, dtype=torch.float32)          
    inv_freq = 1.0 / (base ** (i / dim))                       
    positions = torch.arange(seq_len, dtype=torch.float32)     
    theta = torch.outer(positions, inv_freq)                  
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0, f"last dim must be even, got {x.shape[-1]}"
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
    

def  apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies 1D rope to one axis's slice of head_dim.
    x: (B, n_heads, N, head_dim). cos/sin: (seq_len, head_dim//2) from build_rope_cache,
    where seq_len == N and head_dim//2 == x's last-dim // 2."""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}D"
    B, n_heads, N, head_dim = x.shape
    assert cos.shape[-1] == head_dim // 2, f"Expected cos last dim {head_dim // 2}, got {cos.shape}"
    assert sin.shape[-1] == head_dim // 2, f"Expected sin last dim {head_dim // 2}, got {sin.shape}"

    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, head_dim)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)

    rot = rotate_half(x)
    return x * cos + rot * sin
    

def build_2d_positions(grid_h: int, grid_w: int) -> torch.Tensor:
    n = torch.arange(grid_h * grid_w)
    rows = n // grid_w
    cols = n % grid_w
    return torch.stack([rows, cols], dim=-1)

def build_rope_cache_2d(dim: int, grid_h: int, grid_w: int, base: float = 10000.0):
    """
    Axial 2D RoPE: build_rope_cache() x2 (per axis) + build_2d_positions().
    Called once in VAEEncoder.__init__ and once in VAEDecoder.__init__ --
    each owns its own cache (grids may differ).
    """
    assert dim%4 ==0,"dim must be divided by 4"
    half_dim = dim//2
    positions = build_2d_positions(grid_h=grid_h,grid_w=grid_w)

    cos_h, sin_h = build_rope_cache(seq_len=grid_h,dim=half_dim,base=base)
    cos_w, sin_w = build_rope_cache(seq_len=grid_w,dim=half_dim,base=base)
    pos_h = positions[:,0]
    pos_w = positions[:,1]
    cos_h = cos_h[pos_h]
    sin_h = sin_h[pos_h]
    cos_w = cos_w[pos_w]
    sin_w = sin_w[pos_w]
    cos = torch.cat(
        [cos_h,
         cos_w],
         dim=-1
    )
    sin = torch.cat(
            [sin_h,
             sin_w],
             dim=-1
        )
   

    return cos,sin

def apply_rope_2d(
    x,
    cos,
    sin
):
    """Rotates first half of head_dim with row rope, second half with col rope."""

    dim_half = x.shape[-1] // 2

    x_h = x[..., :dim_half]
    x_w = x[..., dim_half:]


    cos_h = cos[..., :dim_half//2]
    sin_h = sin[..., :dim_half//2]

    cos_w = cos[..., dim_half//2:]
    sin_w = sin[..., dim_half//2:]


    x_h = apply_rope(
        x_h,
        cos_h,
        sin_h
    )

    x_w = apply_rope(
        x_w,
        cos_w,
        sin_w
    )


    return torch.cat(
        [
            x_h,
            x_w
        ],
        dim=-1
    )
# ---------------------------------------------------------------------------
# 4. Attention Modules
# ---------------------------------------------------------------------------

# --- 4.1 Attention Utilities ---
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, N, d_model) -> (B, n_heads, N, head_dim)"""
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}."
    B, N, d_model = x.shape
    assert d_model % n_heads == 0, f"The d_model must divided by n_head: got d_model:{d_model}, n_head:{n_heads}"
    head_dim = d_model // n_heads
    x = x.reshape(B, N, n_heads, head_dim)
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, N, head_dim) -> (B, N, d_model)"""
    assert x.ndim == 4, f"Expected 4D tensor, got {x.ndim}."
    B, n_heads, N, head_dim = x.shape
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, N, head_dim * n_heads).contiguous()
    return x


class QKVProjection(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.qkv = nn.Linear(in_features=d_model, out_features=3 * d_model)

    def forward(self, x: torch.Tensor):
        """x: (B, N, d_model) -> q, k, v each (B, N, d_model)"""
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}."
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v
    
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None,dropout=None) -> torch.Tensor:
    """softmax(qk^T / sqrt(head_dim)) @ v -- mask unused for VAE (non-causal), kept for API parity
    dropout is applied on attention probabilities after softmax.."""
    logits = q @ k.transpose(3, 2)
    _, _, _, d_head = q.shape
    scale = d_head ** (-0.5)
    logits = logits * scale
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -float("inf"))
    attn = torch.softmax(logits, dim=-1)
    if dropout:
        attn = dropout(attn)
    out = attn @ v
    return out


class MultiHeadAttention(nn.Module):
    """Vanilla MHA -- sin-cos model variant. Uses QKVProjection, split_heads,
    scaled_dot_product_attention, merge_heads."""
    def __init__(self, d_model: int, n_heads: int,dropout:float =0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout  =nn.Dropout(dropout)
        self.qkv_proj = QKVProjection(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(x)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        out = scaled_dot_product_attention(q, k, v,dropout=self.attn_dropout)
        out = merge_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(nn.Module):
    """MHA + RoPE on q,k before the dot-product -- PRIORITY variant. Same
    utilities as MultiHeadAttention, plus apply_rope_2d() after split_heads."""
    def __init__(self, d_model: int, n_heads: int,grid_h:int,grid_w:int,dropout:float=0.0,bias = True):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.attn_dropout = nn.Dropout(dropout)
        self.qkv_proj = QKVProjection(d_model=d_model)
        self.out_proj = nn.Linear(in_features=d_model,out_features=d_model,bias=bias)
        assert d_model%n_heads == 0,f"d_model must be divided by n_head"
        head_dim = d_model//n_heads
        cos_cache, sin_cache = build_rope_cache_2d(
            dim=head_dim,
            grid_h=grid_h,
            grid_w=grid_w
        )


        self.register_buffer(
            "cos_cache",
            cos_cache,
            persistent=False
        )

        self.register_buffer(
            "sin_cache",
            sin_cache,
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        q, k, v = self.qkv_proj(x)
        q = split_heads(q, self.n_heads)
        k = split_heads(k, self.n_heads)
        v = split_heads(v, self.n_heads)
        q_rotated = apply_rope_2d(x=q,cos=self.cos_cache,sin=self.sin_cache)
        k_rotated = apply_rope_2d(x=k,cos=self.cos_cache,sin=self.sin_cache)
        out = scaled_dot_product_attention(q_rotated, k_rotated, v,dropout=self.attn_dropout)
        out = merge_heads(out)
        out = self.out_proj(out)
        return out

        

SUPPORTED_ATTENTIONS = {
    "mha": MultiHeadAttention,
    "rope": RoPEAttention,
    # "gqa": GroupedHeadAttention  -- DEFERRED
}

SUPPORTED_POSITION_ENCODINGS = {
    "sincos": build_2d_sincos_pos_embed,
    "rope": build_rope_cache_2d,
}
# GroupedHeadAttention (GQA): DEFERRED. Orthogonal to position-encoding choice;
# adds a third variable to an already two-model comparison. Revisit post-baseline.


# ---------------------------------------------------------------------------
# 5. Transformer Components
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    NOTE: in_dim must equal out_dim when used inside TransformerBlock (residual
    add x + mlp(x)). Split signature kept for reuse outside the block only.
    """
    def __init__(self, in_dim: int, out_dim: int, mlp_ratio: float = 4.0,dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim,out_features=out_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class TransformerBlock(nn.Module):
    """attention_type: 'mha' | 'rope', selected via SUPPORTED_ATTENTIONS."""
    def __init__(self, d_model: int, n_heads: int, attention_type: str, mlp_ratio: float = 4.0):
        super().__init__()
        # attn = SUPPORTED_ATTENTIONS[attention_type](d_model, n_heads)
        # mlp = MLP(d_model, d_model, mlp_ratio)  -- in_dim==out_dim, required for residual
        raise NotImplementedError

    def forward(self, x: torch.Tensor, rope_cache_2d=None) -> torch.Tensor:
        """'mha': self.attn(x). 'rope': self.attn(x, rope_cache_2d) -- must not be None."""
        raise NotImplementedError


class TransformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_heads: int, depth: int, attention_type: str):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """kwargs forwards rope_cache_2d to each TransformerBlock when attention_type=='rope'."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. VAE
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    patchify -> PatchEmbed -> (+sincos pos_embed if 'mha') -> TransformerBackbone
    -> mu_logvar_head -> split into mu, logvar.
    Owns rope_cache_2d: built once in __init__, registered as buffer.
    """
    def __init__(self, config):
        super().__init__()
        # self.patch_embed = PatchEmbed(patch_dim, d_model)
        # if 'mha': register_buffer sincos pos_embed
        # if 'rope': self.register_buffer('rope_cache_2d', build_rope_cache_2d(...))
        # self.backbone = TransformerBackbone(d_model, n_heads, depth, attention_type)
        # self.mu_logvar_head = nn.Linear(d_model, 2 * latent_dim)
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W) -> mu, logvar, each (B, N, latent_dim)
        patchify -> patch_embed -> (+pos_embed if mha) -> backbone(..., rope_cache_2d=self.rope_cache_2d)
        -> mu, logvar = mu_logvar_head(x).chunk(2, dim=-1)
        """
        raise NotImplementedError


class VAEDecoder(nn.Module):
    """
    z -> latent_proj -> (+sincos pos_embed if 'mha') -> TransformerBackbone
    -> PatchProjection -> unpatchify.
    Owns its own rope_cache_2d (separate buffer from VAEEncoder's).
    """
    def __init__(self, config):
        super().__init__()
        # self.latent_proj = nn.Linear(latent_dim, d_model)
        # if 'mha': register_buffer sincos pos_embed
        # if 'rope': self.register_buffer('rope_cache_2d', build_rope_cache_2d(...))
        # self.backbone = TransformerBackbone(d_model, n_heads, depth, attention_type)
        # self.patch_proj = PatchProjection(d_model, patch_dim)
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, latent_dim) -> (B, C, H, W)
        latent_proj -> (+pos_embed if mha) -> backbone(..., rope_cache_2d=self.rope_cache_2d)
        -> patch_proj -> unpatchify
        """
        raise NotImplementedError


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = mu + eps * std, eps ~ N(0,1)"""
    raise NotImplementedError


class VAE(nn.Module):
    """Wraps VAEEncoder + reparameterize + VAEDecoder. attention_type set via config."""
    def __init__(self, config):
        super().__init__()
        # self.encoder = VAEEncoder(config); self.decoder = VAEDecoder(config)
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """Returns: recon, mu, logvar"""
        # mu, logvar = self.encoder(x); z = reparameterize(mu, logvar); recon = self.decoder(z)
        raise NotImplementedError