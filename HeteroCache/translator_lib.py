# translator_lib.py
# transformers==4.35.2 전제 (DynamicCache 사용 안 함)

from dataclasses import dataclass
from typing import Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

KVKind = Literal["k", "v"]


# ---------------------------
# KV pack/unpack helpers
# ---------------------------

@dataclass
class ModelKVSpec:
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_size: int  # n_heads * head_dim


def pack_past_key_values(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HF GPT2 past_key_values:
      tuple(n_layers) of (k, v)
      k,v: [B, n_heads, S, head_dim]
    return:
      K,V: [B, S, L, hidden_size]  where hidden_size=n_heads*head_dim
    """
    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    for (k, v) in past_key_values:
        # [B, nH, S, Hd] -> [B, S, nH, Hd] -> [B, S, nH*Hd]
        k2 = k.permute(0, 2, 1, 3).contiguous().view(k.size(0), k.size(2), -1)
        v2 = v.permute(0, 2, 1, 3).contiguous().view(v.size(0), v.size(2), -1)
        keys.append(k2)
        values.append(v2)

    K = torch.stack(keys, dim=2)   # [B, S, L, D]
    V = torch.stack(values, dim=2) # [B, S, L, D]
    return K, V


def unpack_past_key_values(
    K: torch.Tensor,
    V: torch.Tensor,
    spec: ModelKVSpec
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    K,V: [B, S, L, hidden_size]
    return HF past_key_values tuple:
      k,v: [B, n_heads, S, head_dim]
    """
    B, S, L, D = K.shape
    assert L == spec.n_layers, (L, spec.n_layers)
    assert D == spec.hidden_size, (D, spec.hidden_size)

    out = []
    for layer in range(L):
        k = K[:, :, layer, :].contiguous().view(B, S, spec.n_heads, spec.head_dim).permute(0, 2, 1, 3).contiguous()
        v = V[:, :, layer, :].contiguous().view(B, S, spec.n_heads, spec.head_dim).permute(0, 2, 1, 3).contiguous()
        out.append((k, v))
    return tuple(out)


def cosine_sim_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a1 = a.detach().float().reshape(-1)
    b1 = b.detach().float().reshape(-1)
    denom = (a1.norm() * b1.norm()).clamp_min(eps)
    return float(torch.dot(a1, b1) / denom)


# ---------------------------
# Cross-attention translator blocks (shared)
# ---------------------------

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(mem)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# ---------------------------
# Adapters:
# - K/V share cross-attn blocks
# - ONLY projections differ for K vs V
# ---------------------------

class LocalToSharedAdapterKV(nn.Module):
    """
    T[mi -> Σ] for either K or V:
      input:  [B,S,L,Di]
      output: [B,S,Q]
    """
    def __init__(
        self,
        n_layers: int,
        d_in: int,
        q_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.q_dim = q_dim
        self.d_model = d_model

        self.in_ln = nn.LayerNorm(d_in)
        self.in_proj_k = nn.Linear(d_in, d_model)
        self.in_proj_v = nn.Linear(d_in, d_model)

        self.blocks = nn.ModuleList([CrossAttnBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])

        self.out_ln = nn.LayerNorm(n_layers * d_model)
        self.out_proj_k = nn.Linear(n_layers * d_model, q_dim)
        self.out_proj_v = nn.Linear(n_layers * d_model, q_dim)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, kind: KVKind) -> torch.Tensor:
        B, S, L, Di = x.shape
        assert L == self.n_layers and Di == self.d_in

        x = self.in_ln(x)
        if kind == "k":
            h = self.act(self.in_proj_k(x))
        elif kind == "v":
            h = self.act(self.in_proj_v(x))
        else:
            raise ValueError(kind)

        cur = h[:, :, 0, :]
        outs = []
        for layer_idx, blk in enumerate(self.blocks):
            mem = h[:, :, layer_idx, :]
            cur = blk(cur, mem)
            outs.append(cur)

        cat = self.out_ln(torch.cat(outs, dim=-1))
        if kind == "k":
            y = self.act(self.out_proj_k(cat))
        else:
            y = self.act(self.out_proj_v(cat))
        return y  # [B,S,Q]


class SharedToLocalAdapterKV(nn.Module):
    """
    T[Σ -> mi] for either K or V:
      input:  [B,S,Q]
      output: [B,S,L,Di]
    """
    def __init__(
        self,
        n_layers: int,
        q_dim: int,
        d_out: int,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert q_dim % n_layers == 0, f"q_dim({q_dim}) must be divisible by n_layers({n_layers})"

        self.n_layers = n_layers
        self.q_dim = q_dim
        self.d_out = d_out
        self.d_model = d_model

        self.q_per_layer = q_dim // n_layers

        self.in_ln = nn.LayerNorm(self.q_per_layer)
        self.in_proj_k = nn.Linear(self.q_per_layer, d_model)
        self.in_proj_v = nn.Linear(self.q_per_layer, d_model)

        self.blocks = nn.ModuleList([CrossAttnBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])

        self.out_ln = nn.LayerNorm(n_layers * d_model)
        self.out_proj_k = nn.Linear(n_layers * d_model, n_layers * d_out)
        self.out_proj_v = nn.Linear(n_layers * d_model, n_layers * d_out)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, kind: KVKind) -> torch.Tensor:
        B, S, Q = x.shape
        assert Q == self.q_dim

        x = x.view(B, S, self.n_layers, self.q_per_layer)
        x = self.in_ln(x)

        if kind == "k":
            h = self.act(self.in_proj_k(x))
        elif kind == "v":
            h = self.act(self.in_proj_v(x))
        else:
            raise ValueError(kind)

        cur = h[:, :, 0, :]
        outs = []
        for layer_idx, blk in enumerate(self.blocks):
            mem = h[:, :, layer_idx, :]
            cur = blk(cur, mem)
            outs.append(cur)

        cat = self.out_ln(torch.cat(outs, dim=-1))
        if kind == "k":
            y = self.act(self.out_proj_k(cat))
        else:
            y = self.act(self.out_proj_v(cat))

        y = y.view(B, S, self.n_layers, self.d_out)
        return y  # [B,S,L,d_out]


# ---------------------------
# One-way translator: B (gpt2-medium) -> A (gpt2)
# ---------------------------

class OneWayKVTranslator_B2A(nn.Module):
    """
    B -> Σ -> A
    학습/추론 모두 이 방향만 사용.
    """
    def __init__(
        self,
        b_layers: int, b_hidden: int,
        a_layers: int, a_hidden: int,
        q_dim: int = 1536,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.B_to_S = LocalToSharedAdapterKV(b_layers, b_hidden, q_dim, d_model, n_heads, dropout=dropout)
        self.S_to_A = SharedToLocalAdapterKV(a_layers, q_dim, a_hidden, d_model, n_heads, dropout=dropout)

    @torch.no_grad()
    def translate(self, K_B: torch.Tensor, V_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        K_star = self.B_to_S(K_B, "k")
        V_star = self.B_to_S(V_B, "v")
        K_A = self.S_to_A(K_star, "k")
        V_A = self.S_to_A(V_star, "v")
        return K_A, V_A

    def loss(self, K_B, V_B, K_A_target, V_A_target) -> torch.Tensor:
        K_star = self.B_to_S(K_B, "k")
        V_star = self.B_to_S(V_B, "v")
        K_A = self.S_to_A(K_star, "k")
        V_A = self.S_to_A(V_star, "v")
        return F.mse_loss(K_A, K_A_target) + F.mse_loss(V_A, V_A_target)

    def forward(self, K_B, V_B, K_A_target, V_A_target) -> torch.Tensor:
        return self.loss(K_B, V_B, K_A_target, V_A_target)
