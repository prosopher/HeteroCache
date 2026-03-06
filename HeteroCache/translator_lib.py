# translator_lib.py
# transformers==4.35.2 전제

from dataclasses import dataclass
from typing import Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

KVKind = Literal["k", "v"]


@dataclass
class ModelKVSpec:
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_size: int


def pack_past_key_values(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HF GPT2 past_key_values:
      tuple(n_layers) of (k, v)
      k,v: [B, n_heads, S, head_dim]

    return:
      K,V: [B, S, L, hidden_size]
    """
    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    for (k, v) in past_key_values:
        k2 = k.permute(0, 2, 1, 3).contiguous().view(k.size(0), k.size(2), -1)
        v2 = v.permute(0, 2, 1, 3).contiguous().view(v.size(0), v.size(2), -1)
        keys.append(k2)
        values.append(v2)

    K = torch.stack(keys, dim=2)   # [B,S,L,D]
    V = torch.stack(values, dim=2) # [B,S,L,D]
    return K, V


def unpack_past_key_values(
    K: torch.Tensor,
    V: torch.Tensor,
    spec: ModelKVSpec
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    K,V: [B, S, L, hidden_size]
    return HF past_key_values:
      k,v: [B, n_heads, S, head_dim]
    """
    B, S, L, D = K.shape
    assert L == spec.n_layers, (L, spec.n_layers)
    assert D == spec.hidden_size, (D, spec.hidden_size)

    out = []
    for layer in range(L):
        k = (
            K[:, :, layer, :]
            .contiguous()
            .view(B, S, spec.n_heads, spec.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            V[:, :, layer, :]
            .contiguous()
            .view(B, S, spec.n_heads, spec.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        out.append((k, v))
    return tuple(out)


def cosine_sim_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a1 = a.detach().float().reshape(-1)
    b1 = b.detach().float().reshape(-1)
    denom = (a1.norm() * b1.norm()).clamp_min(eps)
    return float(torch.dot(a1, b1) / denom)


class CrossAttnBlock(nn.Module):
    """
    논문 쪽 설정에 더 가깝게:
    - attention heads/head_dim는 adapter 전체에서 고정
    - FFN 확장 비율은 translation_dim_factor=1 기본
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        translation_dim_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(d_model * translation_dim_factor)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(mem)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln_ff(x)))
        return x


class LocalToSharedAdapterKV(nn.Module):
    """
    T[m_i -> Σ]
    - K/V는 input/output projection만 분리
    - cross-attention stack은 공유
    """
    def __init__(
        self,
        n_layers: int,
        d_in: int,
        q_dim: int,
        adapter_dim: int,
        adapter_n_heads: int,
        translation_dim_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.q_dim = q_dim
        self.adapter_dim = adapter_dim

        self.in_ln = nn.LayerNorm(d_in)
        self.in_proj_k = nn.Linear(d_in, adapter_dim)
        self.in_proj_v = nn.Linear(d_in, adapter_dim)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                d_model=adapter_dim,
                n_heads=adapter_n_heads,
                translation_dim_factor=translation_dim_factor,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.out_ln = nn.LayerNorm(n_layers * adapter_dim)
        self.out_proj_k = nn.Linear(n_layers * adapter_dim, q_dim)
        self.out_proj_v = nn.Linear(n_layers * adapter_dim, q_dim)

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

        cur = h[:, :, 0, :]  # first layer cache as seed
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
    T[Σ -> m_i]
    """
    def __init__(
        self,
        n_layers: int,
        q_dim: int,
        d_out: int,
        adapter_dim: int,
        adapter_n_heads: int,
        translation_dim_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert q_dim % n_layers == 0, f"q_dim({q_dim}) must be divisible by n_layers({n_layers})"

        self.n_layers = n_layers
        self.q_dim = q_dim
        self.d_out = d_out
        self.adapter_dim = adapter_dim
        self.q_per_layer = q_dim // n_layers

        self.in_ln = nn.LayerNorm(self.q_per_layer)
        self.in_proj_k = nn.Linear(self.q_per_layer, adapter_dim)
        self.in_proj_v = nn.Linear(self.q_per_layer, adapter_dim)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                d_model=adapter_dim,
                n_heads=adapter_n_heads,
                translation_dim_factor=translation_dim_factor,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.out_ln = nn.LayerNorm(n_layers * adapter_dim)
        self.out_proj_k = nn.Linear(n_layers * adapter_dim, n_layers * d_out)
        self.out_proj_v = nn.Linear(n_layers * adapter_dim, n_layers * d_out)

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
        return y


class OneWayKVTranslator_B2A(nn.Module):
    """
    B (gpt2-medium) -> Σ -> A (gpt2)
    """
    def __init__(
        self,
        b_layers: int,
        b_hidden: int,
        a_layers: int,
        a_hidden: int,
        q_dim: int = 6144,
        adapter_num_heads: int = 32,
        adapter_head_dim: int = 64,
        translation_dim_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        adapter_dim = adapter_num_heads * adapter_head_dim

        self.q_dim = q_dim
        self.adapter_num_heads = adapter_num_heads
        self.adapter_head_dim = adapter_head_dim
        self.adapter_dim = adapter_dim
        self.translation_dim_factor = translation_dim_factor

        self.B_to_S = LocalToSharedAdapterKV(
            n_layers=b_layers,
            d_in=b_hidden,
            q_dim=q_dim,
            adapter_dim=adapter_dim,
            adapter_n_heads=adapter_num_heads,
            translation_dim_factor=translation_dim_factor,
            dropout=dropout,
        )
        self.S_to_A = SharedToLocalAdapterKV(
            n_layers=a_layers,
            q_dim=q_dim,
            d_out=a_hidden,
            adapter_dim=adapter_dim,
            adapter_n_heads=adapter_num_heads,
            translation_dim_factor=translation_dim_factor,
            dropout=dropout,
        )

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
