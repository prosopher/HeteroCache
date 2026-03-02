# train.py
# Toy KV-cache alignment training via layerwise cross-attention adapters (reconstruction loss only).

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# KV cache utilities
# -------------------------
# Purpose: Flatten a (layer, head, seq, head_dim) style KV tensor into per-token vectors [B, S, D].
def _flatten_kv_tensor(k: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Convert key/value tensor to [B, S, D] where D = n_heads * head_dim.
    Supports common shapes:
      - [B, nH, S, Hd]
      - [B, nH, Hd, S]
    """
    if k.dim() != 4:
        raise ValueError(f"Expected 4D kv tensor, got shape={tuple(k.shape)}")

    B, nH, a, b = k.shape
    if a == seq_len:  # [B, nH, S, Hd]
        return k.permute(0, 2, 1, 3).contiguous().view(B, seq_len, nH * b)
    if b == seq_len:  # [B, nH, Hd, S]
        return k.permute(0, 3, 1, 2).contiguous().view(B, seq_len, nH * a)

    raise ValueError(f"Cannot infer seq dim for kv tensor shape={tuple(k.shape)} with seq_len={seq_len}")


# Purpose: Lightweight container for model architecture metadata we need for adapters.
@dataclass
class ModelSpec:
    name: str
    n_layer: int
    n_embd: int
    n_head: int


# Purpose: Extract full-layer KV blocks as dense tensors [B, S, L, D] from a frozen HF causal LM.
@torch.no_grad()
def extract_kv_block(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      keys:   [B, S, L, D]
      values: [B, S, L, D]
    where D = n_embd (flattened heads).
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values  # length L: each is (k, v)

    B, S = input_ids.shape
    keys_layers, vals_layers = [], []
    for (k, v) in past:
        k_flat = _flatten_kv_tensor(k, S)  # [B,S,D]
        v_flat = _flatten_kv_tensor(v, S)
        keys_layers.append(k_flat)
        vals_layers.append(v_flat)

    keys = torch.stack(keys_layers, dim=2)   # [B,S,L,D]
    vals = torch.stack(vals_layers, dim=2)
    return keys, vals


# -------------------------
# Cross-attn translator (paper-inspired)
# -------------------------
# Purpose: One transformer-style cross-attention block (cross-attn + FFN) used inside adapters.
class CrossAttnLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # h: [B,S,D'], mem: [B,S,D']
        q = self.ln_q(h)
        kv = self.ln_kv(mem)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = h + self.drop(attn_out)
        ff_out = self.ff(self.ln_ff(h))
        h = h + self.drop(ff_out)
        return h


# Purpose: Stack of layerwise cross-attention steps to fuse multi-layer KV into a single representation.
class LayerwiseCrossAttnStack(nn.Module):
    """
    Layer-wise cross-attention stack:
      seed = first layer
      at step t, query comes from previous output,
      key/value come from (mapped) layer index layer_map[t]
    """
    def __init__(
        self,
        n_steps: int,
        d_model: int,
        n_heads: int,
        layer_map: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(layer_map) == n_steps
        self.layer_map = layer_map
        self.layers = nn.ModuleList([CrossAttnLayer(d_model, n_heads, dropout) for _ in range(n_steps)])

    def forward(self, x_layers: torch.Tensor) -> torch.Tensor:
        # x_layers: [B,S,L,D']
        B, S, L, Dp = x_layers.shape
        h = x_layers[:, :, 0, :]  # seed from first layer (paper)
        outs = []
        for step, (idx, layer) in enumerate(zip(self.layer_map, self.layers)):
            mem = x_layers[:, :, idx, :]
            h = layer(h, mem)
            outs.append(h)
        return torch.cat(outs, dim=-1)  # [B,S,n_steps*D']


# Purpose: Adapter that maps a model-local KV block into a shared latent space Σ (dimension Q).
class ToSharedAdapter(nn.Module):
    """
    local [B,S,L,D] -> shared [B,S,Q]
    - separate (LN+Linear+GELU) for K vs V
    - shared cross-attn stack between K and V (paper)
    """
    def __init__(
        self,
        L: int,
        D_in: int,
        Q: int,
        d_model: int,
        n_heads: int,
        n_steps: int,
        layer_map: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.L, self.D_in, self.Q = L, D_in, Q

        # separate input transforms for K and V
        self.in_k = nn.Sequential(nn.LayerNorm(D_in), nn.Linear(D_in, d_model), nn.GELU())
        self.in_v = nn.Sequential(nn.LayerNorm(D_in), nn.Linear(D_in, d_model), nn.GELU())

        # shared cross-attn stack for K/V
        self.cross = LayerwiseCrossAttnStack(n_steps, d_model, n_heads, layer_map, dropout)

        # separate output transforms for K and V
        self.out_k = nn.Sequential(nn.LayerNorm(n_steps * d_model), nn.Linear(n_steps * d_model, Q), nn.GELU())
        self.out_v = nn.Sequential(nn.LayerNorm(n_steps * d_model), nn.Linear(n_steps * d_model, Q), nn.GELU())

    def forward(self, x_local: torch.Tensor, kind: str) -> torch.Tensor:
        # x_local: [B,S,L,D_in]
        if kind not in ("k", "v"):
            raise ValueError("kind must be 'k' or 'v'")
        B, S, L, D = x_local.shape
        assert L == self.L and D == self.D_in

        x = self.in_k(x_local) if kind == "k" else self.in_v(x_local)  # [B,S,L,D']
        hcat = self.cross(x)  # [B,S,n_steps*D']
        return self.out_k(hcat) if kind == "k" else self.out_v(hcat)  # [B,S,Q]


# Purpose: Adapter that maps shared latent Σ (dimension Q) back into a specific model's KV block.
class FromSharedAdapter(nn.Module):
    """
    shared [B,S,Q] -> local [B,S,L,D_out]
    - requires Q % L == 0 (paper's symmetric reshape)
    - separate input/output transforms for K vs V
    - shared cross-attn stack between K and V
    """
    def __init__(
        self,
        L: int,
        D_out: int,
        Q: int,
        d_model: int,
        n_heads: int,
        n_steps: int,
        layer_map: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        if Q % L != 0:
            raise ValueError(f"Need Q % L == 0 for reshape, got Q={Q}, L={L}")
        self.L, self.D_out, self.Q = L, D_out, Q
        self.q_per_layer = Q // L

        self.in_k = nn.Sequential(nn.LayerNorm(self.q_per_layer), nn.Linear(self.q_per_layer, d_model), nn.GELU())
        self.in_v = nn.Sequential(nn.LayerNorm(self.q_per_layer), nn.Linear(self.q_per_layer, d_model), nn.GELU())

        self.cross = LayerwiseCrossAttnStack(n_steps, d_model, n_heads, layer_map, dropout)

        self.out_k = nn.Sequential(nn.LayerNorm(n_steps * d_model), nn.Linear(n_steps * d_model, L * D_out), nn.GELU())
        self.out_v = nn.Sequential(nn.LayerNorm(n_steps * d_model), nn.Linear(n_steps * d_model, L * D_out), nn.GELU())

    def forward(self, x_shared: torch.Tensor, kind: str) -> torch.Tensor:
        # x_shared: [B,S,Q]
        if kind not in ("k", "v"):
            raise ValueError("kind must be 'k' or 'v'")
        B, S, Q = x_shared.shape
        assert Q == self.Q

        # reshape [B,S,Q] -> [B,S,L,Q//L] (paper)
        x = x_shared.view(B, S, self.L, self.q_per_layer).contiguous()

        x = self.in_k(x) if kind == "k" else self.in_v(x)  # [B,S,L,D']
        hcat = self.cross(x)  # [B,S,n_steps*D']
        out = self.out_k(hcat) if kind == "k" else self.out_v(hcat)  # [B,S,L*D_out]
        return out.view(B, S, self.L, self.D_out).contiguous()


# Purpose: Convenience container holding the four adapters needed for bidirectional A<->Σ and B<->Σ.
@dataclass
class BiAdapters:
    # model A adapters
    A_to_shared: ToSharedAdapter
    A_from_shared: FromSharedAdapter
    # model B adapters
    B_to_shared: ToSharedAdapter
    B_from_shared: FromSharedAdapter


# Purpose: Build a simple "sampled layer indices" mapping for layerwise cross-attention steps.
def evenly_spaced_layer_map(L: int, n_steps: int) -> List[int]:
    if n_steps <= 1:
        return [0]
    idxs = []
    for t in range(n_steps):
        # map t -> round(t*(L-1)/(n_steps-1))
        idx = int(round(t * (L - 1) / (n_steps - 1)))
        idxs.append(max(0, min(L - 1, idx)))
    return idxs


# -------------------------
# Dataset: token chunks
# -------------------------
# Purpose: Turn a long token stream into fixed-length training chunks.
class TokenChunkDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        # drop tail
        n = (len(token_ids) // seq_len) * seq_len
        self.tokens = token_ids[:n]

    def __len__(self) -> int:
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len]
        return torch.tensor(chunk, dtype=torch.long)


# Purpose: Collate fixed-length batches and build an attention mask.
def collate_pad(batch: List[torch.Tensor], pad_id: int) -> Dict[str, torch.Tensor]:
    # all fixed length already; still return mask
    x = torch.stack(batch, dim=0)
    attn = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": attn}


# -------------------------
# Train
# -------------------------
# Purpose: End-to-end training loop that learns bidirectional KV translators using only reconstruction MSE.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", type=str, default="openai-community/gpt2")
    p.add_argument("--model_b", type=str, default="openai-community/gpt2-medium")
    p.add_argument("--shared_dim_q", type=int, default=1536)  # must be divisible by both L_a and L_b
    p.add_argument("--adapter_dim", type=int, default=256)
    p.add_argument("--adapter_heads", type=int, default=8)
    p.add_argument("--adapter_steps_a", type=int, default=12)   # how many cross-attn steps (<= L_a)
    p.add_argument("--adapter_steps_b", type=int, default=24)   # how many cross-attn steps (<= L_b)
    p.add_argument("--seq_len", type=int, default=64)           # includes last token; cache uses seq_len-1
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_path", type=str, default="kv_align_adapters.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp16" if torch.cuda.is_available() else "fp32")
    args = p.parse_args()

    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Tokenizer (shared)
    tok = AutoTokenizer.from_pretrained(args.model_a, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load models (frozen)
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=torch_dtype).to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=torch_dtype).to(device).eval()
    for p_ in model_a.parameters():
        p_.requires_grad_(False)
    for p_ in model_b.parameters():
        p_.requires_grad_(False)

    spec_a = ModelSpec(args.model_a, model_a.config.n_layer, model_a.config.n_embd, model_a.config.n_head)
    spec_b = ModelSpec(args.model_b, model_b.config.n_layer, model_b.config.n_embd, model_b.config.n_head)

    Q = args.shared_dim_q
    if Q % spec_a.n_layer != 0 or Q % spec_b.n_layer != 0:
        raise ValueError(f"Need Q divisible by both L_a and L_b. Q={Q}, L_a={spec_a.n_layer}, L_b={spec_b.n_layer}")

    # Build adapters
    steps_a = min(args.adapter_steps_a, spec_a.n_layer)
    steps_b = min(args.adapter_steps_b, spec_b.n_layer)
    map_a = evenly_spaced_layer_map(spec_a.n_layer, steps_a)
    map_b = evenly_spaced_layer_map(spec_b.n_layer, steps_b)

    A_to = ToSharedAdapter(spec_a.n_layer, spec_a.n_embd, Q, args.adapter_dim, args.adapter_heads, steps_a, map_a).to(device)
    A_fr = FromSharedAdapter(spec_a.n_layer, spec_a.n_embd, Q, args.adapter_dim, args.adapter_heads, steps_a, map_a).to(device)

    B_to = ToSharedAdapter(spec_b.n_layer, spec_b.n_embd, Q, args.adapter_dim, args.adapter_heads, steps_b, map_b).to(device)
    B_fr = FromSharedAdapter(spec_b.n_layer, spec_b.n_embd, Q, args.adapter_dim, args.adapter_heads, steps_b, map_b).to(device)

    adapters = BiAdapters(A_to, A_fr, B_to, B_fr)

    # Dataset (toy): wikitext-2 small slice
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"][:2000])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    dataset = TokenChunkDataset(ids, seq_len=args.seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id),
        drop_last=True,
    )

    opt = AdamW(list(A_to.parameters()) + list(A_fr.parameters()) + list(B_to.parameters()) + list(B_fr.parameters()), lr=args.lr)

    def translate(src_keys, src_vals, src_to, tgt_from):
        k_shared = src_to(src_keys, "k")
        v_shared = src_to(src_vals, "v")
        k_tgt = tgt_from(k_shared, "k")
        v_tgt = tgt_from(v_shared, "v")
        return k_tgt, v_tgt

    step = 0
    it = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # Use prefix only for cache extraction (common generation usage)
        prefix_ids = input_ids[:, :-1]
        prefix_attn = attn[:, :-1]

        # target caches (no grad)
        keys_a, vals_a = extract_kv_block(model_a, prefix_ids, prefix_attn)
        keys_b, vals_b = extract_kv_block(model_b, prefix_ids, prefix_attn)

        # A -> B
        pred_k_b, pred_v_b = translate(keys_a.float(), vals_a.float(), adapters.A_to_shared, adapters.B_from_shared)
        # B -> A
        pred_k_a, pred_v_a = translate(keys_b.float(), vals_b.float(), adapters.B_to_shared, adapters.A_from_shared)

        loss = (
            F.mse_loss(pred_k_b, keys_b.float()) +
            F.mse_loss(pred_v_b, vals_b.float()) +
            F.mse_loss(pred_k_a, keys_a.float()) +
            F.mse_loss(pred_v_a, vals_a.float())
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(A_to.parameters()) + list(A_fr.parameters()) + list(B_to.parameters()) + list(B_fr.parameters()),
            1.0,
        )
        opt.step()

        if step % 20 == 0:
            print(f"[step {step:04d}] loss={loss.item():.6f}")

        step += 1

    # Save
    payload = {
        "model_a": spec_a.__dict__,
        "model_b": spec_b.__dict__,
        "shared_dim_q": Q,
        "adapter_dim": args.adapter_dim,
        "adapter_heads": args.adapter_heads,
        "steps_a": steps_a,
        "steps_b": steps_b,
        "layer_map_a": map_a,
        "layer_map_b": map_b,
        "state_dict": {
            "A_to": A_to.state_dict(),
            "A_fr": A_fr.state_dict(),
            "B_to": B_to.state_dict(),
            "B_fr": B_fr.state_dict(),
        },
    }
    torch.save(payload, args.save_path)
    print(f"Saved adapters to {args.save_path}")


if __name__ == "__main__":
    main()
