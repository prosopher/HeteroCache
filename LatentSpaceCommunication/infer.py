# infer.py
# KV translation inference using adapters trained by train.py (imports shared code from train.py).

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import common utilities/adapters from train.py
from train import extract_kv_block, ToSharedAdapter, FromSharedAdapter


def block_to_past(
    keys_block: torch.Tensor,  # [B,S,L,D]
    vals_block: torch.Tensor,  # [B,S,L,D]
    n_head: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    B, S, L, D = keys_block.shape
    head_dim = D // n_head
    past = []
    for l in range(L):
        k = keys_block[:, :, l, :].contiguous().view(B, S, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
        v = vals_block[:, :, l, :].contiguous().view(B, S, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
        past.append((k, v))
    return tuple(past)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="kv_align_adapters.pt")
    ap.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp16" if torch.cuda.is_available() else "fp32")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    payload = torch.load(args.ckpt, map_location="cpu")
    spec_a = payload["model_a"]
    spec_b = payload["model_b"]
    Q = payload["shared_dim_q"]

    # Tokenizer shared (use model_a)
    tok = AutoTokenizer.from_pretrained(spec_a["name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load frozen models
    model_a = AutoModelForCausalLM.from_pretrained(spec_a["name"], torch_dtype=torch_dtype).to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained(spec_b["name"], torch_dtype=torch_dtype).to(device).eval()
    for p in model_a.parameters():
        p.requires_grad_(False)
    for p in model_b.parameters():
        p.requires_grad_(False)

    # Rebuild adapters (must match train.py)
    A_to = ToSharedAdapter(
        L=spec_a["n_layer"], D_in=spec_a["n_embd"], Q=Q,
        d_model=payload["adapter_dim"], n_heads=payload["adapter_heads"],
        n_steps=payload["steps_a"], layer_map=payload["layer_map_a"],
    ).to(device)
    A_fr = FromSharedAdapter(
        L=spec_a["n_layer"], D_out=spec_a["n_embd"], Q=Q,
        d_model=payload["adapter_dim"], n_heads=payload["adapter_heads"],
        n_steps=payload["steps_a"], layer_map=payload["layer_map_a"],
    ).to(device)

    B_to = ToSharedAdapter(
        L=spec_b["n_layer"], D_in=spec_b["n_embd"], Q=Q,
        d_model=payload["adapter_dim"], n_heads=payload["adapter_heads"],
        n_steps=payload["steps_b"], layer_map=payload["layer_map_b"],
    ).to(device)
    B_fr = FromSharedAdapter(
        L=spec_b["n_layer"], D_out=spec_b["n_embd"], Q=Q,
        d_model=payload["adapter_dim"], n_heads=payload["adapter_heads"],
        n_steps=payload["steps_b"], layer_map=payload["layer_map_b"],
    ).to(device)

    A_to.load_state_dict(payload["state_dict"]["A_to"])
    A_fr.load_state_dict(payload["state_dict"]["A_fr"])
    B_to.load_state_dict(payload["state_dict"]["B_to"])
    B_fr.load_state_dict(payload["state_dict"]["B_fr"])
    A_to.eval(); A_fr.eval(); B_to.eval(); B_fr.eval()

    # Prepare input
    ids = tok(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_len)["input_ids"].to(device)
    if ids.shape[1] < 3:
        raise ValueError("Prompt too short; need at least 3 tokens.")
    prefix_ids = ids[:, :-1]
    last_id = ids[:, -1:]

    attn_prefix = torch.ones_like(prefix_ids)

    # ---- A -> B KV translation ----
    keys_a, vals_a = extract_kv_block(model_a, prefix_ids, attn_prefix)
    with torch.no_grad():
        k_shared = A_to(keys_a.float(), "k")
        v_shared = A_to(vals_a.float(), "v")
        pred_k_b = B_fr(k_shared, "k")
        pred_v_b = B_fr(v_shared, "v")

    past_b_pred = block_to_past(pred_k_b.to(device, dtype=torch_dtype), pred_v_b.to(device, dtype=torch_dtype), model_b.config.n_head)

    # Reference B cache (ground truth) on same prefix
    keys_b, vals_b = extract_kv_block(model_b, prefix_ids, attn_prefix)
    past_b_true = block_to_past(keys_b.to(device, dtype=torch_dtype), vals_b.to(device, dtype=torch_dtype), model_b.config.n_head)

    # Compare next-token logits using (past, last_token)
    with torch.no_grad():
        logits_pred = model_b(input_ids=last_id, past_key_values=past_b_pred, use_cache=True).logits[:, -1, :]
        logits_true = model_b(input_ids=last_id, past_key_values=past_b_true, use_cache=True).logits[:, -1, :]

    cos = F.cosine_similarity(logits_pred, logits_true, dim=-1).item()
    print(f"[A->B] cosine(logits_pred, logits_true) = {cos:.4f}")

    topk = 10
    pred_top = torch.topk(logits_pred, k=topk, dim=-1).indices[0].tolist()
    true_top = torch.topk(logits_true, k=topk, dim=-1).indices[0].tolist()
    print("[A->B] top-10 tokens (pred):", [tok.decode([t]) for t in pred_top])
    print("[A->B] top-10 tokens (true):", [tok.decode([t]) for t in true_top])

    # ---- B -> A KV translation (bidirectional) ----
    with torch.no_grad():
        k_shared2 = B_to(keys_b.float(), "k")
        v_shared2 = B_to(vals_b.float(), "v")
        pred_k_a = A_fr(k_shared2, "k")
        pred_v_a = A_fr(v_shared2, "v")

    past_a_pred = block_to_past(pred_k_a.to(device, dtype=torch_dtype), pred_v_a.to(device, dtype=torch_dtype), model_a.config.n_head)
    keys_a_true, vals_a_true = extract_kv_block(model_a, prefix_ids, attn_prefix)
    past_a_true = block_to_past(keys_a_true.to(device, dtype=torch_dtype), vals_a_true.to(device, dtype=torch_dtype), model_a.config.n_head)

    with torch.no_grad():
        logits_pred2 = model_a(input_ids=last_id, past_key_values=past_a_pred, use_cache=True).logits[:, -1, :]
        logits_true2 = model_a(input_ids=last_id, past_key_values=past_a_true, use_cache=True).logits[:, -1, :]

    cos2 = F.cosine_similarity(logits_pred2, logits_true2, dim=-1).item()
    print(f"[B->A] cosine(logits_pred, logits_true) = {cos2:.4f}")


if __name__ == "__main__":
    main()
