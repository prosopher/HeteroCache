# code2_infer_B2A.py
# 목표: 학습된 B->A translator 기반 KV 변환 추론 + 디버깅
# - cosine: translated(A) vs original(A)
# - prefix 2개에 대해 생성 비교
# - argparse 금지
# - transformers==4.35.2

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from translator_lib import (
    OneWayKVTranslator_B2A,
    ModelKVSpec,
    pack_past_key_values,
    unpack_past_key_values,
    cosine_sim_flat,
)

CKPT_PATH = "translator_B2A_paperlike_ckpt.pt"
MAX_NEW_TOKENS = 30

PREFIXES = [
    "Seoul is the capital of",
    "Paris is the capital of",
]


@torch.no_grad()
def get_past_excluding_last_token(model, input_ids: torch.Tensor):
    """
    past만으로 generate 시작:
    - past는 prefix[:-1] 캐시
    - 첫 step 입력은 prefix[-1]
    """
    assert input_ids.size(1) >= 2
    out = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
    return out.past_key_values


@torch.no_grad()
def greedy_generate_from_past(model, tokenizer, prefix_ids, past_excl_last, max_new_tokens: int):
    model.eval()
    input_ids = prefix_ids[:, -1:]
    generated = prefix_ids.clone()
    past = past_excl_last

    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        past = out.past_key_values
        input_ids = next_id

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["MODEL_A_NAME"])
    tokenizer.pad_token = tokenizer.eos_token

    modelA = GPT2LMHeadModel.from_pretrained(cfg["MODEL_A_NAME"]).to(device).eval()
    modelB = GPT2LMHeadModel.from_pretrained(cfg["MODEL_B_NAME"]).to(device).eval()
    modelA.config.pad_token_id = tokenizer.eos_token_id
    modelB.config.pad_token_id = tokenizer.eos_token_id

    translator = OneWayKVTranslator_B2A(
        b_layers=cfg["b_layers"],
        b_hidden=cfg["b_hidden"],
        a_layers=cfg["a_layers"],
        a_hidden=cfg["a_hidden"],
        q_dim=cfg["Q_DIM"],
        adapter_num_heads=cfg["ADAPTER_NUM_HEADS"],
        adapter_head_dim=cfg["ADAPTER_HEAD_DIM"],
        translation_dim_factor=cfg["TRANSLATION_DIM_FACTOR"],
        dropout=cfg["DROPOUT"],
    ).to(device).eval()
    translator.load_state_dict(ckpt["translator_state"])

    specA = ModelKVSpec(
        n_layers=modelA.config.n_layer,
        n_heads=modelA.config.n_head,
        head_dim=modelA.config.n_embd // modelA.config.n_head,
        hidden_size=modelA.config.n_embd,
    )

    for prefix in PREFIXES:
        print("\n" + "=" * 80)
        print("PREFIX:", prefix)

        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        if prefix_ids.size(1) < 2:
            print("prefix token length too short; skip")
            continue

        # original past (excluding last token)
        pastA = get_past_excluding_last_token(modelA, prefix_ids)
        pastB = get_past_excluding_last_token(modelB, prefix_ids)

        K_A_true, V_A_true = pack_past_key_values(pastA)
        K_B, V_B = pack_past_key_values(pastB)

        # translate B -> A
        with torch.no_grad():
            K_A_pred, V_A_pred = translator.translate(K_B, V_B)

        # cosine debug: predicted(A) vs true(A)
        cos_k = cosine_sim_flat(K_A_pred, K_A_true)
        cos_v = cosine_sim_flat(V_A_pred, V_A_true)
        print(f"[cosine] translated(A) vs original(A): key={cos_k:.6f}, val={cos_v:.6f}")

        # build past for model A from translated KV
        pastA_from_B = unpack_past_key_values(K_A_pred, V_A_pred, specA)

        # generations
        out1 = greedy_generate_from_past(modelA, tokenizer, prefix_ids, pastA, MAX_NEW_TOKENS)
        out2 = greedy_generate_from_past(modelA, tokenizer, prefix_ids, pastA_from_B, MAX_NEW_TOKENS)
        out3 = greedy_generate_from_past(modelB, tokenizer, prefix_ids, pastB, MAX_NEW_TOKENS)

        print("\n[generate] (1) A_original -> model A")
        print(out1)
        print("\n[generate] (2) B_to_A     -> model A")
        print(out2)
        print("\n[generate] (3) B_original -> model B")
        print(out3)


if __name__ == "__main__":
    main()
