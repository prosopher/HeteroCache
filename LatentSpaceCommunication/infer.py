# code2_kv_translate_infer.py
# 목표: 학습된 Cross Attention Translator 기반 KV 변환 추론 + 디버깅
# - Round-trip KV cosine similarity
# - Prefix 2개에 대해 4가지 generate 비교
#
# transformers==4.35.2 전제

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from translator_lib import (
    SharedSpaceKVTranslator,
    ModelKVSpec,
    pack_past_key_values,
    unpack_past_key_values,
    cosine_sim_flat,
)

# -------------------
# 설정 (argparse 금지)
# -------------------
CKPT_PATH = "translator_ckpt.pt"
MAX_NEW_TOKENS = 30

PREFIXES = [
    "Seoul is the capital of",
    "Paris is the capital of",
]


@torch.no_grad()
def get_past_excluding_last_token(model, input_ids: torch.Tensor) :
    """
    past만으로 generate를 시작하려면,
    past가 prefix[:-1]에 대한 캐시여야 하고,
    첫 step 입력으로 prefix[-1]을 넣어서 next token을 예측하게 만드는 게 깔끔합니다.
    """
    assert input_ids.size(1) >= 2, "prefix는 최소 2토큰 이상이어야 합니다."
    out = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
    return out.past_key_values


@torch.no_grad()
def greedy_generate_from_past(model, tokenizer, prefix_ids, past_excl_last, max_new_tokens: int):
    model.eval()

    # 첫 입력은 prefix의 마지막 토큰
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

    # translator
    translator = SharedSpaceKVTranslator(
        a_layers=cfg["a_layers"], a_hidden=cfg["a_hidden"],
        b_layers=cfg["b_layers"], b_hidden=cfg["b_hidden"],
        q_dim=cfg["Q_DIM"], d_model=cfg["D_MODEL"], n_heads=cfg["N_HEADS"], dropout=cfg["DROPOUT"]
    ).to(device).eval()
    translator.load_state_dict(ckpt["translator_state"])

    # KV spec (unpack용)
    specA = ModelKVSpec(
        n_layers=modelA.config.n_layer,
        n_heads=modelA.config.n_head,
        head_dim=modelA.config.n_embd // modelA.config.n_head,
        hidden_size=modelA.config.n_embd,
    )
    specB = ModelKVSpec(
        n_layers=modelB.config.n_layer,
        n_heads=modelB.config.n_head,
        head_dim=modelB.config.n_embd // modelB.config.n_head,
        hidden_size=modelB.config.n_embd,
    )

    for prefix in PREFIXES:
        print("\n" + "=" * 80)
        print("PREFIX:", prefix)

        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        if prefix_ids.size(1) < 2:
            print("prefix token length too short; skip")
            continue

        # 1) original past (excluding last token)
        pastA = get_past_excluding_last_token(modelA, prefix_ids)
        pastB = get_past_excluding_last_token(modelB, prefix_ids)

        K_A, V_A = pack_past_key_values(pastA)  # [B,S,L_A,D_A]
        K_B, V_B = pack_past_key_values(pastB)  # [B,S,L_B,D_B]

        # 2) translate
        with torch.no_grad():
            K_A_to_B, V_A_to_B = translator.translate_A_to_B(K_A, V_A)  # packed B-shape
            K_B_to_A, V_B_to_A = translator.translate_B_to_A(K_B, V_B)  # packed A-shape

            # 3) round-trip for debugging
            K_A_round, V_A_round = translator.translate_B_to_A(K_A_to_B, V_A_to_B)  # A -> B -> A
            K_B_round, V_B_round = translator.translate_A_to_B(K_B_to_A, V_B_to_A)  # B -> A -> B

        # cosine similarity (round-trip vs original)
        cos_A_k = cosine_sim_flat(K_A, K_A_round)
        cos_A_v = cosine_sim_flat(V_A, V_A_round)
        cos_B_k = cosine_sim_flat(K_B, K_B_round)
        cos_B_v = cosine_sim_flat(V_B, V_B_round)
        print(f"[cosine] A round-trip: key={cos_A_k:.6f}, val={cos_A_v:.6f}")
        print(f"[cosine] B round-trip: key={cos_B_k:.6f}, val={cos_B_v:.6f}")

        # 4) unpack to HF past for generation
        pastA_original = pastA
        pastB_original = pastB
        pastA_from_B = unpack_past_key_values(K_B_to_A, V_B_to_A, specA)   # B_original -> A
        pastB_from_A = unpack_past_key_values(K_A_to_B, V_A_to_B, specB)   # A_original -> B

        # 5) requested generations (4 cases)
        out1 = greedy_generate_from_past(modelA, tokenizer, prefix_ids, pastA_original, MAX_NEW_TOKENS)
        out2 = greedy_generate_from_past(modelB, tokenizer, prefix_ids, pastB_from_A,  MAX_NEW_TOKENS)
        out3 = greedy_generate_from_past(modelA, tokenizer, prefix_ids, pastA_from_B,  MAX_NEW_TOKENS)
        out4 = greedy_generate_from_past(modelB, tokenizer, prefix_ids, pastB_original, MAX_NEW_TOKENS)

        print("\n[generate] (1) A_original  -> model A")
        print(out1)
        print("\n[generate] (2) A_to_B      -> model B")
        print(out2)
        print("\n[generate] (3) B_to_A      -> model A")
        print(out3)
        print("\n[generate] (4) B_original  -> model B")
        print(out4)


if __name__ == "__main__":
    main()
