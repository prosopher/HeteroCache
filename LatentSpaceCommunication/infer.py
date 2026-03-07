"""
Code 2
- Goal: run KV translation inference with the trained cross-attention translator
- Debug: measure cosine similarity between round-trip KV and original KV
- Test prefixes:
    * "Seoul is the capital of"
    * "Paris is the capital of"
- Generation variants:
    * generate(past_kv=A_original, model=A)
    * generate(past_kv=A_to_B,   model=B)
    * generate(past_kv=B_to_A,   model=A)
    * generate(past_kv=B_original, model=B)

Important HF note
- In transformers==4.35.2, raw past_key_values alone are not enough to sample the very first next token.
- So this script stores the KV cache for prefix[:-1] and uses the last prefix token as the seed token.
- Functionally, this still means "continue generation from the prefix KV".
"""

from pathlib import Path

import torch

from common import *

# -----------------------------------------------------------------------------
# No argparse by request. Edit values here.
# -----------------------------------------------------------------------------
CONFIG = InferenceConfig(
    checkpoint_path="./outputs/lsc_toy/final_checkpoint.pt",
    max_new_tokens=32,
    do_sample=False,
    temperature=1.0,
    top_k=50,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    prefixes=(
        "Seoul is the capital of",
        "Paris is the capital of",
    ),
)


def generate_and_print(
    title: str,
    model,
    tokenizer,
    full_prefix_ids: torch.Tensor,
    seed_token: torch.Tensor,
    past_key_values,
) -> None:
    generated_ids = generate_from_past(
        model=model,
        seed_token=seed_token,
        past_key_values=past_key_values,
        max_new_tokens=CONFIG.max_new_tokens,
        do_sample=CONFIG.do_sample,
        temperature=CONFIG.temperature,
        top_k=CONFIG.top_k,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = decode_full_generation(
        tokenizer=tokenizer,
        prefix_ids=full_prefix_ids,
        generated_ids=generated_ids,
    )
    print(f"\n[{title}]\n{text}")


@torch.no_grad()
def main() -> None:
    set_seed(CONFIG.seed)
    checkpoint_path = Path(CONFIG.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run code1_train_cross_attention_translator.py first, or change CONFIG.checkpoint_path."
        )

    train_config, translator_pool, model_specs, model_a, model_b, tokenizer = load_translator_pool_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device_override=CONFIG.device,
    )

    print(f"[Setup] loaded checkpoint from {checkpoint_path}")
    print(f"[Setup] A={train_config.model_a_id}, B={train_config.model_b_id}, device={CONFIG.device}")

    for prefix_text in CONFIG.prefixes:
        print("\n" + "=" * 100)
        print(f"Prefix: {prefix_text}")
        print("=" * 100)

        prepared_a = prepare_generation_prefix(
            model=model_a,
            tokenizer=tokenizer,
            text=prefix_text,
            device=CONFIG.device,
        )
        prepared_b = prepare_generation_prefix(
            model=model_b,
            tokenizer=tokenizer,
            text=prefix_text,
            device=CONFIG.device,
        )

        past_a_original = prepared_a["past_key_values"]
        past_b_original = prepared_b["past_key_values"]

        past_a_to_b = translator_pool.translate_past_key_values(
            past_key_values=past_a_original,
            src_name="A",
            dst_name="B",
            dst_spec=model_specs["B"],
        )
        past_b_to_a = translator_pool.translate_past_key_values(
            past_key_values=past_b_original,
            src_name="B",
            dst_name="A",
            dst_spec=model_specs["A"],
        )

        # Cross-model round-trip for debugging
        past_a_round_trip = translator_pool.translate_past_key_values(
            past_key_values=past_a_to_b,
            src_name="B",
            dst_name="A",
            dst_spec=model_specs["A"],
        )
        past_b_round_trip = translator_pool.translate_past_key_values(
            past_key_values=past_b_to_a,
            src_name="A",
            dst_name="B",
            dst_spec=model_specs["B"],
        )

        cos_a = cosine_similarity_between_past(past_a_original, past_a_round_trip)
        cos_b = cosine_similarity_between_past(past_b_original, past_b_round_trip)
        print(f"[Round-Trip Cosine] A original vs A->B->A = {cos_a:.6f}")
        print(f"[Round-Trip Cosine] B original vs B->A->B = {cos_b:.6f}")

        generate_and_print(
            title="generate(past_kv=A_original, model=A)",
            model=model_a,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_a["full_prefix_ids"],
            seed_token=prepared_a["seed_token"],
            past_key_values=past_a_original,
        )
        generate_and_print(
            title="generate(past_kv=A_to_B, model=B)",
            model=model_b,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_a["full_prefix_ids"],
            seed_token=prepared_a["seed_token"],
            past_key_values=past_a_to_b,
        )
        generate_and_print(
            title="generate(past_kv=B_to_A, model=A)",
            model=model_a,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_b["full_prefix_ids"],
            seed_token=prepared_b["seed_token"],
            past_key_values=past_b_to_a,
        )
        generate_and_print(
            title="generate(past_kv=B_original, model=B)",
            model=model_b,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_b["full_prefix_ids"],
            seed_token=prepared_b["seed_token"],
            past_key_values=past_b_original,
        )


if __name__ == "__main__":
    main()
