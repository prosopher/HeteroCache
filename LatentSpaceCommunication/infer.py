"""
Code 2
- Goal: run top-layer KV translation inference with the trained translator
- Debug: measure cosine similarity between round-trip translated top-layer KV and original top-layer KV
- Accuracy-oriented evaluation: build the target model's own prefix KV by forward pass, then replace only its top N layers with translated KV
"""

from pathlib import Path

import torch

from common import (
    InferenceConfig,
    cosine_similarity_between_past,
    decode_full_generation,
    generate_from_past,
    load_translator_pool_from_checkpoint,
    prepare_generation_prefix,
    replace_top_layers,
    set_seed,
    slice_top_layers,
)


CONFIG = InferenceConfig(
    checkpoint_path="./outputs/lsc_toy_topn/final_checkpoint.pt",
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

    top_n = train_config.top_layers_to_translate

    print(f"[Setup] loaded checkpoint from {checkpoint_path}")
    print(f"[Setup] A={train_config.model_a_id}, B={train_config.model_b_id}, device={CONFIG.device}")
    print(f"[Setup] top_layers_to_translate={top_n}")

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

        past_a_to_b_top = translator_pool.translate_top_layers(
            past_key_values=past_a_original,
            src_name="A",
            dst_name="B",
            dst_spec=model_specs["B"],
        )
        past_b_to_a_top = translator_pool.translate_top_layers(
            past_key_values=past_b_original,
            src_name="B",
            dst_name="A",
            dst_spec=model_specs["A"],
        )

        past_b_mixed = replace_top_layers(
            base_past_key_values=past_b_original,
            translated_top_past_key_values=past_a_to_b_top,
        )
        past_a_mixed = replace_top_layers(
            base_past_key_values=past_a_original,
            translated_top_past_key_values=past_b_to_a_top,
        )

        past_a_round_trip_top = translator_pool.translate_top_layers(
            past_key_values=past_b_mixed,
            src_name="B",
            dst_name="A",
            dst_spec=model_specs["A"],
        )
        past_b_round_trip_top = translator_pool.translate_top_layers(
            past_key_values=past_a_mixed,
            src_name="A",
            dst_name="B",
            dst_spec=model_specs["B"],
        )

        cos_a = cosine_similarity_between_past(slice_top_layers(past_a_original, top_n), past_a_round_trip_top)
        cos_b = cosine_similarity_between_past(slice_top_layers(past_b_original, top_n), past_b_round_trip_top)
        print(f"[Top-Layer Round-Trip Cosine] A top-{top_n} original vs A->B->A = {cos_a:.6f}")
        print(f"[Top-Layer Round-Trip Cosine] B top-{top_n} original vs B->A->B = {cos_b:.6f}")

        generate_and_print(
            title="generate(past_kv=A_original, model=A)",
            model=model_a,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_a["full_prefix_ids"],
            seed_token=prepared_a["seed_token"],
            past_key_values=past_a_original,
        )
        generate_and_print(
            title=f"generate(past_kv=B_original with translated A top-{top_n}, model=B)",
            model=model_b,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_b["full_prefix_ids"],
            seed_token=prepared_b["seed_token"],
            past_key_values=past_b_mixed,
        )
        generate_and_print(
            title=f"generate(past_kv=A_original with translated B top-{top_n}, model=A)",
            model=model_a,
            tokenizer=tokenizer,
            full_prefix_ids=prepared_a["full_prefix_ids"],
            seed_token=prepared_a["seed_token"],
            past_key_values=past_a_mixed,
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
