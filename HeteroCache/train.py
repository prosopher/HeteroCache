"""
Code 1
- Goal: train a top-layer-only toy KV translator from GPT-2 Medium (B) to GPT-2 (A)
- Data: OpenWebText
- Objective: suffix LM loss only (B->A only), no reconstruction loss

Key change from the previous version
- We do not replace the entire target KV cache.
- We only translate the top N layers, in a direct layer-to-layer 1:1 fashion.
- During training, model A first builds its own prefix KV cache.
- Then only model A's top N layers are replaced by translated model B layers.
"""

from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import (
    TrainConfig,
    WarmupCosineScheduler,
    build_models_and_tokenizer,
    build_training_dataloader,
    build_translator_pool,
    compute_suffix_lm_loss,
    count_trainable_parameters,
    extract_past_key_values,
    replace_top_layers,
    save_checkpoint,
    set_seed,
    split_prefix_and_suffix_for_exact_next_token_loss,
    write_json,
)


CONFIG = TrainConfig(
    model_a_id="gpt2",
    model_b_id="gpt2-medium",
    output_dir="./outputs/lsc_toy_topn",
    max_steps=50000,
    batch_size=1,
    grad_accum_steps=16,
    total_tokens=128,
    prefix_tokens=64,
    learning_rate=1e-4,
    weight_decay=1e-2,
    warmup_steps=1000,
    grad_clip_norm=1.0,
    log_every=25,
    seed=42,
    shuffle_buffer=50_000,
    top_layers_to_translate=6,
    translator_dim=1024,
    translator_heads=16,
    translator_depth=2,
    translator_mlp_ratio=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float32",
)


def main() -> None:
    set_seed(CONFIG.seed)
    output_dir = Path(CONFIG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.json", CONFIG.__dict__)

    print(f"[Setup] device={CONFIG.device}")
    print(f"[Setup] loading models: A={CONFIG.model_a_id}, B={CONFIG.model_b_id}")
    model_a, model_b, tokenizer = build_models_and_tokenizer(CONFIG)
    translator_pool, model_specs = build_translator_pool(model_a, model_b, CONFIG)
    translator_pool.train()

    print("[Setup] model specs")
    print(f"  A: layers={model_specs['A'].num_layers}, hidden={model_specs['A'].hidden_size}, heads={model_specs['A'].num_heads}")
    print(f"  B: layers={model_specs['B'].num_layers}, hidden={model_specs['B'].hidden_size}, heads={model_specs['B'].num_heads}")
    print(f"[Setup] top_layers_to_translate = {CONFIG.top_layers_to_translate}")
    print(f"[Setup] trainable translator params = {count_trainable_parameters(translator_pool):,}")

    dataloader = build_training_dataloader(tokenizer, CONFIG)

    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=CONFIG.warmup_steps,
        total_steps=CONFIG.max_steps,
    )

    running_loss = 0.0
    progress_bar = tqdm(range(1, CONFIG.max_steps + 1), desc="Training")

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0

        for _ in range(CONFIG.grad_accum_steps):
            input_ids = next(dataloader).to(CONFIG.device)
            prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
                input_ids=input_ids,
                prefix_tokens=CONFIG.prefix_tokens,
            )

            with torch.no_grad():
                past_a = extract_past_key_values(model_a, prefix_cache_ids)
                past_b = extract_past_key_values(model_b, prefix_cache_ids)

            translated_b_to_a_top = translator_pool.translate_top_layers(
                past_key_values=past_b,
                src_name="B",
                dst_name="A",
                dst_spec=model_specs["A"],
            )

            mixed_past_for_a = replace_top_layers(
                base_past_key_values=past_a,
                translated_top_past_key_values=translated_b_to_a_top,
            )

            loss_b_to_a = compute_suffix_lm_loss(
                target_model=model_a,
                translated_past_key_values=mixed_past_for_a,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
            )

            loss = loss_b_to_a / CONFIG.grad_accum_steps
            loss.backward()
            step_loss_value += loss.item()

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), CONFIG.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss_value
        if step % CONFIG.log_every == 0:
            avg_loss = running_loss / CONFIG.log_every
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.lr:.2e}")
            print(f"[Step {step:04d}] b_to_a_suffix_lm_loss={avg_loss:.4f} | lr={scheduler.lr:.2e}")
            running_loss = 0.0

    final_path = output_dir / "final_checkpoint.pt"
    save_checkpoint(
        output_path=str(final_path),
        translator_pool=translator_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=CONFIG,
        step=CONFIG.max_steps,
        extra={
            "note": "Final checkpoint trained with top-layer-only B_to_A suffix LM loss.",
            "model_a": CONFIG.model_a_id,
            "model_b": CONFIG.model_b_id,
            "top_layers_to_translate": CONFIG.top_layers_to_translate,
            "direction": "B_to_A",
        },
    )
    print(f"[Done] final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
