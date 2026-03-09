"""
Code 1
- Goal: train a toy cross-attention translator between GPT-2 and GPT-2 Medium
- Data: OpenWebText
- Objective: suffix LM loss only (bidirectional sum per step), no reconstruction loss

Important
- This example is intentionally compact and pragmatic.
- It follows the paper's high-level recipe with a shared latent space and per-model in/out translators,
  but uses a smaller, easier-to-read implementation suitable for experimentation.
- It is written to be compatible with transformers==4.35.2, where past_key_values are plain tuples,
  so DynamicCache is not used.
"""

import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import (
    TrainConfig,
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
    WarmupCosineScheduler,
    write_json,
)


# -----------------------------------------------------------------------------
# No argparse by request. Edit values here.
# -----------------------------------------------------------------------------
CONFIG = TrainConfig(
    model_a_id="gpt2",
    model_b_id="gpt2-medium",
    output_dir="./outputs/lsc_toy",
    max_steps=10_000,
    batch_size=1,
    grad_accum_steps=16,
    total_tokens=128,
    prefix_tokens=64,
    learning_rate=1e-4,
    weight_decay=1e-2,
    warmup_steps=500,
    grad_clip_norm=1.0,
    log_every=25,
    save_every=100,
    seed=42,
    shuffle_buffer=50_000,
    shared_slots=32,
    shared_dim=128,
    translator_dim=1024,
    translator_heads=4,
    translator_mlp_ratio=4,
    top_layers_ratio=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="float32",
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lsc_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def main() -> None:
    set_seed(CONFIG.seed)
    output_dir = Path(CONFIG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.json", CONFIG.__dict__)

    log_path = output_dir / "train.log"
    logger = setup_logger(log_path)
    logger.info("Starting training")
    logger.info("train_config=%s", CONFIG.__dict__)

    logger.info("[Setup] device=%s", CONFIG.device)
    logger.info("[Setup] loading models: A=%s, B=%s", CONFIG.model_a_id, CONFIG.model_b_id)
    model_a, model_b, tokenizer = build_models_and_tokenizer(CONFIG)
    translator_pool, model_specs, translated_model_specs = build_translator_pool(model_a, model_b, CONFIG)
    translator_pool.train()

    logger.info("[Setup] full model specs")
    logger.info(
        "  A: layers=%d, hidden=%d, heads=%d",
        model_specs["A"].num_layers,
        model_specs["A"].hidden_size,
        model_specs["A"].num_heads,
    )
    logger.info(
        "  B: layers=%d, hidden=%d, heads=%d",
        model_specs["B"].num_layers,
        model_specs["B"].hidden_size,
        model_specs["B"].num_heads,
    )
    logger.info("[Setup] translated top-layer specs")
    logger.info(
        "  A_top: layers=%d / %d",
        translated_model_specs["A"].num_layers,
        model_specs["A"].num_layers,
    )
    logger.info(
        "  B_top: layers=%d / %d",
        translated_model_specs["B"].num_layers,
        model_specs["B"].num_layers,
    )
    logger.info("[Setup] top_layers_ratio = %.4f", CONFIG.top_layers_ratio)
    logger.info("[Setup] trainable translator params = %s", f"{count_trainable_parameters(translator_pool):,}")

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

            translated_a_to_b_top = translator_pool.translate_top_layers(
                past_key_values=past_a,
                src_name="A",
                dst_name="B",
                dst_spec=translated_model_specs["B"],
            )
            translated_b_to_a_top = translator_pool.translate_top_layers(
                past_key_values=past_b,
                src_name="B",
                dst_name="A",
                dst_spec=translated_model_specs["A"],
            )

            mixed_past_for_b = replace_top_layers(
                base_past_key_values=past_b,
                translated_top_past_key_values=translated_a_to_b_top,
            )
            mixed_past_for_a = replace_top_layers(
                base_past_key_values=past_a,
                translated_top_past_key_values=translated_b_to_a_top,
            )

            loss_a_to_b = compute_suffix_lm_loss(
                target_model=model_b,
                past_key_values=mixed_past_for_b,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
            )
            loss_b_to_a = compute_suffix_lm_loss(
                target_model=model_a,
                past_key_values=mixed_past_for_a,
                lm_input_ids=lm_input_ids,
                lm_labels=lm_labels,
            )

            loss = loss_a_to_b + loss_b_to_a
            loss = loss / CONFIG.grad_accum_steps
            loss.backward()
            step_loss_value += loss.item()

        torch.nn.utils.clip_grad_norm_(translator_pool.parameters(), CONFIG.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss_value
        if step % CONFIG.log_every == 0:
            avg_loss = running_loss / CONFIG.log_every
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{scheduler.lr:.2e}",
            )
            logger.info(
                "[Step %04d] total_bidirectional_suffix_lm_loss=%.4f | lr=%.2e",
                step,
                avg_loss,
                scheduler.lr,
            )
            running_loss = 0.0

        # if step % CONFIG.save_every == 0:
        #     checkpoint_path = output_dir / f"checkpoint_step_{step:04d}.pt"
        #     save_checkpoint(
        #         output_path=str(checkpoint_path),
        #         translator_pool=translator_pool,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         train_config=CONFIG,
        #         step=step,
        #         extra={
        #             "note": "Toy checkpoint trained with bidirectional suffix LM loss only.",
        #             "model_a": CONFIG.model_a_id,
        #             "model_b": CONFIG.model_b_id,
        #             "top_layers_ratio": CONFIG.top_layers_ratio,
        #         },
        #     )
        #     logger.info("[Checkpoint] saved to %s", checkpoint_path)

    final_path = output_dir / "final_checkpoint.pt"
    save_checkpoint(
        output_path=str(final_path),
        translator_pool=translator_pool,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=CONFIG,
        step=CONFIG.max_steps,
        extra={
            "note": "Final toy checkpoint trained with bidirectional suffix LM loss only.",
            "model_a": CONFIG.model_a_id,
            "model_b": CONFIG.model_b_id,
            "top_layers_ratio": CONFIG.top_layers_ratio,
        },
    )
    logger.info("[Done] final checkpoint saved to %s", final_path)
    logger.info("Saved train log to %s", log_path)


if __name__ == "__main__":
    main()
