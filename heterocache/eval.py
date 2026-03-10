from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from common import *
from heterocache.train import load_translator_pool_from_checkpoint


CONFIG = EvalConfig()


@torch.inference_mode()
def evaluate_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs,
    model_a,
    model_b,
    active_directions,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device
    choice_labels = get_choice_labels(spec.answer_mode)
    choice_token_ids = build_choice_token_ids(tokenizer, choice_labels)

    path_metrics = {
        direction: RunningAverage()
        for direction in active_directions
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]

            prefix = prepare_question_prefix(tokenizer=tokenizer, question=question, device=device)
            prefix_cache_ids = prefix["cache_ids"]
            seed_token = prefix["seed_token"]

            past_a = extract_past_key_values(model_a, prefix_cache_ids)
            past_b = extract_past_key_values(model_b, prefix_cache_ids)

            direction_contexts = {
                "A_to_B": {
                    "source_past": past_a,
                    "source_name": "A",
                    "target_name": "B",
                    "target_spec": model_specs["B"],
                    "target_full_past": past_b,
                    "target_model": model_b,
                },
                "B_to_A": {
                    "source_past": past_b,
                    "source_name": "B",
                    "target_name": "A",
                    "target_spec": model_specs["A"],
                    "target_full_past": past_a,
                    "target_model": model_a,
                },
            }

            for direction in active_directions:
                context = direction_contexts[direction]

                translated_top_past = translator_pool.translate_top_layers(
                    past_key_values=context["source_past"],
                    src_name=context["source_name"],
                    dst_name=context["target_name"],
                    dst_spec=context["target_spec"],
                )

                target_top = slice_top_layers(
                    past_key_values=context["target_full_past"],
                    top_layers_to_translate=train_config.top_layers_to_translate,
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                mixed_target_past = replace_top_layers(
                    base_past_key_values=context["target_full_past"],
                    translated_top_past_key_values=translated_top_past,
                )

                translated_scores = score_answer_choices(
                    model=context["target_model"],
                    past_key_values=mixed_target_past,
                    seed_token=seed_token,
                    choice_token_ids=choice_token_ids,
                )
                native_scores = score_answer_choices(
                    model=context["target_model"],
                    past_key_values=context["target_full_past"],
                    seed_token=seed_token,
                    choice_token_ids=choice_token_ids,
                )

                translated_pred = predict_answer_label(translated_scores)
                native_pred = predict_answer_label(native_scores)

                acc = 1.0 if translated_pred == gold_answer else 0.0
                native_acc = 1.0 if native_pred == gold_answer else 0.0

                path_metrics[direction].update(cosine_value, acc, native_acc, 1)

            processed_examples += 1

        if batch_idx % 50 == 0:
            logger.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_path_metrics(path_metrics)


@torch.inference_mode()
def log_qa_score_samples(
    spec: HFDatasetSpec,
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs,
    model_a,
    model_b,
    active_directions,
    logger: logging.Logger,
) -> None:
    if eval_config.num_generation_samples_per_dataset <= 0:
        return

    sample_dataloader = build_sample_dataloader(
        spec=spec,
        eval_config=eval_config,
    )
    choice_labels = get_choice_labels(spec.answer_mode)
    choice_token_ids = build_choice_token_ids(tokenizer, choice_labels)

    logger.info("===== SAMPLE QA SCORES: %s =====", spec.name_for_log)

    sample_idx = 0
    for batch in sample_dataloader:
        example = batch[0]
        question = example["question"]
        gold_answer = example["answer"]

        prefix = prepare_question_prefix(tokenizer=tokenizer, question=question, device=train_config.device)
        prefix_cache_ids = prefix["cache_ids"]
        seed_token = prefix["seed_token"]
        prefix_text = prefix["prefix_text"]

        past_a = extract_past_key_values(model_a, prefix_cache_ids)
        past_b = extract_past_key_values(model_b, prefix_cache_ids)

        direction_contexts = {
            "A_to_B": {
                "source_past": past_a,
                "source_name": "A",
                "target_name": "B",
                "target_spec": model_specs["B"],
                "target_full_past": past_b,
                "target_model": model_b,
            },
            "B_to_A": {
                "source_past": past_b,
                "source_name": "B",
                "target_name": "A",
                "target_spec": model_specs["A"],
                "target_full_past": past_a,
                "target_model": model_a,
            },
        }

        logger.info(
            "--- Sample %d/%d | %s ---",
            sample_idx + 1,
            eval_config.num_generation_samples_per_dataset,
            spec.name_for_log,
        )
        logger.info("prefix(question):\n%s", prefix_text)
        logger.info("gold_answer: %s", gold_answer)

        for direction in active_directions:
            context = direction_contexts[direction]

            translated_top_past = translator_pool.translate_top_layers(
                past_key_values=context["source_past"],
                src_name=context["source_name"],
                dst_name=context["target_name"],
                dst_spec=context["target_spec"],
            )
            mixed_target_past = replace_top_layers(
                base_past_key_values=context["target_full_past"],
                translated_top_past_key_values=translated_top_past,
            )

            translated_scores = score_answer_choices(
                model=context["target_model"],
                past_key_values=mixed_target_past,
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )
            native_scores = score_answer_choices(
                model=context["target_model"],
                past_key_values=context["target_full_past"],
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )

            translated_pred = predict_answer_label(translated_scores)
            native_pred = predict_answer_label(native_scores)
            pretty_name = build_direction_pretty_name(direction, train_config.model_a_id, train_config.model_b_id)

            logger.info(
                "%s translated_scores: %s | pred=%s | correct=%s",
                pretty_name,
                format_choice_scores(translated_scores),
                translated_pred,
                translated_pred == gold_answer,
            )
            logger.info(
                "%s native_baseline_scores: %s | pred=%s | correct=%s",
                pretty_name,
                format_choice_scores(native_scores),
                native_pred,
                native_pred == gold_answer,
            )

        sample_idx += 1
        if sample_idx >= eval_config.num_generation_samples_per_dataset:
            break


def run_eval(eval_config: Optional[EvalConfig] = None) -> Path:
    eval_config = EvalConfig(**(CONFIG.__dict__ if eval_config is None else eval_config.__dict__))
    set_seed(eval_config.seed)

    checkpoint_path = Path(eval_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    write_json(checkpoint_path.parent / "eval_config.json", asdict(eval_config))

    log_path = checkpoint_path.parent / eval_config.log_filename
    logger = setup_logger("heterocache_eval", log_path)
    logger.info("Starting evaluation")
    logger.info("checkpoint_path=%s", checkpoint_path)
    logger.info("eval_config=%s", asdict(eval_config))

    (
        train_config,
        translator_pool,
        model_specs,
        model_a,
        model_b,
        tokenizer,
    ) = load_translator_pool_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device_override=eval_config.device,
    )

    active_directions = parse_train_directions(train_config.train_directions)

    translator_pool.eval()
    model_a.eval()
    model_b.eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("model_A=%s", train_config.model_a_id)
    logger.info("model_B=%s", train_config.model_b_id)
    logger.info("top_layers_to_translate=%d", train_config.top_layers_to_translate)
    logger.info("active_directions=%s", active_directions)
    logger.info("translation_mode=replace_top_layers_after_target_forward")
    logger.info("qa_eval_log_path=%s", log_path)

    dataset_specs = [
        HFDatasetSpec(
            name_for_log="BoolQ/validation",
            dataset_path="boolq",
            dataset_name=None,
            split="validation",
            answer_mode="boolq",
            question_field="question",
            streaming=False,
        ),
        HFDatasetSpec(
            name_for_log="PubMedQA/pqa_labeled/train",
            dataset_path="qiaojin/PubMedQA",
            dataset_name="pqa_labeled",
            split="train",
            answer_mode="pubmed_qa",
            question_field="question",
            streaming=False,
        ),
    ]

    all_results = {}

    for spec in dataset_specs:
        logger.info("Preparing dataloader for %s", spec.name_for_log)
        dataloader = build_eval_dataloader(
            spec=spec,
            eval_config=eval_config,
        )

        results = evaluate_dataset(
            spec=spec,
            dataloader=dataloader,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            model_a=model_a,
            model_b=model_b,
            active_directions=active_directions,
            logger=logger,
        )
        all_results[spec.name_for_log] = results

        log_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            model_a_id=train_config.model_a_id,
            model_b_id=train_config.model_b_id,
            active_directions=active_directions,
        )

        log_qa_score_samples(
            spec=spec,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            model_a=model_a,
            model_b=model_b,
            active_directions=active_directions,
            logger=logger,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("===== FINAL SUMMARY =====")
    for dataset_name, result in all_results.items():
        summary_parts = []
        for direction in active_directions:
            row = result[direction]
            summary_parts.append(
                f"{direction}(cos={row['cosine']:.6f}, acc={row['accuracy']:.6f}, native_acc={row['native_accuracy']:.6f})"
            )
        avg_row = result["AVG"]
        summary_parts.append(
            f"AVG(cos={avg_row['cosine']:.6f}, acc={avg_row['accuracy']:.6f}, native_acc={avg_row['native_accuracy']:.6f})"
        )
        logger.info("%s | %s", dataset_name, " | ".join(summary_parts))

    overall_results = summarize_overall_results(all_results, active_directions)
    log_overall_result(
        logger=logger,
        results=overall_results,
        model_a_id=train_config.model_a_id,
        model_b_id=train_config.model_b_id,
        active_directions=active_directions,
    )

    logger.info("Done. Saved log to %s", log_path)
    return log_path
