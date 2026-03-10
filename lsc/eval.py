from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from common import *
from lsc.train import load_translator_pool_from_checkpoint


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
    translated_model_specs,
    model_a,
    model_b,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device
    choice_labels = get_choice_labels(spec.answer_mode)
    choice_token_ids = build_choice_token_ids(tokenizer, choice_labels)

    path_metrics = {
        "A_to_B": RunningAverage(),
        "B_to_A": RunningAverage(),
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

            target_top_b = slice_top_layers(
                past_key_values=past_b,
                top_layers_to_translate=translated_model_specs["B"].num_layers,
            )
            target_top_a = slice_top_layers(
                past_key_values=past_a,
                top_layers_to_translate=translated_model_specs["A"].num_layers,
            )

            cosine_a_to_b = cosine_similarity_between_past(translated_a_to_b_top, target_top_b)
            cosine_b_to_a = cosine_similarity_between_past(translated_b_to_a_top, target_top_a)

            mixed_past_for_b = replace_top_layers(
                base_past_key_values=past_b,
                translated_top_past_key_values=translated_a_to_b_top,
            )
            mixed_past_for_a = replace_top_layers(
                base_past_key_values=past_a,
                translated_top_past_key_values=translated_b_to_a_top,
            )

            translated_scores_b = score_answer_choices(
                model=model_b,
                past_key_values=mixed_past_for_b,
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )
            translated_scores_a = score_answer_choices(
                model=model_a,
                past_key_values=mixed_past_for_a,
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )

            native_scores_b = score_answer_choices(
                model=model_b,
                past_key_values=past_b,
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )
            native_scores_a = score_answer_choices(
                model=model_a,
                past_key_values=past_a,
                seed_token=seed_token,
                choice_token_ids=choice_token_ids,
            )

            translated_pred_b = predict_answer_label(translated_scores_b)
            translated_pred_a = predict_answer_label(translated_scores_a)
            native_pred_b = predict_answer_label(native_scores_b)
            native_pred_a = predict_answer_label(native_scores_a)

            acc_a_to_b = 1.0 if translated_pred_b == gold_answer else 0.0
            acc_b_to_a = 1.0 if translated_pred_a == gold_answer else 0.0
            native_acc_b = 1.0 if native_pred_b == gold_answer else 0.0
            native_acc_a = 1.0 if native_pred_a == gold_answer else 0.0

            path_metrics["A_to_B"].update(cosine_a_to_b, acc_a_to_b, native_acc_b, 1)
            path_metrics["B_to_A"].update(cosine_b_to_a, acc_b_to_a, native_acc_a, 1)

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
    translated_model_specs,
    model_a,
    model_b,
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

        translated_scores_b = score_answer_choices(
            model=model_b,
            past_key_values=mixed_past_for_b,
            seed_token=seed_token,
            choice_token_ids=choice_token_ids,
        )
        translated_scores_a = score_answer_choices(
            model=model_a,
            past_key_values=mixed_past_for_a,
            seed_token=seed_token,
            choice_token_ids=choice_token_ids,
        )

        native_scores_b = score_answer_choices(
            model=model_b,
            past_key_values=past_b,
            seed_token=seed_token,
            choice_token_ids=choice_token_ids,
        )
        native_scores_a = score_answer_choices(
            model=model_a,
            past_key_values=past_a,
            seed_token=seed_token,
            choice_token_ids=choice_token_ids,
        )

        translated_pred_b = predict_answer_label(translated_scores_b)
        translated_pred_a = predict_answer_label(translated_scores_a)
        native_pred_b = predict_answer_label(native_scores_b)
        native_pred_a = predict_answer_label(native_scores_a)

        logger.info(
            "--- Sample %d/%d | %s ---",
            sample_idx + 1,
            eval_config.num_generation_samples_per_dataset,
            spec.name_for_log,
        )
        logger.info("prefix(question):\n%s", prefix_text)
        logger.info("gold_answer: %s", gold_answer)
        logger.info(
            "A_to_B translated_scores: %s | pred=%s | correct=%s",
            format_choice_scores(translated_scores_b),
            translated_pred_b,
            translated_pred_b == gold_answer,
        )
        logger.info(
            "A_to_B native_baseline_scores: %s | pred=%s | correct=%s",
            format_choice_scores(native_scores_b),
            native_pred_b,
            native_pred_b == gold_answer,
        )
        logger.info(
            "B_to_A translated_scores: %s | pred=%s | correct=%s",
            format_choice_scores(translated_scores_a),
            translated_pred_a,
            translated_pred_a == gold_answer,
        )
        logger.info(
            "B_to_A native_baseline_scores: %s | pred=%s | correct=%s",
            format_choice_scores(native_scores_a),
            native_pred_a,
            native_pred_a == gold_answer,
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
    logger = setup_logger("lsc_eval", log_path)
    logger.info("Starting evaluation")
    logger.info("checkpoint_path=%s", checkpoint_path)
    logger.info("eval_config=%s", asdict(eval_config))

    (
        train_config,
        translator_pool,
        model_specs,
        translated_model_specs,
        model_a,
        model_b,
        tokenizer,
    ) = load_translator_pool_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device_override=eval_config.device,
    )

    translator_pool.eval()
    model_a.eval()
    model_b.eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("model_A=%s", train_config.model_a_id)
    logger.info("model_B=%s", train_config.model_b_id)
    logger.info("top_layers_ratio=%.6f", train_config.top_layers_ratio)
    logger.info(
        "translated_layers: A_top=%d/%d | B_top=%d/%d",
        translated_model_specs["A"].num_layers,
        model_specs["A"].num_layers,
        translated_model_specs["B"].num_layers,
        model_specs["B"].num_layers,
    )
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
            translated_model_specs=translated_model_specs,
            model_a=model_a,
            model_b=model_b,
            logger=logger,
        )
        all_results[spec.name_for_log] = results

        log_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            model_a_id=train_config.model_a_id,
            model_b_id=train_config.model_b_id,
            active_directions=("A_to_B", "B_to_A"),
        )

        log_qa_score_samples(
            spec=spec,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
            translated_model_specs=translated_model_specs,
            model_a=model_a,
            model_b=model_b,
            logger=logger,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("===== FINAL SUMMARY =====")
    for dataset_name, result in all_results.items():
        logger.info(
            "%s | A_to_B(cos=%.6f, acc=%.6f, native_acc=%.6f) | "
            "B_to_A(cos=%.6f, acc=%.6f, native_acc=%.6f) | "
            "AVG(cos=%.6f, acc=%.6f, native_acc=%.6f)",
            dataset_name,
            result["A_to_B"]["cosine"],
            result["A_to_B"]["accuracy"],
            result["A_to_B"]["native_accuracy"],
            result["B_to_A"]["cosine"],
            result["B_to_A"]["accuracy"],
            result["B_to_A"]["native_accuracy"],
            result["AVG"]["cosine"],
            result["AVG"]["accuracy"],
            result["AVG"]["native_accuracy"],
        )

    overall_results = summarize_overall_results(all_results, ("A_to_B", "B_to_A"))
    log_overall_result(
        logger=logger,
        results=overall_results,
        model_a_id=train_config.model_a_id,
        model_b_id=train_config.model_b_id,
        active_directions=("A_to_B", "B_to_A"),
    )

    logger.info("Done. Saved log to %s", log_path)
    return log_path
