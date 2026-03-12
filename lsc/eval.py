from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from eval_util import *
from lsc.train import load_translator_pool_from_checkpoint


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
    active_directions,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device

    path_metrics = {
        direction: RunningAverage()
        for direction in active_directions
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            gold_answer = example["answer"]

            prefix = prepare_question_prefix(
                tokenizer=tokenizer,
                question=question,
                device=device,
                choices=example.get("choices"),
                subject=example.get("subject"),
                context=example.get("context"),
                answer_mode=spec.answer_mode,
            )
            prefix_cache_ids = prefix["cache_ids"]
            seed_token = prefix["seed_token"]

            candidate_token_ids = build_logit_answer_candidates(
                tokenizer=tokenizer,
                spec=spec,
                example=example,
            )

            past_a = extract_past_key_values(model_a, prefix_cache_ids)
            past_b = extract_past_key_values(model_b, prefix_cache_ids)

            direction_contexts = {
                "A_to_B": {
                    "source_past": past_a,
                    "source_name": "A",
                    "target_name": "B",
                    "target_top_spec": translated_model_specs["B"],
                    "target_full_past": past_b,
                    "target_model": model_b,
                },
                "B_to_A": {
                    "source_past": past_b,
                    "source_name": "B",
                    "target_name": "A",
                    "target_top_spec": translated_model_specs["A"],
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
                    dst_spec=context["target_top_spec"],
                )

                target_top = slice_top_layers(
                    past_key_values=context["target_full_past"],
                    top_layers_to_translate=context["target_top_spec"].num_layers,
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
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
                )
                native_scores = score_answer_choices(
                    model=context["target_model"],
                    past_key_values=context["target_full_past"],
                    seed_token=seed_token,
                    choice_token_ids=candidate_token_ids,
                    normalize_by_length=True,
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
def evaluate_generation_dataset(
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
    active_directions,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device

    path_metrics = {
        direction: GenerationRunningAverage()
        for direction in active_directions
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            context_text = example["context"]
            gold_answers = example["answers"]

            question_cache_ids = None

            if spec.answer_mode == "squad":
                context_prefix = prepare_generation_context_inputs(
                    tokenizer=tokenizer,
                    context=context_text,
                    device=device,
                )
                context_input_ids = context_prefix["input_ids"]

                question_prefix = prepare_generation_question_prefix(
                    tokenizer=tokenizer,
                    question=question,
                    device=device,
                )
                question_cache_ids = question_prefix["cache_ids"]
                seed_token = question_prefix["seed_token"]

                past_a = extract_past_key_values(model_a, context_input_ids)
                past_b = extract_past_key_values(model_b, context_input_ids)
            else:
                prefix = prepare_generation_prefix(
                    tokenizer=tokenizer,
                    context=context_text,
                    question=question,
                    device=device,
                )
                prefix_cache_ids = prefix["cache_ids"]
                seed_token = prefix["seed_token"]

                past_a = extract_past_key_values(model_a, prefix_cache_ids)
                past_b = extract_past_key_values(model_b, prefix_cache_ids)

            direction_contexts = {
                "A_to_B": {
                    "source_past": past_a,
                    "source_name": "A",
                    "target_name": "B",
                    "target_top_spec": translated_model_specs["B"],
                    "target_full_past": past_b,
                    "target_model": model_b,
                },
                "B_to_A": {
                    "source_past": past_b,
                    "source_name": "B",
                    "target_name": "A",
                    "target_top_spec": translated_model_specs["A"],
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
                    dst_spec=context["target_top_spec"],
                )

                target_top = slice_top_layers(
                    past_key_values=context["target_full_past"],
                    top_layers_to_translate=context["target_top_spec"].num_layers,
                )
                cosine_value = cosine_similarity_between_past(translated_top_past, target_top)

                mixed_target_past = replace_top_layers(
                    base_past_key_values=context["target_full_past"],
                    translated_top_past_key_values=translated_top_past,
                )

                if spec.answer_mode == "squad":
                    translated_generation_past = append_input_ids_to_past(
                        model=context["target_model"],
                        past_key_values=mixed_target_past,
                        input_ids=question_cache_ids,
                    )
                    native_generation_past = append_input_ids_to_past(
                        model=context["target_model"],
                        past_key_values=context["target_full_past"],
                        input_ids=question_cache_ids,
                    )
                else:
                    translated_generation_past = mixed_target_past
                    native_generation_past = context["target_full_past"]

                translated_answer = generate_greedy_answer(
                    model=context["target_model"],
                    tokenizer=tokenizer,
                    past_key_values=translated_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=eval_config.generation_max_new_tokens,
                )
                native_answer = generate_greedy_answer(
                    model=context["target_model"],
                    tokenizer=tokenizer,
                    past_key_values=native_generation_past,
                    seed_token=seed_token,
                    max_new_tokens=eval_config.generation_max_new_tokens,
                )

                exact_match = compute_generation_exact_match(translated_answer, gold_answers)
                f1 = compute_generation_f1(translated_answer, gold_answers)
                native_exact_match = compute_generation_exact_match(native_answer, gold_answers)
                native_f1 = compute_generation_f1(native_answer, gold_answers)

                path_metrics[direction].update(
                    cosine_value=cosine_value,
                    exact_match_value=exact_match,
                    f1_value=f1,
                    native_exact_match_value=native_exact_match,
                    native_f1_value=native_f1,
                    n=1,
                )

            processed_examples += 1

        if batch_idx % 25 == 0:
            logger.info(
                "[%s] generation progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                eval_config.max_examples_per_dataset,
            )

    return summarize_generation_path_metrics(path_metrics)


def run_eval(eval_config: EvalConfig) -> Path:
    if eval_config.checkpoint_path is None:
        raise ValueError("EvalConfig.checkpoint_path must be set before run_eval.")
    if eval_config.output_path is None:
        raise ValueError("EvalConfig.output_path must be initialized before run_eval.")

    set_seed(eval_config.seed)

    checkpoint_path = Path(eval_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = get_eval_config_path(eval_config.output_path)
    write_json(str(config_path), asdict(eval_config))

    log_path = get_eval_log_path(eval_config.output_path)
    logger = setup_logger(f"{eval_config.alg}_eval", log_path)
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

    active_directions = parse_train_directions(train_config.train_directions)

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
    logger.info("active_directions=%s", active_directions)
    logger.info("translation_mode=replace_top_layers_after_target_forward")
    logger.info("qa_eval_log_path=%s", log_path)

    logit_dataset_specs = [
        HFDatasetSpec(
            name_for_log="BoolQ/validation",
            dataset_path="google/boolq",
            dataset_name=None,
            split="validation",
            answer_mode="boolq",
            question_field="question",
            context_field="passage",
            streaming=False,
        ),
        HFDatasetSpec(
            name_for_log="PubMedQA/pqa_labeled/train",
            dataset_path="qiaojin/PubMedQA",
            dataset_name="pqa_labeled",
            split="train",
            answer_mode="pubmed_qa",
            question_field="question",
            context_field="context",
            streaming=False,
        ),
        HFDatasetSpec(
            name_for_log="MMLU/all/validation",
            dataset_path="cais/mmlu",
            dataset_name="all",
            split="validation",
            answer_mode="mmlu",
            question_field="question",
            subject_field="subject",
            streaming=False,
        ),
    ]

    generation_dataset_specs = [
        HFDatasetSpec(
            name_for_log="SQuAD/validation",
            dataset_path="rajpurkar/squad",
            dataset_name=None,
            split="validation",
            answer_mode="squad",
            question_field="question",
            context_field="context",
            answers_field="answers",
            streaming=False,
        ),
    ]

    all_logit_results = {}
    all_generation_results = {}

    for spec in logit_dataset_specs:
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
            active_directions=active_directions,
            logger=logger,
        )
        all_logit_results[spec.name_for_log] = results

        log_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            model_a_id=train_config.model_a_id,
            model_b_id=train_config.model_b_id,
            active_directions=active_directions,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for spec in generation_dataset_specs:
        logger.info("Preparing generation dataloader for %s", spec.name_for_log)
        dataloader = build_eval_dataloader(
            spec=spec,
            eval_config=eval_config,
        )

        results = evaluate_generation_dataset(
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
            active_directions=active_directions,
            logger=logger,
        )
        all_generation_results[spec.name_for_log] = results

        log_generation_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            model_a_id=train_config.model_a_id,
            model_b_id=train_config.model_b_id,
            active_directions=active_directions,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary_markdown = build_final_summary_markdown(
        alg=eval_config.alg,
        model_a_id=train_config.model_a_id,
        model_b_id=train_config.model_b_id,
        active_directions=active_directions,
        all_logit_results=all_logit_results,
        all_generation_results=all_generation_results,
    )
    logger.info("===== FINAL MARKDOWN SUMMARY =====\n%s", final_summary_markdown)

    logger.info("Done. Saved log to %s", log_path)
    return log_path
