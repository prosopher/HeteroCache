from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from common import *


@dataclass
class EvalConfig:
    checkpoint_path: str = "./outputs/lsc_toy/final_checkpoint.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # evaluation sampling
    batch_size: int = 1
    num_workers: int = 0
    max_examples_per_dataset: int = 512
    seed: int = 42

    # streaming / shuffling
    shuffle_eval_stream: bool = True
    shuffle_buffer: int = 10_000

    # score sanity-check samples
    num_generation_samples_per_dataset: int = 2

    log_filename: str = "eval.log"


CONFIG = EvalConfig()


@dataclass
class HFDatasetSpec:
    name_for_log: str
    dataset_path: str
    dataset_name: Optional[str]
    split: str
    answer_mode: str
    question_field: str = "question"
    streaming: bool = False


class HFQAPairStream(IterableDataset):
    def __init__(
        self,
        spec: HFDatasetSpec,
        max_examples: int,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _load_dataset(self):
        candidates = [(self.spec.dataset_path, self.spec.dataset_name)]

        last_error = None
        for dataset_path, dataset_name in candidates:
            try:
                if dataset_name is None:
                    return load_dataset(
                        dataset_path,
                        split=self.spec.split,
                        streaming=self.spec.streaming,
                    )
                return load_dataset(
                    dataset_path,
                    dataset_name,
                    split=self.spec.split,
                    streaming=self.spec.streaming,
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to load dataset {self.spec.name_for_log} "
            f"with candidates={candidates}"
        ) from last_error

    def __iter__(self):
        dataset = self._load_dataset()
        if self.shuffle:
            if self.spec.streaming:
                dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            else:
                dataset = dataset.shuffle(seed=self.seed)

        emitted = 0

        for example in dataset:
            qa_pair = extract_question_and_answer(self.spec, example)
            if qa_pair is None:
                continue

            yield qa_pair
            emitted += 1
            if emitted >= self.max_examples:
                return


class RunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.accuracy_sum = 0.0
        self.native_accuracy_sum = 0.0
        self.count = 0

    def update(self, cosine_value: float, accuracy_value: float, native_accuracy_value: float, n: int) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.accuracy_sum += float(accuracy_value) * n
        self.native_accuracy_sum += float(native_accuracy_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "accuracy": self.accuracy_sum / self.count,
            "native_accuracy": self.native_accuracy_sum / self.count,
            "count": self.count,
        }


def build_eval_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFQAPairStream(
        spec=spec,
        max_examples=eval_config.max_examples_per_dataset,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    return DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        collate_fn=lambda batch: batch,
    )


def build_sample_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFQAPairStream(
        spec=spec,
        max_examples=eval_config.num_generation_samples_per_dataset,
        shuffle=False,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=eval_config.num_workers,
        collate_fn=lambda batch: batch,
    )


def extract_question_and_answer(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, str]]:
    question = example.get(spec.question_field, "")
    if not isinstance(question, str) or not question.strip():
        return None

    if spec.answer_mode == "boolq":
        answer_value = example.get("answer", None)
        if not isinstance(answer_value, bool):
            return None

        return {
            "question": question.strip(),
            "answer": "yes" if answer_value else "no",
        }

    if spec.answer_mode == "pubmed_qa":
        answer_value = example.get("final_decision", None)
        if not isinstance(answer_value, str):
            return None

        normalized_answer = answer_value.strip().lower()
        if normalized_answer not in {"yes", "no", "maybe"}:
            return None

        return {
            "question": question.strip(),
            "answer": normalized_answer,
        }

    raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")


def format_question_prefix(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def prepare_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_question_prefix(question)
    tokenized = tokenizer(prefix_text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Question prefix must tokenize to at least 2 tokens.")
    cache_ids = input_ids[:, :-1]
    seed_token = input_ids[:, -1:]
    return {
        "prefix_text": prefix_text,
        "full_prefix_ids": input_ids,
        "cache_ids": cache_ids,
        "seed_token": seed_token,
    }


def get_choice_labels(answer_mode: str) -> List[str]:
    if answer_mode == "boolq":
        return ["yes", "no"]
    if answer_mode == "pubmed_qa":
        return ["yes", "no", "maybe"]
    raise ValueError(f"Unsupported answer_mode: {answer_mode}")


def build_choice_token_ids(tokenizer, choice_labels: List[str]) -> Dict[str, torch.Tensor]:
    choice_token_ids = {}
    for label in choice_labels:
        token_ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
        if len(token_ids) < 1:
            raise ValueError(f"Failed to tokenize answer label: {label}")
        choice_token_ids[label] = torch.tensor(token_ids, dtype=torch.long)
    return choice_token_ids


def score_candidate_logprob(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> float:
    device = seed_token.device
    candidate_ids = candidate_token_ids.to(device).unsqueeze(0)

    if candidate_ids.shape[1] == 1:
        scoring_input_ids = seed_token
    else:
        scoring_input_ids = torch.cat([seed_token, candidate_ids[:, :-1]], dim=1)

    outputs = model(
        input_ids=scoring_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    log_probs = F.log_softmax(outputs.logits, dim=-1)
    token_log_probs = log_probs.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=1).item()


def score_answer_choices(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    choice_token_ids: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        label: score_candidate_logprob(
            model=model,
            past_key_values=past_key_values,
            seed_token=seed_token,
            candidate_token_ids=choice_token_ids[label],
        )
        for label in choice_token_ids
    }


def predict_answer_label(choice_scores: Dict[str, float]) -> str:
    return max(choice_scores.items(), key=lambda item: item[1])[0]


def format_choice_scores(choice_scores: Dict[str, float]) -> str:
    return " | ".join(
        f"{label}={score:.6f}"
        for label, score in choice_scores.items()
    )


def summarize_path_metrics(path_metrics: Dict[str, RunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}
    total_cosine_sum = 0.0
    total_accuracy_sum = 0.0
    total_native_accuracy_sum = 0.0
    total_count = 0

    for path_name, meter in path_metrics.items():
        path_result = meter.summary()
        results[path_name] = path_result

        total_cosine_sum += meter.cosine_sum
        total_accuracy_sum += meter.accuracy_sum
        total_native_accuracy_sum += meter.native_accuracy_sum
        total_count += meter.count

    if total_count == 0:
        results["AVG"] = {
            "cosine": float("nan"),
            "accuracy": float("nan"),
            "native_accuracy": float("nan"),
            "count": 0,
        }
    else:
        results["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "accuracy": total_accuracy_sum / total_count,
            "native_accuracy": total_native_accuracy_sum / total_count,
            "count": total_count,
        }

    return results


def summarize_overall_results(all_results: Dict[str, Dict[str, Dict[str, float]]], active_directions) -> Dict[str, Dict[str, float]]:
    overall = {
        direction: {"cosine_sum": 0.0, "accuracy_sum": 0.0, "native_accuracy_sum": 0.0, "count": 0}
        for direction in active_directions
    }

    for dataset_result in all_results.values():
        for direction in active_directions:
            row = dataset_result[direction]
            count = int(row["count"])
            overall[direction]["cosine_sum"] += float(row["cosine"]) * count
            overall[direction]["accuracy_sum"] += float(row["accuracy"]) * count
            overall[direction]["native_accuracy_sum"] += float(row["native_accuracy"]) * count
            overall[direction]["count"] += count

    summarized = {}
    total_cosine_sum = 0.0
    total_accuracy_sum = 0.0
    total_native_accuracy_sum = 0.0
    total_count = 0

    for direction in active_directions:
        count = overall[direction]["count"]
        if count == 0:
            summarized[direction] = {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        else:
            summarized[direction] = {
                "cosine": overall[direction]["cosine_sum"] / count,
                "accuracy": overall[direction]["accuracy_sum"] / count,
                "native_accuracy": overall[direction]["native_accuracy_sum"] / count,
                "count": count,
            }

        total_cosine_sum += overall[direction]["cosine_sum"]
        total_accuracy_sum += overall[direction]["accuracy_sum"]
        total_native_accuracy_sum += overall[direction]["native_accuracy_sum"]
        total_count += count

    if total_count == 0:
        summarized["AVG"] = {
            "cosine": float("nan"),
            "accuracy": float("nan"),
            "native_accuracy": float("nan"),
            "count": 0,
        }
    else:
        summarized["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "accuracy": total_accuracy_sum / total_count,
            "native_accuracy": total_native_accuracy_sum / total_count,
            "count": total_count,
        }

    return summarized


def build_direction_pretty_name(direction: str, model_a_id: str, model_b_id: str) -> str:
    if direction == "A_to_B":
        return f"A_to_B ({model_a_id} -> {model_b_id})"
    if direction == "B_to_A":
        return f"B_to_A ({model_b_id} -> {model_a_id})"
    return direction


@torch.inference_mode()
def evaluate_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    tokenizer,
    train_config,
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
                CONFIG.max_examples_per_dataset,
            )

    return summarize_path_metrics(path_metrics)


def log_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
    active_directions,
) -> None:
    logger.info("===== %s =====", dataset_name)
    for direction in list(active_directions) + ["AVG"]:
        row = results[direction]
        pretty_name = (
            "AVG (weighted over evaluated paths)"
            if direction == "AVG"
            else build_direction_pretty_name(direction, model_a_id, model_b_id)
        )
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            int(row["count"]),
        )


def log_overall_result(
    logger: logging.Logger,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
    active_directions,
) -> None:
    logger.info("===== OVERALL AVERAGE ACROSS DATASETS =====")
    for direction in list(active_directions) + ["AVG"]:
        row = results[direction]
        pretty_name = (
            "AVG (weighted over all datasets and evaluated paths)"
            if direction == "AVG"
            else build_direction_pretty_name(direction, model_a_id, model_b_id)
        )
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            int(row["count"]),
        )


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


def main() -> None:
    eval_config = CONFIG
    set_seed(eval_config.seed)

    checkpoint_path = Path(eval_config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log_path = checkpoint_path.parent / eval_config.log_filename
    logger = setup_logger("eval", log_path)
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


if __name__ == "__main__":
    main()
