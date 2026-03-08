# eval.py
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from common import (
    cosine_similarity_between_past,
    extract_past_key_values,
    load_translator_pool_from_checkpoint,
    replace_top_layers,
    set_seed,
    slice_top_layers,
)


@dataclass
class EvalConfig:
    checkpoint_path: str = "./outputs/lsc_toy_topn/final_checkpoint.pt"
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


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lsc_eval")
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

    if spec.answer_mode != "boolq":
        raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")

    answer_value = example.get("answer", None)
    if not isinstance(answer_value, bool):
        return None

    return {
        "question": question.strip(),
        "answer": "yes" if answer_value else "no",
    }


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


def build_boolq_choice_token_ids(tokenizer) -> Dict[str, torch.Tensor]:
    choice_token_ids = {}
    for label in ("yes", "no"):
        token_ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
        if len(token_ids) < 1:
            raise ValueError(f"Failed to tokenize BoolQ label: {label}")
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


def score_boolq_choices(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    boolq_choice_token_ids: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        "yes": score_candidate_logprob(
            model=model,
            past_key_values=past_key_values,
            seed_token=seed_token,
            candidate_token_ids=boolq_choice_token_ids["yes"],
        ),
        "no": score_candidate_logprob(
            model=model,
            past_key_values=past_key_values,
            seed_token=seed_token,
            candidate_token_ids=boolq_choice_token_ids["no"],
        ),
    }


def predict_boolq_label(choice_scores: Dict[str, float]) -> str:
    return "yes" if choice_scores["yes"] >= choice_scores["no"] else "no"


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


def summarize_overall_results(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    overall = {
        "A_to_B": {"cosine_sum": 0.0, "accuracy_sum": 0.0, "native_accuracy_sum": 0.0, "count": 0},
        "B_to_A": {"cosine_sum": 0.0, "accuracy_sum": 0.0, "native_accuracy_sum": 0.0, "count": 0},
    }

    for dataset_result in all_results.values():
        for key in ("A_to_B", "B_to_A"):
            row = dataset_result[key]
            count = int(row["count"])
            overall[key]["cosine_sum"] += float(row["cosine"]) * count
            overall[key]["accuracy_sum"] += float(row["accuracy"]) * count
            overall[key]["native_accuracy_sum"] += float(row["native_accuracy"]) * count
            overall[key]["count"] += count

    summarized = {}
    total_cosine_sum = 0.0
    total_accuracy_sum = 0.0
    total_native_accuracy_sum = 0.0
    total_count = 0

    for key in ("A_to_B", "B_to_A"):
        count = overall[key]["count"]
        if count == 0:
            summarized[key] = {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "native_accuracy": float("nan"),
                "count": 0,
            }
        else:
            summarized[key] = {
                "cosine": overall[key]["cosine_sum"] / count,
                "accuracy": overall[key]["accuracy_sum"] / count,
                "native_accuracy": overall[key]["native_accuracy_sum"] / count,
                "count": count,
            }

        total_cosine_sum += overall[key]["cosine_sum"]
        total_accuracy_sum += overall[key]["accuracy_sum"]
        total_native_accuracy_sum += overall[key]["native_accuracy_sum"]
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
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    device = train_config.device
    boolq_choice_token_ids = build_boolq_choice_token_ids(tokenizer)

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
                dst_spec=model_specs["B"],
            )
            translated_b_to_a_top = translator_pool.translate_top_layers(
                past_key_values=past_b,
                src_name="B",
                dst_name="A",
                dst_spec=model_specs["A"],
            )

            target_top_b = slice_top_layers(
                past_key_values=past_b,
                top_layers_to_translate=train_config.top_layers_to_translate,
            )
            target_top_a = slice_top_layers(
                past_key_values=past_a,
                top_layers_to_translate=train_config.top_layers_to_translate,
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

            translated_scores_b = score_boolq_choices(
                model=model_b,
                past_key_values=mixed_past_for_b,
                seed_token=seed_token,
                boolq_choice_token_ids=boolq_choice_token_ids,
            )
            translated_scores_a = score_boolq_choices(
                model=model_a,
                past_key_values=mixed_past_for_a,
                seed_token=seed_token,
                boolq_choice_token_ids=boolq_choice_token_ids,
            )

            native_scores_b = score_boolq_choices(
                model=model_b,
                past_key_values=past_b,
                seed_token=seed_token,
                boolq_choice_token_ids=boolq_choice_token_ids,
            )
            native_scores_a = score_boolq_choices(
                model=model_a,
                past_key_values=past_a,
                seed_token=seed_token,
                boolq_choice_token_ids=boolq_choice_token_ids,
            )

            translated_pred_b = predict_boolq_label(translated_scores_b)
            translated_pred_a = predict_boolq_label(translated_scores_a)
            native_pred_b = predict_boolq_label(native_scores_b)
            native_pred_a = predict_boolq_label(native_scores_a)

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
                CONFIG.max_examples_per_dataset,
            )

    return summarize_path_metrics(path_metrics)


def log_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
) -> None:
    pretty_names = {
        "A_to_B": f"A_to_B ({model_a_id} -> {model_b_id})",
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (weighted over both paths)",
    }

    logger.info("===== %s =====", dataset_name)
    for key in ("A_to_B", "B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_names[key],
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
) -> None:
    pretty_names = {
        "A_to_B": f"A_to_B ({model_a_id} -> {model_b_id})",
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (weighted over all datasets and both paths)",
    }

    logger.info("===== OVERALL AVERAGE ACROSS DATASETS =====")
    for key in ("A_to_B", "B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_names[key],
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            int(row["count"]),
        )


@torch.inference_mode()
def log_boolq_score_samples(
    spec: HFDatasetSpec,
    tokenizer,
    train_config,
    eval_config: EvalConfig,
    translator_pool,
    model_specs,
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
    boolq_choice_token_ids = build_boolq_choice_token_ids(tokenizer)

    logger.info("===== SAMPLE BOOLQ SCORES: %s =====", spec.name_for_log)

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
            dst_spec=model_specs["B"],
        )
        translated_b_to_a_top = translator_pool.translate_top_layers(
            past_key_values=past_b,
            src_name="B",
            dst_name="A",
            dst_spec=model_specs["A"],
        )

        mixed_past_for_b = replace_top_layers(
            base_past_key_values=past_b,
            translated_top_past_key_values=translated_a_to_b_top,
        )
        mixed_past_for_a = replace_top_layers(
            base_past_key_values=past_a,
            translated_top_past_key_values=translated_b_to_a_top,
        )

        translated_scores_b = score_boolq_choices(
            model=model_b,
            past_key_values=mixed_past_for_b,
            seed_token=seed_token,
            boolq_choice_token_ids=boolq_choice_token_ids,
        )
        translated_scores_a = score_boolq_choices(
            model=model_a,
            past_key_values=mixed_past_for_a,
            seed_token=seed_token,
            boolq_choice_token_ids=boolq_choice_token_ids,
        )

        native_scores_b = score_boolq_choices(
            model=model_b,
            past_key_values=past_b,
            seed_token=seed_token,
            boolq_choice_token_ids=boolq_choice_token_ids,
        )
        native_scores_a = score_boolq_choices(
            model=model_a,
            past_key_values=past_a,
            seed_token=seed_token,
            boolq_choice_token_ids=boolq_choice_token_ids,
        )

        translated_pred_b = predict_boolq_label(translated_scores_b)
        translated_pred_a = predict_boolq_label(translated_scores_a)
        native_pred_b = predict_boolq_label(native_scores_b)
        native_pred_a = predict_boolq_label(native_scores_a)

        logger.info(
            "--- Sample %d/%d | %s ---",
            sample_idx + 1,
            eval_config.num_generation_samples_per_dataset,
            spec.name_for_log,
        )
        logger.info("prefix(question):\n%s", prefix_text)
        logger.info("gold_answer: %s", gold_answer)
        logger.info(
            "A_to_B translated_scores: yes=%.6f | no=%.6f | pred=%s | correct=%s",
            translated_scores_b["yes"],
            translated_scores_b["no"],
            translated_pred_b,
            translated_pred_b == gold_answer,
        )
        logger.info(
            "A_to_B native_baseline_scores: yes=%.6f | no=%.6f | pred=%s | correct=%s",
            native_scores_b["yes"],
            native_scores_b["no"],
            native_pred_b,
            native_pred_b == gold_answer,
        )
        logger.info(
            "B_to_A translated_scores: yes=%.6f | no=%.6f | pred=%s | correct=%s",
            translated_scores_a["yes"],
            translated_scores_a["no"],
            translated_pred_a,
            translated_pred_a == gold_answer,
        )
        logger.info(
            "B_to_A native_baseline_scores: yes=%.6f | no=%.6f | pred=%s | correct=%s",
            native_scores_a["yes"],
            native_scores_a["no"],
            native_pred_a,
            native_pred_a == gold_answer,
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
    logger = setup_logger(log_path)
    logger.info("Starting evaluation")
    logger.info("checkpoint_path=%s", checkpoint_path)
    logger.info("eval_config=%s", asdict(eval_config))

    train_config, translator_pool, model_specs, model_a, model_b, tokenizer = load_translator_pool_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device_override=eval_config.device,
    )

    translator_pool.eval()
    model_a.eval()
    model_b.eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("model_A=%s", train_config.model_a_id)
    logger.info("model_B=%s", train_config.model_b_id)
    logger.info("top_layers_to_translate=%d", train_config.top_layers_to_translate)
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
            logger=logger,
        )
        all_results[spec.name_for_log] = results

        log_dataset_result(
            logger=logger,
            dataset_name=spec.name_for_log,
            results=results,
            model_a_id=train_config.model_a_id,
            model_b_id=train_config.model_b_id,
        )

        log_boolq_score_samples(
            spec=spec,
            tokenizer=tokenizer,
            train_config=train_config,
            eval_config=eval_config,
            translator_pool=translator_pool,
            model_specs=model_specs,
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

    overall_results = summarize_overall_results(all_results)
    log_overall_result(
        logger=logger,
        results=overall_results,
        model_a_id=train_config.model_a_id,
        model_b_id=train_config.model_b_id,
    )

    logger.info("Done. Saved log to %s", log_path)


if __name__ == "__main__":
    main()
