# eval.py
import logging
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from common import (
    cosine_similarity_between_past,
    extract_past_key_values,
    generate_from_past,
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

    # answer generation length control
    qa_max_new_tokens: int = 32
    qa_generation_buffer_tokens: int = 4

    # generation sanity-check samples
    num_generation_samples_per_dataset: int = 2
    sample_generation_max_new_tokens: int = 32
    sample_do_sample: bool = False
    sample_temperature: float = 1.0
    sample_top_k: int = 50

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
            qa_pair = extract_question_and_answers(self.spec, example)
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
        self.count = 0

    def update(self, cosine_value: float, accuracy_value: float, n: int) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.accuracy_sum += float(accuracy_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "accuracy": self.accuracy_sum / self.count,
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


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_question_and_answers(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, List[str]]]:
    question = example.get(spec.question_field, "")
    if not isinstance(question, str) or not question.strip():
        return None

    if spec.answer_mode == "trivia_qa":
        answer_obj = example.get("answer", {})
        if not isinstance(answer_obj, dict):
            return None
        value = answer_obj.get("value", "")
        aliases = answer_obj.get("aliases", []) or []

        answers = []
        if isinstance(value, str) and value.strip():
            answers.append(value.strip())
        answers.extend(alias.strip() for alias in aliases if isinstance(alias, str) and alias.strip())
    elif spec.answer_mode == "boolq":
        answer_value = example.get("answer", None)
        if not isinstance(answer_value, bool):
            return None
        answers = ["yes" if answer_value else "no"]
    else:
        raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")

    answers = deduplicate_preserve_order(answers)
    if not answers:
        return None

    return {
        "question": question.strip(),
        "answers": answers,
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


def get_answer_generation_max_new_tokens(tokenizer, answers: Sequence[str], eval_config: EvalConfig) -> int:
    max_ref_tokens = 1
    for answer in answers:
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        max_ref_tokens = max(max_ref_tokens, len(answer_ids))
    return max(1, min(eval_config.qa_max_new_tokens, max_ref_tokens + eval_config.qa_generation_buffer_tokens))


def decode_generated_answer(tokenizer, generated_ids: torch.Tensor) -> str:
    if generated_ids.numel() == 0:
        return ""
    return tokenizer.decode(generated_ids[0].detach().cpu(), skip_special_tokens=True).strip()


def truncate_prediction_for_match(text: str) -> str:
    truncated = text.strip()
    for delimiter in ["\n", "\r", "\t", "Question:", "question:", "Q:", "q:"]:
        if delimiter in truncated:
            truncated = truncated.split(delimiter, 1)[0].strip()
    return truncated


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_qa_text(text: str) -> str:
    text = truncate_prediction_for_match(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_correct_qa_prediction(prediction: str, references: Sequence[str]) -> bool:
    normalized_prediction = normalize_qa_text(prediction)
    if not normalized_prediction:
        return False

    for reference in references:
        normalized_reference = normalize_qa_text(reference)
        if not normalized_reference:
            continue
        if normalized_prediction == normalized_reference:
            return True
        if normalized_prediction.startswith(normalized_reference + " "):
            return True
    return False


def summarize_path_metrics(path_metrics: Dict[str, RunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}
    total_cosine_sum = 0.0
    total_accuracy_sum = 0.0
    total_count = 0

    for path_name, meter in path_metrics.items():
        path_result = meter.summary()
        results[path_name] = path_result

        total_cosine_sum += meter.cosine_sum
        total_accuracy_sum += meter.accuracy_sum
        total_count += meter.count

    if total_count == 0:
        results["AVG"] = {
            "cosine": float("nan"),
            "accuracy": float("nan"),
            "count": 0,
        }
    else:
        results["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "accuracy": total_accuracy_sum / total_count,
            "count": total_count,
        }

    return results


def summarize_overall_results(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    overall = {
        "B_to_A": {"cosine_sum": 0.0, "accuracy_sum": 0.0, "count": 0},
    }

    for dataset_result in all_results.values():
        for key in ("B_to_A",):
            row = dataset_result[key]
            count = int(row["count"])
            overall[key]["cosine_sum"] += float(row["cosine"]) * count
            overall[key]["accuracy_sum"] += float(row["accuracy"]) * count
            overall[key]["count"] += count

    summarized = {}
    total_cosine_sum = 0.0
    total_accuracy_sum = 0.0
    total_count = 0

    for key in ("B_to_A",):
        count = overall[key]["count"]
        if count == 0:
            summarized[key] = {
                "cosine": float("nan"),
                "accuracy": float("nan"),
                "count": 0,
            }
        else:
            summarized[key] = {
                "cosine": overall[key]["cosine_sum"] / count,
                "accuracy": overall[key]["accuracy_sum"] / count,
                "count": count,
            }

        total_cosine_sum += overall[key]["cosine_sum"]
        total_accuracy_sum += overall[key]["accuracy_sum"]
        total_count += count

    if total_count == 0:
        summarized["AVG"] = {
            "cosine": float("nan"),
            "accuracy": float("nan"),
            "count": 0,
        }
    else:
        summarized["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "accuracy": total_accuracy_sum / total_count,
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

    path_metrics = {
        "B_to_A": RunningAverage(),
    }

    processed_examples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        for example in batch:
            question = example["question"]
            answers = example["answers"]

            prefix = prepare_question_prefix(tokenizer=tokenizer, question=question, device=device)
            prefix_cache_ids = prefix["cache_ids"]
            seed_token = prefix["seed_token"]

            past_a = extract_past_key_values(model_a, prefix_cache_ids)
            past_b = extract_past_key_values(model_b, prefix_cache_ids)

            translated_b_to_a_top = translator_pool.translate_top_layers(
                past_key_values=past_b,
                src_name="B",
                dst_name="A",
                dst_spec=model_specs["A"],
            )

            target_top_a = slice_top_layers(
                past_key_values=past_a,
                top_layers_to_translate=train_config.top_layers_to_translate,
            )

            cosine_b_to_a = cosine_similarity_between_past(translated_b_to_a_top, target_top_a)

            mixed_past_for_a = replace_top_layers(
                base_past_key_values=past_a,
                translated_top_past_key_values=translated_b_to_a_top,
            )

            max_new_tokens = get_answer_generation_max_new_tokens(tokenizer, answers, eval_config=CONFIG)

            translated_a_generated_ids = generate_from_past(
                model=model_a,
                seed_token=seed_token,
                past_key_values=mixed_past_for_a,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_k=1,
                eos_token_id=tokenizer.eos_token_id,
            )

            translated_a_answer = decode_generated_answer(tokenizer, translated_a_generated_ids)

            acc_b_to_a = 1.0 if is_correct_qa_prediction(translated_a_answer, answers) else 0.0

            path_metrics["B_to_A"].update(cosine_b_to_a, acc_b_to_a, 1)

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
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (weighted over both paths)",
    }

    logger.info("===== %s =====", dataset_name)
    for key in ("B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | count=%d",
            pretty_names[key],
            row["cosine"],
            row["accuracy"],
            int(row["count"]),
        )


def log_overall_result(
    logger: logging.Logger,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
) -> None:
    pretty_names = {
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (weighted over all datasets and both paths)",
    }

    logger.info("===== OVERALL AVERAGE ACROSS DATASETS =====")
    for key in ("B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | count=%d",
            pretty_names[key],
            row["cosine"],
            row["accuracy"],
            int(row["count"]),
        )


@torch.inference_mode()
def log_generation_samples(
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

    logger.info("===== SAMPLE GENERATIONS: %s =====", spec.name_for_log)

    sample_idx = 0
    for batch in sample_dataloader:
        example = batch[0]
        question = example["question"]
        answers = example["answers"]

        prefix = prepare_question_prefix(tokenizer=tokenizer, question=question, device=train_config.device)
        prefix_cache_ids = prefix["cache_ids"]
        seed_token = prefix["seed_token"]
        prefix_text = prefix["prefix_text"]

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

        max_new_tokens = min(
            eval_config.sample_generation_max_new_tokens,
            get_answer_generation_max_new_tokens(tokenizer, answers, eval_config),
        )
        if max_new_tokens <= 0:
            continue

        translated_a_generated_ids = generate_from_past(
            model=model_a,
            seed_token=seed_token,
            past_key_values=mixed_past_for_a,
            max_new_tokens=max_new_tokens,
            do_sample=eval_config.sample_do_sample,
            temperature=eval_config.sample_temperature,
            top_k=eval_config.sample_top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

        translated_a_answer = decode_generated_answer(tokenizer, translated_a_generated_ids)

        logger.info(
            "--- Sample %d/%d | %s ---",
            sample_idx + 1,
            eval_config.num_generation_samples_per_dataset,
            spec.name_for_log,
        )
        logger.info("prefix(question):\n%s", prefix_text)
        logger.info("reference_answers:\n%s", " | ".join(answers))
        logger.info(
            "B_to_A generated_answer:\n%s\ncorrect=%s",
            translated_a_answer,
            is_correct_qa_prediction(translated_a_answer, answers),
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
            name_for_log="TriviaQA/rc.nocontext/validation",
            dataset_path="trivia_qa",
            dataset_name="rc.nocontext",
            split="validation",
            answer_mode="trivia_qa",
            question_field="question",
            streaming=False,
        ),
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

        log_generation_samples(
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
            "%s | B_to_A(cos=%.6f, acc=%.6f) | AVG(cos=%.6f, acc=%.6f)",
            dataset_name,
            result["B_to_A"]["cosine"],
            result["B_to_A"]["accuracy"],
            result["AVG"]["cosine"],
            result["AVG"]["accuracy"],
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
