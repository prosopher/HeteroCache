# eval.py
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from common import (
    compute_suffix_lm_loss,
    cosine_similarity_between_past,
    decode_full_generation,
    extract_past_key_values,
    generate_from_past,
    load_translator_pool_from_checkpoint,
    replace_top_layers,
    set_seed,
    slice_top_layers,
    split_prefix_and_suffix_for_exact_next_token_loss,
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

    # if None, use values restored from checkpoint train_config
    total_tokens: Optional[int] = None
    prefix_tokens: Optional[int] = None

    # streaming / shuffling
    shuffle_eval_stream: bool = True
    shuffle_buffer: int = 10_000

    # generation sanity-check samples
    num_generation_samples_per_dataset: int = 3
    sample_generation_max_new_tokens: int = 64
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
    text_field: str = "text"
    streaming: bool = True


class HFTextChunkStream(IterableDataset):
    def __init__(
        self,
        spec: HFDatasetSpec,
        tokenizer,
        sequence_length: int,
        max_examples: int,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _load_dataset(self):
        candidates = [(self.spec.dataset_path, self.spec.dataset_name)]
        if self.spec.dataset_path == "allenai/c4":
            candidates.append(("c4", self.spec.dataset_name))

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
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must have eos_token_id set.")

        token_buffer = []
        emitted = 0

        for example in dataset:
            text = example.get(self.spec.text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue

            token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(token_ids) < 8:
                continue

            token_buffer.extend(token_ids)
            token_buffer.append(eos_token_id)

            while len(token_buffer) >= self.sequence_length:
                chunk = token_buffer[: self.sequence_length]
                token_buffer = token_buffer[self.sequence_length :]
                yield torch.tensor(chunk, dtype=torch.long)

                emitted += 1
                if emitted >= self.max_examples:
                    return


class RunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.loss_sum = 0.0
        self.count = 0

    def update(self, cosine_value: float, loss_value: float, n: int) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.loss_sum += float(loss_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "loss": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "loss": self.loss_sum / self.count,
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
    tokenizer,
    sequence_length: int,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFTextChunkStream(
        spec=spec,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_examples=eval_config.max_examples_per_dataset,
        shuffle=eval_config.shuffle_eval_stream,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    return DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
    )


def build_sample_dataloader(
    spec: HFDatasetSpec,
    tokenizer,
    sequence_length: int,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFTextChunkStream(
        spec=spec,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_examples=eval_config.num_generation_samples_per_dataset,
        shuffle=False,
        seed=eval_config.seed,
        shuffle_buffer=eval_config.shuffle_buffer,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=eval_config.num_workers,
    )


def summarize_dataset_metrics(metric: RunningAverage) -> Dict[str, Dict[str, float]]:
    result = metric.summary()
    return {
        "B_to_A": result,
        "AVG": result,
    }


def summarize_overall_results(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    cosine_sum = 0.0
    loss_sum = 0.0
    count = 0

    for dataset_result in all_results.values():
        row = dataset_result["B_to_A"]
        row_count = int(row["count"])
        cosine_sum += float(row["cosine"]) * row_count
        loss_sum += float(row["loss"]) * row_count
        count += row_count

    if count == 0:
        result = {
            "cosine": float("nan"),
            "loss": float("nan"),
            "count": 0,
        }
    else:
        result = {
            "cosine": cosine_sum / count,
            "loss": loss_sum / count,
            "count": count,
        }

    return {
        "B_to_A": result,
        "AVG": result,
    }


@torch.inference_mode()
def evaluate_dataset(
    spec: HFDatasetSpec,
    dataloader: DataLoader,
    train_config,
    translator_pool,
    model_specs,
    model_a,
    model_b,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    path_metric = RunningAverage()
    processed_examples = 0

    for batch_idx, input_ids in enumerate(dataloader, start=1):
        input_ids = input_ids.to(train_config.device)

        prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=train_config.prefix_tokens,
        )

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

        loss_b_to_a = compute_suffix_lm_loss(
            target_model=model_a,
            translated_past_key_values=mixed_past_for_a,
            lm_input_ids=lm_input_ids,
            lm_labels=lm_labels,
        ).item()

        batch_size = input_ids.size(0)
        path_metric.update(cosine_b_to_a, loss_b_to_a, batch_size)

        processed_examples += batch_size
        if batch_idx % 50 == 0:
            logger.info(
                "[%s] progress: %d/%d examples",
                spec.name_for_log,
                processed_examples,
                CONFIG.max_examples_per_dataset,
            )

    return summarize_dataset_metrics(path_metric)


def log_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    model_b_id: str,
    model_a_id: str,
) -> None:
    pretty_names = {
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (same as B_to_A in one-way evaluation)",
    }

    logger.info("===== %s =====", dataset_name)
    for key in ("B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | loss=%.6f | count=%d",
            pretty_names[key],
            row["cosine"],
            row["loss"],
            int(row["count"]),
        )


def log_overall_result(
    logger: logging.Logger,
    results: Dict[str, Dict[str, float]],
    model_b_id: str,
    model_a_id: str,
) -> None:
    pretty_names = {
        "B_to_A": f"B_to_A ({model_b_id} -> {model_a_id})",
        "AVG": "AVG (weighted over all datasets; same as B_to_A in one-way evaluation)",
    }

    logger.info("===== OVERALL AVERAGE ACROSS DATASETS =====")
    for key in ("B_to_A", "AVG"):
        row = results[key]
        logger.info(
            "%s | cosine=%.6f | loss=%.6f | count=%d",
            pretty_names[key],
            row["cosine"],
            row["loss"],
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
        tokenizer=tokenizer,
        sequence_length=train_config.total_tokens,
        eval_config=eval_config,
    )

    logger.info("===== SAMPLE GENERATIONS: %s =====", spec.name_for_log)

    sample_idx = 0
    for input_ids in sample_dataloader:
        input_ids = input_ids.to(train_config.device)

        prefix_cache_ids, lm_input_ids, lm_labels = split_prefix_and_suffix_for_exact_next_token_loss(
            input_ids=input_ids,
            prefix_tokens=train_config.prefix_tokens,
        )
        prefix_full_ids = input_ids[:, : train_config.prefix_tokens]
        reference_suffix_ids = input_ids[:, train_config.prefix_tokens :]

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

        loss_a_to_b = compute_suffix_lm_loss(
            target_model=model_b,
            translated_past_key_values=mixed_past_for_b,
            lm_input_ids=lm_input_ids,
            lm_labels=lm_labels,
        ).item()

        loss_b_to_a = compute_suffix_lm_loss(
            target_model=model_a,
            translated_past_key_values=mixed_past_for_a,
            lm_input_ids=lm_input_ids,
            lm_labels=lm_labels,
        ).item()

        max_new_tokens = min(
            eval_config.sample_generation_max_new_tokens,
            reference_suffix_ids.shape[1],
        )
        if max_new_tokens <= 0:
            continue

        seed_token = prefix_full_ids[:, -1:]

        baseline_b_generated_ids = generate_from_past(
            model=model_b,
            seed_token=seed_token,
            past_key_values=past_b,
            max_new_tokens=max_new_tokens,
            do_sample=eval_config.sample_do_sample,
            temperature=eval_config.sample_temperature,
            top_k=eval_config.sample_top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        translated_b_generated_ids = generate_from_past(
            model=model_b,
            seed_token=seed_token,
            past_key_values=mixed_past_for_b,
            max_new_tokens=max_new_tokens,
            do_sample=eval_config.sample_do_sample,
            temperature=eval_config.sample_temperature,
            top_k=eval_config.sample_top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        baseline_a_generated_ids = generate_from_past(
            model=model_a,
            seed_token=seed_token,
            past_key_values=past_a,
            max_new_tokens=max_new_tokens,
            do_sample=eval_config.sample_do_sample,
            temperature=eval_config.sample_temperature,
            top_k=eval_config.sample_top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
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

        prefix_text = tokenizer.decode(prefix_full_ids[0].detach().cpu(), skip_special_tokens=True)
        reference_suffix_text = tokenizer.decode(
            reference_suffix_ids[0, :max_new_tokens].detach().cpu(),
            skip_special_tokens=True,
        )
        reference_full_text = tokenizer.decode(
            input_ids[0, : train_config.prefix_tokens + max_new_tokens].detach().cpu(),
            skip_special_tokens=True,
        )

        baseline_b_full_text = decode_full_generation(
            tokenizer=tokenizer,
            prefix_ids=prefix_full_ids,
            generated_ids=baseline_b_generated_ids,
        )
        translated_b_full_text = decode_full_generation(
            tokenizer=tokenizer,
            prefix_ids=prefix_full_ids,
            generated_ids=translated_b_generated_ids,
        )
        baseline_a_full_text = decode_full_generation(
            tokenizer=tokenizer,
            prefix_ids=prefix_full_ids,
            generated_ids=baseline_a_generated_ids,
        )
        translated_a_full_text = decode_full_generation(
            tokenizer=tokenizer,
            prefix_ids=prefix_full_ids,
            generated_ids=translated_a_generated_ids,
        )

        logger.info("--- Sample %d/%d | %s ---", sample_idx + 1, eval_config.num_generation_samples_per_dataset, spec.name_for_log)
        logger.info("prefix_text:\n%s", prefix_text)
        logger.info("reference_suffix_text(first %d tokens):\n%s", max_new_tokens, reference_suffix_text)
        logger.info("reference_full_text(prefix + first %d suffix tokens):\n%s", max_new_tokens, reference_full_text)

        logger.info("A_to_B sample metric | cosine=%.6f | loss=%.6f", cosine_a_to_b, loss_a_to_b)
        logger.info("A_to_B target_baseline_full:\n%s", baseline_b_full_text)
        logger.info("A_to_B translated_full:\n%s", translated_b_full_text)

        logger.info("B_to_A sample metric | cosine=%.6f | loss=%.6f", cosine_b_to_a, loss_b_to_a)
        logger.info("B_to_A target_baseline_full:\n%s", baseline_a_full_text)
        logger.info("B_to_A translated_full:\n%s", translated_a_full_text)

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

    if eval_config.total_tokens is not None:
        train_config.total_tokens = eval_config.total_tokens
    if eval_config.prefix_tokens is not None:
        train_config.prefix_tokens = eval_config.prefix_tokens

    if train_config.total_tokens <= train_config.prefix_tokens:
        raise ValueError(
            f"total_tokens ({train_config.total_tokens}) must be > "
            f"prefix_tokens ({train_config.prefix_tokens})."
        )

    translator_pool.eval()
    model_a.eval()
    model_b.eval()

    logger.info("restored_train_config=%s", asdict(train_config))
    logger.info("model_A=%s", train_config.model_a_id)
    logger.info("model_B=%s", train_config.model_b_id)
    logger.info("top_layers_to_translate=%d", train_config.top_layers_to_translate)
    logger.info("total_tokens=%d | prefix_tokens=%d", train_config.total_tokens, train_config.prefix_tokens)
    logger.info("eval_log_path=%s", log_path)

    dataset_specs = [
        HFDatasetSpec(
            name_for_log="OpenWebText/train",
            dataset_path="openwebtext",
            dataset_name=None,
            split="train",
            text_field="text",
            streaming=True,
        ),
        HFDatasetSpec(
            name_for_log="WikiText-2-raw-v1/test",
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_field="text",
            streaming=True,
        ),
        HFDatasetSpec(
            name_for_log="C4/en/validation",
            dataset_path="allenai/c4",
            dataset_name="en",
            split="validation",
            text_field="text",
            streaming=True,
        ),
    ]

    all_results = {}

    for spec in dataset_specs:
        logger.info("Preparing dataloader for %s", spec.name_for_log)
        dataloader = build_eval_dataloader(
            spec=spec,
            tokenizer=tokenizer,
            sequence_length=train_config.total_tokens,
            eval_config=eval_config,
        )

        results = evaluate_dataset(
            spec=spec,
            dataloader=dataloader,
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
            model_b_id=train_config.model_b_id,
            model_a_id=train_config.model_a_id,
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
            "%s | B_to_A(cos=%.6f, loss=%.6f) | AVG(cos=%.6f, loss=%.6f)",
            dataset_name,
            result["B_to_A"]["cosine"],
            result["B_to_A"]["loss"],
            result["AVG"]["cosine"],
            result["AVG"]["loss"],
        )

    overall_results = summarize_overall_results(all_results)
    log_overall_result(
        logger=logger,
        results=overall_results,
        model_b_id=train_config.model_b_id,
        model_a_id=train_config.model_a_id,
    )

    logger.info("Done. Saved log to %s", log_path)


if __name__ == "__main__":
    main()
