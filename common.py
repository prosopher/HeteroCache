import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
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


PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def parse_train_directions(train_directions: str) -> List[str]:
    allowed = {"A_to_B", "B_to_A"}
    parsed = [item.strip() for item in train_directions.split(",") if item.strip()]
    if not parsed:
        raise ValueError("train_directions must contain at least one direction.")

    deduped = []
    seen = set()
    for item in parsed:
        if item not in allowed:
            raise ValueError(
                f"Unsupported train direction: {item}. "
                f"Allowed values are: {sorted(allowed)}"
            )
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def freeze_model(model: PreTrainedModel) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def load_frozen_model(model_id: str, device: str, dtype: str = "float32") -> PreTrainedModel:
    torch_dtype = get_torch_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    freeze_model(model)
    return model


def get_model_spec(model: PreTrainedModel) -> ModelSpec:
    config = model.config
    num_heads = getattr(config, "n_head", None)
    hidden_size = getattr(config, "n_embd", None)
    num_layers = getattr(config, "n_layer", None)
    if num_heads is None or hidden_size is None or num_layers is None:
        raise ValueError("This toy example expects GPT-2 style configs with n_head/n_embd/n_layer.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    return ModelSpec(
        model_id=getattr(config, "_name_or_path", "unknown"),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )


class OpenWebTextSequenceStream(IterableDataset):
    """
    Streams OpenWebText and yields fixed-length token chunks.

    This behaves like a rolling token buffer, which is usually a better fit for
    language-model style toy experiments than per-document truncation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self) -> Iterable[torch.Tensor]:
        stream = load_dataset("openwebtext", split="train", streaming=True)
        stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        token_buffer: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for example in stream:
            text = example.get("text", "")
            if not text or text.isspace():
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


@torch.no_grad()
def extract_past_key_values(model: PreTrainedModel, input_ids: torch.Tensor) -> PastKeyValues:
    outputs = model(input_ids=input_ids, use_cache=True)
    return outputs.past_key_values


def past_key_values_to_blocks(past_key_values: PastKeyValues) -> Tuple[torch.Tensor, torch.Tensor]:
    key_layers = []
    value_layers = []
    for key, value in past_key_values:
        batch_size, num_heads, seq_len, head_dim = key.shape
        key_flat = key.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        value_flat = value.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)
        key_layers.append(key_flat)
        value_layers.append(value_flat)
    key_block = torch.stack(key_layers, dim=2)
    value_block = torch.stack(value_layers, dim=2)
    return key_block, value_block


def slice_top_layers(
    past_key_values: PastKeyValues,
    top_layers_to_translate: int,
) -> PastKeyValues:
    if top_layers_to_translate < 1:
        raise ValueError("top_layers_to_translate must be >= 1")
    if top_layers_to_translate > len(past_key_values):
        raise ValueError(
            f"Cannot slice {top_layers_to_translate} layers from cache with only {len(past_key_values)} layers."
        )
    return tuple(past_key_values[-top_layers_to_translate:])


def replace_top_layers(
    base_past_key_values: PastKeyValues,
    translated_top_past_key_values: PastKeyValues,
) -> PastKeyValues:
    num_replace = len(translated_top_past_key_values)
    if num_replace < 1:
        raise ValueError("translated_top_past_key_values must contain at least one layer.")
    if num_replace > len(base_past_key_values):
        raise ValueError(
            f"Cannot replace {num_replace} layers in cache with only {len(base_past_key_values)} layers."
        )

    base_list = list(base_past_key_values)
    start_idx = len(base_list) - num_replace

    for offset, translated_layer in enumerate(translated_top_past_key_values):
        base_key, base_value = base_list[start_idx + offset]
        translated_key, translated_value = translated_layer

        if base_key.shape != translated_key.shape:
            raise ValueError(
                f"Key shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_key.shape)} vs translated={tuple(translated_key.shape)}"
            )
        if base_value.shape != translated_value.shape:
            raise ValueError(
                f"Value shape mismatch at replaced layer {offset}: "
                f"base={tuple(base_value.shape)} vs translated={tuple(translated_value.shape)}"
            )

        base_list[start_idx + offset] = (translated_key, translated_value)

    return tuple(base_list)


def flatten_past_key_values(past_key_values: PastKeyValues) -> torch.Tensor:
    flat_parts = []
    for key, value in past_key_values:
        flat_parts.append(key.reshape(key.shape[0], -1))
        flat_parts.append(value.reshape(value.shape[0], -1))
    return torch.cat(flat_parts, dim=1)


def cosine_similarity_between_past(a: PastKeyValues, b: PastKeyValues) -> float:
    flat_a = flatten_past_key_values(a)
    flat_b = flatten_past_key_values(b)
    return F.cosine_similarity(flat_a, flat_b, dim=1).mean().item()


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def move_past_to_device(past_key_values: PastKeyValues, device: str) -> PastKeyValues:
    moved = []
    for key, value in past_key_values:
        moved.append((key.to(device), value.to(device)))
    return tuple(moved)


def compute_suffix_lm_loss(
    target_model: PreTrainedModel,
    past_key_values: PastKeyValues,
    lm_input_ids: torch.Tensor,
    lm_labels: torch.Tensor,
) -> torch.Tensor:
    outputs = target_model(
        input_ids=lm_input_ids,
        past_key_values=past_key_values,
        use_cache=False,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        lm_labels.reshape(-1),
        reduction="mean",
    )


class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def build_training_dataloader(tokenizer: PreTrainedTokenizerBase, config) -> InfiniteDataLoader:
    dataset = OpenWebTextSequenceStream(
        tokenizer=tokenizer,
        sequence_length=config.total_tokens,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    return InfiniteDataLoader(dataloader)


def split_prefix_and_suffix_for_exact_next_token_loss(
    input_ids: torch.Tensor,
    prefix_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prefix_tokens < 2:
        raise ValueError("prefix_tokens must be >= 2")
    prefix_cache_ids = input_ids[:, : prefix_tokens - 1]
    lm_input_ids = input_ids[:, prefix_tokens - 1 : -1]
    lm_labels = input_ids[:, prefix_tokens:]
    return prefix_cache_ids, lm_input_ids, lm_labels


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_id = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_id += 1
        if self.step_id <= self.warmup_steps:
            multiplier = self.step_id / self.warmup_steps
        else:
            progress = (self.step_id - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def build_models_and_tokenizer(config) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(config.model_a_id)
    model_a = load_frozen_model(config.model_a_id, device=config.device, dtype=config.dtype)
    model_b = load_frozen_model(config.model_b_id, device=config.device, dtype=config.dtype)
    return model_a, model_b, tokenizer


def save_checkpoint(
    output_path: str,
    translator_pool: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    train_config,
    step: int,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "translator_pool": translator_pool.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_config": asdict(train_config),
        "scheduler_step": scheduler.step_id,
    }
    if extra is not None:
        payload["extra"] = extra
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_train_config_from_checkpoint(checkpoint_path: str, config_cls):
    payload = torch.load(checkpoint_path, map_location="cpu")
    return config_cls(**payload["train_config"])


def write_json(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def build_timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_timestamped_output_dir(
    alg: str,
    output_root: str = "outputs",
    timestamp: Optional[str] = None,
) -> Path:
    run_timestamp = timestamp or build_timestamp_string()
    return Path(output_root) / f"{alg}_{run_timestamp}"


def resolve_latest_checkpoint_for_alg(
    alg: str,
    output_root: str = "outputs",
    checkpoint_name: str = "final_checkpoint.pt",
) -> Path:
    output_root_path = Path(output_root)
    candidates = sorted(
        path
        for path in output_root_path.glob(f"{alg}_*")
        if path.is_dir() and (path / checkpoint_name).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directories found for alg={alg!r} under {output_root_path}"
        )
    return candidates[-1] / checkpoint_name


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


def summarize_overall_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    active_directions,
) -> Dict[str, Dict[str, float]]:
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
