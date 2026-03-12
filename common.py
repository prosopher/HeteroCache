import argparse
import json
import logging
import math
import random
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


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

    stream_handler = TqdmLoggingHandler()
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


@dataclass(frozen=True)
class Node:
    id: str
    model_id: str


@dataclass(frozen=True)
class Edge:
    id: str
    src_id: str
    dst_id: str


def index_to_node_id(index: int) -> str:
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")

    chars = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        chars.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(chars))


def parse_model_ids_csv(model_ids: str) -> List[str]:
    parsed = [item.strip() for item in str(model_ids).split(",") if item.strip()]
    if len(parsed) < 2:
        raise ValueError("model_ids must contain at least two comma-separated model ids.")

    duplicates = [item for item, count in Counter(parsed).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate model_ids are not allowed: {sorted(duplicates)}")
    return parsed


def build_nodes_from_model_ids(model_ids: str) -> List[Node]:
    return [
        Node(id=index_to_node_id(index), model_id=model_id)
        for index, model_id in enumerate(parse_model_ids_csv(model_ids))
    ]


def build_allowed_edge_ids(nodes: List[Node]) -> List[str]:
    edge_ids = []
    for src_node in nodes:
        for dst_node in nodes:
            if src_node.id == dst_node.id:
                continue
            edge_ids.append(f"{src_node.id}_to_{dst_node.id}")
    return edge_ids


def parse_model_directions(model_directions: str, allowed_directions: Optional[Iterable[str]] = None) -> List[str]:
    parsed = [item.strip() for item in str(model_directions).split(",") if item.strip()]
    if not parsed:
        raise ValueError("model_directions must contain at least one direction.")

    allowed_set = None
    if allowed_directions is not None:
        allowed_set = set(allowed_directions)
        if not allowed_set:
            raise ValueError("allowed_directions must not be empty when provided.")

    deduped = []
    seen = set()
    for item in parsed:
        if allowed_set is not None and item not in allowed_set:
            raise ValueError(
                f"Unsupported model direction: {item}. "
                f"Allowed values are: {sorted(allowed_set)}"
            )
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_node_map(nodes: Iterable[Node]) -> Dict[str, Node]:
    return {node.id: node for node in nodes}


def build_edge_map(edges: Iterable[Edge]) -> Dict[str, Edge]:
    return {edge.id: edge for edge in edges}


def build_edges_from_nodes(nodes: List[Node], model_directions: str) -> List[Edge]:
    node_map = build_node_map(nodes)
    direction_ids = parse_model_directions(
        model_directions,
        allowed_directions=build_allowed_edge_ids(nodes),
    )

    edges = []
    for direction_id in direction_ids:
        match = re.fullmatch(r"([A-Z]+)_to_([A-Z]+)", direction_id)
        if match is None:
            raise ValueError(
                f"Invalid model direction format: {direction_id}. Expected format is <SRC>_to_<DST>."
            )
        src_id, dst_id = match.groups()
        if src_id == dst_id:
            raise ValueError(f"Self-direction is not allowed: {direction_id}")
        if src_id not in node_map or dst_id not in node_map:
            raise ValueError(
                f"Unknown node in model direction: {direction_id}. Available node ids are: {sorted(node_map)}"
            )
        edges.append(
            Edge(
                id=direction_id,
                src_id=src_id,
                dst_id=dst_id,
            )
        )
    return edges


def build_nodes_and_edges(model_ids: str, model_directions: str) -> Tuple[List[Node], List[Edge]]:
    nodes = build_nodes_from_model_ids(model_ids)
    edges = build_edges_from_nodes(nodes, model_directions)
    return nodes, edges


def get_model_directions_value(config) -> str:
    model_directions = getattr(config, "model_directions", None)
    if model_directions is not None:
        return model_directions

    legacy_train_directions = getattr(config, "train_directions", None)
    if legacy_train_directions is not None:
        return legacy_train_directions

    raise AttributeError("config must define model_directions")


def normalize_train_config_dict(train_config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(train_config)
    if "model_directions" not in normalized and "train_directions" in normalized:
        normalized["model_directions"] = normalized.pop("train_directions")
    return normalized


def parse_train_directions(train_directions: str, allowed_directions: Optional[Iterable[str]] = None) -> List[str]:
    return parse_model_directions(train_directions, allowed_directions=allowed_directions)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
        raise ValueError("This example expects GPT-2 style configs with n_head/n_embd/n_layer.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    return ModelSpec(
        model_id=getattr(config, "_name_or_path", "unknown"),
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )


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


def format_memory_gib(num_bytes: float) -> str:
    gib = float(num_bytes) / (1024 ** 3)
    return f"{gib:.2f} GiB"


class GPUMemoryTracker:
    def __init__(self, device: str) -> None:
        self.device = device
        self.enabled = torch.cuda.is_available() and str(device).startswith("cuda")
        self.total_allocated_bytes = 0.0
        self.num_samples = 0
        self.peak_allocated_bytes = 0

        if self.enabled:
            self.device_index = torch.device(device).index
            if self.device_index is None:
                self.device_index = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(self.device_index)
        else:
            self.device_index = None

    def update(self) -> None:
        if not self.enabled:
            return

        allocated = torch.cuda.memory_allocated(self.device_index)
        peak = torch.cuda.max_memory_allocated(self.device_index)

        self.total_allocated_bytes += float(allocated)
        self.num_samples += 1
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, int(peak))

    @property
    def avg_allocated_bytes(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.total_allocated_bytes / self.num_samples

    def summary(self) -> Dict[str, object]:
        if not self.enabled:
            return {
                "enabled": False,
                "avg_allocated_bytes": None,
                "peak_allocated_bytes": None,
                "avg_allocated_pretty": "N/A",
                "peak_allocated_pretty": "N/A",
                "num_samples": 0,
            }

        return {
            "enabled": True,
            "avg_allocated_bytes": self.avg_allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "avg_allocated_pretty": format_memory_gib(self.avg_allocated_bytes),
            "peak_allocated_pretty": format_memory_gib(self.peak_allocated_bytes),
            "num_samples": self.num_samples,
        }


def move_past_to_device(past_key_values: PastKeyValues, device: str) -> PastKeyValues:
    moved = []
    for key, value in past_key_values:
        moved.append((key.to(device), value.to(device)))
    return tuple(moved)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON config at {path_obj} must contain a top-level object.")
    return payload


def write_json(path: str, payload: Dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def build_timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_timestamped_output_path(
    alg: str,
    outputs_path: str = "outputs",
    timestamp: Optional[str] = None,
) -> Path:
    run_timestamp = timestamp or build_timestamp_string()
    return Path(outputs_path) / f"{alg}_{run_timestamp}"


T = TypeVar("T")


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _unwrap_optional_type(annotation):
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _resolve_argparse_type(annotation):
    annotation = _unwrap_optional_type(annotation)
    if annotation is bool:
        return parse_bool_arg
    if annotation in {str, int, float}:
        return annotation
    return str


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    config_cls: Type[T],
    exclude_fields: Optional[set[str]] = None,
) -> None:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()

    for field_info in fields(config_cls):
        if field_info.name in excluded:
            continue

        option_name = f"--{field_info.name.replace('_', '-')}"
        parser.add_argument(
            option_name,
            dest=field_info.name,
            type=_resolve_argparse_type(field_info.type),
            default=argparse.SUPPRESS,
        )


def extract_dataclass_kwargs_from_namespace(
    config_cls: Type[T],
    args: argparse.Namespace,
    exclude_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()
    kwargs: Dict[str, Any] = {}

    for field_info in fields(config_cls):
        if field_info.name in excluded:
            continue
        if hasattr(args, field_info.name):
            kwargs[field_info.name] = getattr(args, field_info.name)

    return kwargs


def build_dataclass_kwargs_from_json_and_namespace(
    config_cls: Type[T],
    default_config_path: Union[str, Path],
    args: argparse.Namespace,
    exclude_fields: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls} must be a dataclass type.")

    excluded = exclude_fields or set()
    default_kwargs = read_json(default_config_path)

    valid_field_names = {field_info.name for field_info in fields(config_cls)}
    if "model_directions" in valid_field_names and "model_directions" not in default_kwargs and "train_directions" in default_kwargs:
        default_kwargs["model_directions"] = default_kwargs.pop("train_directions")
    unknown_keys = sorted(set(default_kwargs) - valid_field_names)
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in {default_config_path}: {unknown_keys}"
        )

    merged_kwargs = {
        key: value
        for key, value in default_kwargs.items()
        if key not in excluded
    }
    merged_kwargs.update(
        extract_dataclass_kwargs_from_namespace(
            config_cls=config_cls,
            args=args,
            exclude_fields=exclude_fields,
        )
    )

    missing_keys = [
        field_info.name
        for field_info in fields(config_cls)
        if field_info.name not in excluded and field_info.name not in merged_kwargs
    ]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys in {default_config_path}: {missing_keys}"
        )

    return merged_kwargs
