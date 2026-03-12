from common import *


class OpenWebTextSequenceStream(IterableDataset):
    """
    Streams OpenWebText and yields fixed-length token chunks.

    This behaves like a rolling token buffer, which is usually a better fit for
    language-model experiments than per-document truncation.
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
            token_ids = self.tokenizer(text, add_special_tokens=False, verbose=False).input_ids
            if len(token_ids) < 8:
                continue
            token_buffer.extend(token_ids)
            token_buffer.append(eos_token_id)
            while len(token_buffer) >= self.sequence_length:
                chunk = token_buffer[: self.sequence_length]
                token_buffer = token_buffer[self.sequence_length :]
                yield torch.tensor(chunk, dtype=torch.long)


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


def build_models_and_tokenizer(config) -> Tuple[Dict[str, PreTrainedModel], PreTrainedTokenizerBase, List[Node], List[Edge]]:
    nodes, edges = build_nodes_and_edges(config.model_ids, get_model_directions_value(config))
    tokenizer = load_tokenizer(nodes[0].model_id)
    models = {
        node.id: load_frozen_model(node.model_id, device=config.device, dtype=config.dtype)
        for node in nodes
    }
    return models, tokenizer, nodes, edges


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
    return config_cls(**normalize_train_config_dict(payload["train_config"]))


def get_train_config_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "train_config.json"


def get_train_log_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "train.log"


def get_train_checkpoint_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "final_checkpoint_path.pt"


def initialize_train_output_paths(config) -> None:
    alg = getattr(config, "alg", "")
    output_path = getattr(config, "output_path", None)
    outputs_path = getattr(config, "outputs_path", "outputs")
    timestamp = getattr(config, "timestamp", None)

    if output_path is None:
        if not alg:
            return
        if timestamp is None:
            timestamp = build_timestamp_string()
            setattr(config, "timestamp", timestamp)
        output_path_obj = build_timestamped_output_path(
            alg=alg,
            outputs_path=outputs_path,
            timestamp=timestamp,
        )
    else:
        output_path_obj = Path(output_path)

    setattr(config, "output_path", str(output_path_obj))
