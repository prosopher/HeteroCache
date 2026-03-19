from common import *



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
        split="train",
        shuffle=True,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    return InfiniteDataLoader(dataloader)




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
    nodes, edges = build_nodes_and_edges(
        config.model_ids,
        getattr(config, "model_directions", None),
    )
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
    return config_cls(**payload["train_config"])


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
