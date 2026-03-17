from common import *


@dataclass
class EvalConfig:
    alg: str
    outputs_path: str
    timestamp: Optional[str]
    output_path: Optional[str]
    checkpoint_path: Optional[str]
    device: str

    # evaluation sampling
    batch_size: int
    num_workers: int
    max_examples_per_dataset: int
    seed: int

    # streaming / shuffling
    shuffle_eval_stream: bool
    shuffle_buffer: int

    # generation QA
    generation_max_new_tokens: int
    enable_generation_eval: bool = True


    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        initialize_eval_output_paths(self)


@dataclass
class HFDatasetSpec:
    name_for_log: str
    dataset_path: str
    dataset_name: Optional[str]
    split: str
    answer_mode: str
    question_field: str = "question"
    context_field: Optional[str] = None
    answers_field: Optional[str] = None
    subject_field: Optional[str] = None
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
        if self.spec.dataset_name is None:
            return load_dataset(
                self.spec.dataset_path,
                split=self.spec.split,
                streaming=self.spec.streaming,
            )
        return load_dataset(
            self.spec.dataset_path,
            self.spec.dataset_name,
            split=self.spec.split,
            streaming=self.spec.streaming,
        )

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


DEFAULT_MULTINEWS_SUMMARY_TASK = "Summarize the news articles above."


def get_squad_v11_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="SQuAD-v1.1/validation",
        dataset_path="rajpurkar/squad",
        dataset_name=None,
        split="validation",
        answer_mode="squad",
        question_field="question",
        context_field="context",
        answers_field="answers",
        streaming=False,
    )


def get_multinews_generation_dataset_spec() -> HFDatasetSpec:
    return HFDatasetSpec(
        name_for_log="MultiNews/validation",
        dataset_path="Awesome075/multi_news_parquet",
        dataset_name=None,
        split="validation",
        answer_mode="multinews",
        context_field="document",
        answers_field="summary",
        streaming=False,
    )


def _normalize_answer_texts(raw_value: Any) -> List[str]:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return [text] if text else []

    if isinstance(raw_value, list):
        return [
            item.strip()
            for item in raw_value
            if isinstance(item, str) and item.strip()
        ]

    if isinstance(raw_value, dict):
        raw_texts = raw_value.get("text", [])
        if isinstance(raw_texts, list):
            return [
                item.strip()
                for item in raw_texts
                if isinstance(item, str) and item.strip()
            ]

    return []


def normalize_multinews_context_text(raw_value: Any) -> Optional[str]:
    context = normalize_context_text(raw_value)
    if context is None:
        return None
    normalized = context.replace(" ||||| ", "\n\n").replace("|||||", "\n\n").strip()
    return normalized or None


def extract_generation_example(spec: HFDatasetSpec, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if spec.answer_mode == "squad":
        question = example.get(spec.question_field, "")
        if not isinstance(question, str) or not question.strip():
            return None

        context_field = spec.context_field or "context"
        answers_field = spec.answers_field or "answers"

        context = example.get(context_field, "")
        if not isinstance(context, str) or not context.strip():
            return None

        answer_texts = _normalize_answer_texts(example.get(answers_field, None))
        if not answer_texts:
            return None

        return {
            "question": question.strip(),
            "context": context.strip(),
            "answers": answer_texts,
        }

    if spec.answer_mode == "multinews":
        question_value = example.get(spec.question_field, None)
        if isinstance(question_value, str) and question_value.strip():
            question = question_value.strip()
        else:
            question = DEFAULT_MULTINEWS_SUMMARY_TASK

        context_field = spec.context_field or "document"
        answers_field = spec.answers_field or "summary"

        context = normalize_multinews_context_text(example.get(context_field, None))
        if context is None:
            return None

        answer_texts = _normalize_answer_texts(example.get(answers_field, None))
        if not answer_texts:
            return None

        return {
            "question": question,
            "context": context,
            "answers": answer_texts,
        }

    raise ValueError(f"Unsupported generation answer_mode: {spec.answer_mode}")


class HFGenerationExampleStream(IterableDataset):
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
        if self.spec.dataset_name is None:
            return load_dataset(
                self.spec.dataset_path,
                split=self.spec.split,
                streaming=self.spec.streaming,
            )
        return load_dataset(
            self.spec.dataset_path,
            self.spec.dataset_name,
            split=self.spec.split,
            streaming=self.spec.streaming,
        )

    def __iter__(self):
        dataset = self._load_dataset()
        if self.shuffle:
            if self.spec.streaming:
                dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            else:
                dataset = dataset.shuffle(seed=self.seed)

        emitted = 0

        for example in dataset:
            generation_example = extract_generation_example(self.spec, example)
            if generation_example is None:
                continue

            yield generation_example
            emitted += 1
            if emitted >= self.max_examples:
                return


def build_generation_eval_dataloader(
    spec: HFDatasetSpec,
    eval_config: EvalConfig,
) -> DataLoader:
    dataset = HFGenerationExampleStream(
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


class GenerationRunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.f1_sum = 0.0
        self.native_f1_sum = 0.0
        self.count = 0

    def update(
        self,
        cosine_value: float,
        f1_value: float,
        native_f1_value: float,
        n: int,
    ) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.f1_sum += float(f1_value) * n
        self.native_f1_sum += float(native_f1_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "f1": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "f1": self.f1_sum / self.count,
            "native_f1": self.native_f1_sum / self.count,
            "count": self.count,
        }


def get_eval_config_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "eval_config.json"


def get_eval_log_path(output_path: Union[str, Path]) -> Path:
    return Path(output_path) / "eval.log"


def initialize_eval_output_paths(config) -> None:
    output_path = getattr(config, "output_path", None)
    checkpoint_path = getattr(config, "checkpoint_path", None)

    if output_path is None:
        if checkpoint_path is not None:
            output_path_obj = Path(checkpoint_path).parent
        else:
            alg = getattr(config, "alg", "")
            if not alg:
                return
            outputs_path = getattr(config, "outputs_path", "outputs")
            timestamp = getattr(config, "timestamp", None)
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


def resolve_latest_checkpoint_for_alg(
    alg: str,
    outputs_path: str = "outputs",
    checkpoint_name: str = "final_checkpoint_path.pt",
) -> Path:
    outputs_path_obj = Path(outputs_path)
    candidates = sorted(
        path
        for path in outputs_path_obj.glob(f"{alg}_*")
        if path.is_dir() and (path / checkpoint_name).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directories found for alg={alg!r} under {outputs_path_obj}"
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


def normalize_context_text(raw_value: Any) -> Optional[str]:
    def _collect_strings(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                parts.extend(_collect_strings(item))
            return parts

        if isinstance(value, dict):
            for key in ("contexts", "context", "text", "abstract", "passage", "sentences"):
                if key in value:
                    parts = _collect_strings(value[key])
                    if parts:
                        return parts
            return []

        return []

    parts = _collect_strings(raw_value)
    if not parts:
        return None
    return "\n".join(parts)


def extract_question_and_answer(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, Any]]:
    question = example.get(spec.question_field, "")
    if not isinstance(question, str) or not question.strip():
        return None

    if spec.answer_mode == "boolq":
        context_field = spec.context_field or "passage"
        context = normalize_context_text(example.get(context_field, None))
        if context is None:
            return None

        answer_value = example.get("answer", None)
        if not isinstance(answer_value, bool):
            return None

        return {
            "question": question.strip(),
            "context": context,
            "answer": "yes" if answer_value else "no",
        }

    if spec.answer_mode == "pubmed_qa":
        context_field = spec.context_field or "context"
        context = normalize_context_text(example.get(context_field, None))
        if context is None:
            return None

        answer_value = example.get("final_decision", None)
        if not isinstance(answer_value, str):
            return None

        normalized_answer = answer_value.strip().lower()
        if normalized_answer not in {"yes", "no", "maybe"}:
            return None

        return {
            "question": question.strip(),
            "context": context,
            "answer": normalized_answer,
        }

    if spec.answer_mode == "squad":
        context_field = spec.context_field or "context"
        answers_field = spec.answers_field or "answers"

        context = example.get(context_field, "")
        answers = example.get(answers_field, None)

        if not isinstance(context, str) or not context.strip():
            return None

        answer_texts: List[str] = []
        if isinstance(answers, dict):
            raw_texts = answers.get("text", [])
            if isinstance(raw_texts, list):
                answer_texts = [
                    item.strip()
                    for item in raw_texts
                    if isinstance(item, str) and item.strip()
                ]
        elif isinstance(answers, list):
            answer_texts = [
                item.strip()
                for item in answers
                if isinstance(item, str) and item.strip()
            ]

        if not answer_texts:
            return None

        return {
            "question": question.strip(),
            "context": context.strip(),
            "answers": answer_texts,
        }

    raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")


def format_question_prefix(
    question: str,
    choices: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
) -> str:
    question = question.strip()

    if answer_mode == "boolq":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("BoolQ requires passage context.")
        return (
            "Read the passage and answer the question with yes or no only.\n\n"
            f"Passage: {context.strip()}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if answer_mode == "pubmed_qa":
        if not isinstance(context, str) or not context.strip():
            raise ValueError("PubMedQA requires abstract context.")
        return (
            "Read the abstract and answer the biomedical research question with yes, no, or maybe only.\n\n"
            f"Abstract: {context.strip()}\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if not choices:
        return f"Question: {question}\nAnswer:"

    prompt_lines = [f"Question: {question}", "Choices:"]
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        prompt_lines.append(f"{label}. {choice.strip()}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


def format_squad_v11_context_prefix(context: str) -> str:
    return (
        "Read the passage and answer the question briefly. Use a short phrase from the passage when possible.\n\n"
        f"Passage: {context.strip()}\n"
    )


def format_squad_v11_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_squad_v11_context_inputs(tokenizer, context: str, device: str) -> Dict[str, Any]:
    prefix_text = format_squad_v11_context_prefix(context=context)
    return prepare_full_text_inputs(tokenizer=tokenizer, text=prefix_text, device=device)


def prepare_squad_v11_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_squad_v11_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_multinews_context_prefix(context: str) -> str:
    return (
        "Read the following news articles and write a concise summary.\n\n"
        f"Articles:\n{context.strip()}\n"
    )


def format_multinews_question_prefix(question: str) -> str:
    return (
        f"Task: {question.strip()}\n"
        "Summary:"
    )


def prepare_multinews_context_inputs(
    tokenizer,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_multinews_context_prefix(context=context)
    return prepare_full_text_inputs(
        tokenizer=tokenizer,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_multinews_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_multinews_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_generation_prompt(context: str, question: str) -> str:
    return (
        "Read the passage and answer the question briefly.\n\n"
        f"Context: {context.strip()}\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_text_prefix(tokenizer, prefix_text: str, device: str) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(prefix_text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("Prefix must tokenize to at least 2 tokens.")
    cache_ids = input_ids[:, :-1]
    seed_token = input_ids[:, -1:]
    return {
        "prefix_text": prefix_text,
        "full_prefix_ids": input_ids,
        "cache_ids": cache_ids,
        "seed_token": seed_token,
    }


def prepare_question_prefix(
    tokenizer,
    question: str,
    device: str,
    choices: Optional[List[str]] = None,
    subject: Optional[str] = None,
    context: Optional[str] = None,
    answer_mode: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    prefix_text = format_question_prefix(
        question,
        choices=choices,
        subject=subject,
        context=context,
        answer_mode=answer_mode,
    )
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def prepare_generation_prefix(tokenizer, context: str, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_prompt(context=context, question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_generation_context_prefix(context: str) -> str:
    return (
        "Read the passage and answer the question briefly.\n\n"
        f"Context: {context.strip()}\n"
    )


def format_generation_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def prepare_full_text_inputs(
    tokenizer,
    text: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_input_tokens is not None:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = int(max_input_tokens)
    tokenized = tokenizer(text, **tokenizer_kwargs)
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 1:
        raise ValueError("Text must tokenize to at least 1 token.")
    return {
        "text": text,
        "input_ids": input_ids,
        "was_truncated": bool(max_input_tokens is not None and input_ids.shape[1] >= int(max_input_tokens)),
    }


def prepare_generation_context_inputs(
    tokenizer,
    context: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    prefix_text = format_generation_context_prefix(context=context)
    return prepare_full_text_inputs(
        tokenizer=tokenizer,
        text=prefix_text,
        device=device,
        max_input_tokens=max_input_tokens,
    )


def prepare_generation_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def get_model_context_limit(model: PreTrainedModel, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "n_positions", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_ctx", None),
    ]
    if tokenizer is not None:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            candidates.append(tokenizer_limit)

    limits = [int(value) for value in candidates if isinstance(value, int) and value > 0]
    if not limits:
        return 1024
    return min(limits)


def get_answer_token_budget_for_spec(spec: HFDatasetSpec, eval_config) -> int:
    return int(getattr(eval_config, "generation_max_new_tokens"))


def compute_benchmark_context_budget(
    tokenizer: PreTrainedTokenizerBase,
    spec: HFDatasetSpec,
    question: str,
    eval_config,
    models: Dict[str, PreTrainedModel],
) -> int:
    shared_limit = min(get_model_context_limit(model, tokenizer) for model in models.values())
    question_prefix = prepare_generation_task_question_prefix(
        spec=spec,
        tokenizer=tokenizer,
        question=question,
        device="cpu",
    )
    reserved_tokens = (
        int(question_prefix["cache_ids"].shape[1])
        + int(question_prefix["seed_token"].shape[1])
        + get_answer_token_budget_for_spec(spec, eval_config)
    )
    budget = shared_limit - reserved_tokens
    if budget < 16:
        raise ValueError(
            f"Insufficient context budget for {spec.name_for_log}: "
            f"shared_limit={shared_limit}, reserved_tokens={reserved_tokens}"
        )
    return budget


def prepare_generation_task_question_prefix(
    spec: HFDatasetSpec,
    tokenizer,
    question: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode == "squad":
        return prepare_squad_v11_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
    if spec.answer_mode == "multinews":
        return prepare_multinews_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
    return prepare_generation_question_prefix(
        tokenizer=tokenizer,
        question=question,
        device=device,
    )


def prepare_generation_task_inputs(
    spec: HFDatasetSpec,
    tokenizer,
    context: str,
    question: str,
    device: str,
    max_input_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    if spec.answer_mode == "squad":
        context_prefix = prepare_squad_v11_context_inputs(
            tokenizer=tokenizer,
            context=context,
            device=device,
        )
        question_prefix = prepare_squad_v11_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
        return {
            "context_prefix": context_prefix,
            "question_prefix": question_prefix,
            "cache_input_ids": context_prefix["input_ids"],
            "question_cache_ids": question_prefix["cache_ids"],
            "seed_token": question_prefix["seed_token"],
            "was_truncated": False,
        }

    if spec.answer_mode == "multinews":
        context_prefix = prepare_multinews_context_inputs(
            tokenizer=tokenizer,
            context=context,
            device=device,
            max_input_tokens=max_input_tokens,
        )
        question_prefix = prepare_multinews_question_prefix(
            tokenizer=tokenizer,
            question=question,
            device=device,
        )
        return {
            "context_prefix": context_prefix,
            "question_prefix": question_prefix,
            "cache_input_ids": context_prefix["input_ids"],
            "question_cache_ids": question_prefix["cache_ids"],
            "seed_token": question_prefix["seed_token"],
            "was_truncated": bool(context_prefix.get("was_truncated", False)),
        }

    prefix = prepare_generation_prefix(
        tokenizer=tokenizer,
        context=context,
        question=question,
        device=device,
    )
    return {
        "cache_input_ids": prefix["cache_ids"],
        "question_cache_ids": None,
        "seed_token": prefix["seed_token"],
        "was_truncated": False,
    }


def predict_generation_task_answer(
    spec: HFDatasetSpec,
    model,
    tokenizer,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    eval_config,
    context: str,
    question_cache_ids: Optional[torch.Tensor] = None,
) -> str:
    generation_past = past_key_values
    if question_cache_ids is not None:
        generation_past = append_input_ids_to_past(
            model=model,
            past_key_values=past_key_values,
            input_ids=question_cache_ids,
        )

    return generate_greedy_answer(
        model=model,
        tokenizer=tokenizer,
        past_key_values=generation_past,
        seed_token=seed_token,
        max_new_tokens=int(getattr(eval_config, "generation_max_new_tokens")),
    )



@torch.inference_mode()
def append_input_ids_to_past(
    model,
    past_key_values: PastKeyValues,
    input_ids: torch.Tensor,
) -> PastKeyValues:
    if input_ids.shape[1] == 0:
        return past_key_values

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return outputs.past_key_values


def build_text_candidate_token_ids(
    tokenizer,
    candidates: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    token_ids_by_label: Dict[str, torch.Tensor] = {}

    for label, text in candidates.items():
        normalized_text = text.strip()
        token_ids = tokenizer(
            f" {normalized_text}",
            add_special_tokens=False,
        ).input_ids
        if len(token_ids) < 1:
            raise ValueError(f"Failed to tokenize candidate text for label={label}: {text!r}")
        token_ids_by_label[label] = torch.tensor(token_ids, dtype=torch.long)

    return token_ids_by_label


def build_logit_answer_candidates(
    tokenizer,
    spec: HFDatasetSpec,
    example: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    if spec.answer_mode == "boolq":
        return build_text_candidate_token_ids(
            tokenizer,
            {"yes": "yes", "no": "no"},
        )

    if spec.answer_mode == "pubmed_qa":
        return build_text_candidate_token_ids(
            tokenizer,
            {"yes": "yes", "no": "no", "maybe": "maybe"},
        )

    raise ValueError(f"Unsupported answer_mode for logit scoring: {spec.answer_mode}")


def score_candidate_logprob(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    normalize_by_length: bool = True,
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

    score = token_log_probs.sum(dim=1).item()
    if normalize_by_length:
        score /= candidate_ids.shape[1]
    return score


def score_answer_choices(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    choice_token_ids: Dict[str, torch.Tensor],
    normalize_by_length: bool = True,
) -> Dict[str, float]:
    return {
        label: score_candidate_logprob(
            model=model,
            past_key_values=past_key_values,
            seed_token=seed_token,
            candidate_token_ids=choice_token_ids[label],
            normalize_by_length=normalize_by_length,
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


@torch.inference_mode()
def generate_greedy_answer(
    model,
    tokenizer,
    past_key_values: PastKeyValues,
    seed_token: torch.Tensor,
    max_new_tokens: int,
) -> str:
    generated_token_ids: List[int] = []
    current_input_ids = seed_token
    current_past = past_key_values
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_input_ids,
            past_key_values=current_past,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token_id = int(next_token.item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_token_ids.append(next_token_id)
        current_input_ids = next_token
        current_past = outputs.past_key_values

    decoded = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return postprocess_generated_answer(decoded)


def postprocess_generated_answer(text: str) -> str:
    cleaned = text.strip()
    for stopper in ["\n", "\r", "Question:", "Context:", "Answer:"]:
        if stopper in cleaned:
            cleaned = cleaned.split(stopper, 1)[0].strip()
    return cleaned


def normalize_qa_text(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _compute_pair_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_qa_text(prediction).split()
    gold_tokens = normalize_qa_text(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_generation_f1(prediction: str, gold_answers: List[str]) -> float:
    return max(_compute_pair_f1(prediction, gold_answer) for gold_answer in gold_answers)


def summarize_path_metrics(path_metrics: Dict[str, RunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}

    for path_name, meter in path_metrics.items():
        results[path_name] = meter.summary()

    return results


def summarize_generation_path_metrics(path_metrics: Dict[str, GenerationRunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}

    for path_name, meter in path_metrics.items():
        results[path_name] = meter.summary()

    return results


def build_direction_pretty_name(direction: str, nodes: List[Node], edges: List[Edge]) -> str:
    node_map = build_node_map(nodes)
    edge_map = build_edge_map(edges)
    edge = edge_map.get(direction)
    if edge is None:
        return direction
    src_model_id = node_map[edge.src_id].model_id
    dst_model_id = node_map[edge.dst_id].model_id
    return f"{edge.id} ({src_model_id} -> {dst_model_id})"


def log_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
    active_directions,
) -> None:
    logger.info("===== %s =====", dataset_name)
    for direction in active_directions:
        row = results[direction]
        pretty_name = build_direction_pretty_name(direction, nodes, edges)
        logger.info(
            "%s | cosine=%.6f | accuracy=%.6f | native_accuracy=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["accuracy"],
            row["native_accuracy"],
            int(row["count"]),
        )


def log_generation_dataset_result(
    logger: logging.Logger,
    dataset_name: str,
    results: Dict[str, Dict[str, float]],
    nodes: List[Node],
    edges: List[Edge],
    active_directions,
) -> None:
    logger.info("===== %s =====", dataset_name)
    for direction in active_directions:
        row = results[direction]
        pretty_name = build_direction_pretty_name(direction, nodes, edges)
        logger.info(
            "%s | cosine=%.6f | f1=%.6f | native_f1=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["f1"],
            row["native_f1"],
            int(row["count"]),
        )


def _is_valid_summary_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def _summary_mean(values: List[float]) -> float:
    valid_values = [float(value) for value in values if _is_valid_summary_value(value)]
    if not valid_values:
        return float("nan")
    return sum(valid_values) / len(valid_values)


def _format_summary_float(value: float) -> str:
    if not _is_valid_summary_value(value):
        return "N/A"
    return f"{float(value):.3f}"


def _format_summary_percent(value: float) -> str:
    if not _is_valid_summary_value(value):
        return "N/A"
    return f"{float(value) * 100.0:.1f}%"


def build_direction_summary_markdown_table(
    alg: str,
    direction: str,
    nodes: List[Node],
    edges: List[Edge],
    all_logit_results: Dict[str, Dict[str, Dict[str, float]]],
    all_generation_results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    node_map = build_node_map(nodes)
    edge_map = build_edge_map(edges)
    edge = edge_map.get(direction)

    logit_dataset_keys = [
        ("BoolQ", "BoolQ/validation"),
        ("PubMedQA", "PubMedQA/pqa_labeled/train"),
    ]
    multinews_dataset_key = "MultiNews/validation"

    logit_rows = {
        display_name: all_logit_results.get(dataset_key, {}).get(direction, {})
        for display_name, dataset_key in logit_dataset_keys
    }
    multinews_row = all_generation_results.get(multinews_dataset_key, {}).get(direction, {})

    translated_cosine_avg = _summary_mean([
        logit_rows["BoolQ"].get("cosine", float("nan")),
        logit_rows["PubMedQA"].get("cosine", float("nan")),
    ])
    translated_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("accuracy", float("nan")),
        logit_rows["PubMedQA"].get("accuracy", float("nan")),
    ])
    native_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("native_accuracy", float("nan")),
        logit_rows["PubMedQA"].get("native_accuracy", float("nan")),
    ])

    if edge is None:
        target_model_id = "target"
        direction_title = direction
    else:
        src_model_id = node_map[edge.src_id].model_id
        target_model_id = node_map[edge.dst_id].model_id
        direction_title = f"{edge.id} 방향 ({src_model_id} -> {target_model_id})"

    lines = [
        f"### {direction_title}",
        "",
        "| Method | Cosine Sim Avg | BoolQ | PubMedQA | Acc Avg | MultiNews |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {target_model_id} (baseline) | N/A | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(native_accuracy_avg)} | "
            f"{_format_summary_float(multinews_row.get('native_f1', float('nan')))} |"
        ),
        (
            f"| {alg} | "
            f"{_format_summary_float(translated_cosine_avg)} | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(translated_accuracy_avg)} | "
            f"{_format_summary_float(multinews_row.get('f1', float('nan')))} |"
        ),
    ]
    return "\n".join(lines)


def build_final_summary_markdown(
    alg: str,
    nodes: List[Node],
    edges: List[Edge],
    active_directions,
    all_logit_results: Dict[str, Dict[str, Dict[str, float]]],
    all_generation_results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    sections = []

    for direction in active_directions:
        sections.append(
            build_direction_summary_markdown_table(
                alg=alg,
                direction=direction,
                nodes=nodes,
                edges=edges,
                all_logit_results=all_logit_results,
                all_generation_results=all_generation_results,
            )
        )

    return "\n\n".join(sections)
