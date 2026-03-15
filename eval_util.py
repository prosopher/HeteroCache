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

    # generation QA (kept for non-SQuAD free-form eval paths)
    generation_max_new_tokens: int
    squad_max_answer_tokens: int

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


class GenerationRunningAverage:
    def __init__(self) -> None:
        self.cosine_sum = 0.0
        self.exact_match_sum = 0.0
        self.f1_sum = 0.0
        self.native_exact_match_sum = 0.0
        self.native_f1_sum = 0.0
        self.count = 0

    def update(
        self,
        cosine_value: float,
        exact_match_value: float,
        f1_value: float,
        native_exact_match_value: float,
        native_f1_value: float,
        n: int,
    ) -> None:
        self.cosine_sum += float(cosine_value) * n
        self.exact_match_sum += float(exact_match_value) * n
        self.f1_sum += float(f1_value) * n
        self.native_exact_match_sum += float(native_exact_match_value) * n
        self.native_f1_sum += float(native_f1_value) * n
        self.count += n

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "cosine": float("nan"),
                "exact_match": float("nan"),
                "f1": float("nan"),
                "native_exact_match": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        return {
            "cosine": self.cosine_sum / self.count,
            "exact_match": self.exact_match_sum / self.count,
            "f1": self.f1_sum / self.count,
            "native_exact_match": self.native_exact_match_sum / self.count,
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

    if spec.answer_mode == "mmlu":
        choices = example.get("choices", None)
        if not isinstance(choices, list) or len(choices) != 4:
            return None

        normalized_choices = [
            choice.strip()
            for choice in choices
            if isinstance(choice, str) and choice.strip()
        ]
        if len(normalized_choices) != 4:
            return None

        answer_value = example.get("answer", None)
        if isinstance(answer_value, int):
            if answer_value < 0 or answer_value >= 4:
                return None
            normalized_answer = chr(ord("A") + answer_value)
        elif isinstance(answer_value, str):
            normalized_answer = answer_value.strip().upper()
            if normalized_answer not in {"A", "B", "C", "D"}:
                return None
        else:
            return None

        qa_example = {
            "question": question.strip(),
            "choices": normalized_choices,
            "answer": normalized_answer,
        }

        subject_field = spec.subject_field or "subject"
        subject = example.get(subject_field, None)
        if isinstance(subject, str) and subject.strip():
            qa_example["subject"] = subject.strip()

        return qa_example

    if spec.answer_mode == "squad":
        context_field = spec.context_field or "context"
        answers_field = spec.answers_field or "answers"

        context = example.get(context_field, "")
        answers = example.get(answers_field, None)

        if not isinstance(context, str) or not context.strip():
            return None

        answer_texts: List[str] = []
        answer_starts: List[int] = []
        if isinstance(answers, dict):
            raw_texts = answers.get("text", [])
            if isinstance(raw_texts, list):
                answer_texts = [
                    item.strip()
                    for item in raw_texts
                    if isinstance(item, str) and item.strip()
                ]
            raw_starts = answers.get("answer_start", [])
            if isinstance(raw_starts, list):
                answer_starts = [
                    int(item)
                    for item in raw_starts
                    if isinstance(item, int) and item >= 0
                ]
        elif isinstance(answers, list):
            answer_texts = [
                item.strip()
                for item in answers
                if isinstance(item, str) and item.strip()
            ]

        if not answer_texts:
            return None

        result = {
            "question": question.strip(),
            "context": context.strip(),
            "answers": answer_texts,
        }
        if answer_starts:
            result["answer_starts"] = answer_starts
        return result

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

    if answer_mode == "mmlu":
        if not choices or len(choices) != 4:
            raise ValueError("MMLU requires exactly 4 choices.")

        prompt_lines = []
        if isinstance(subject, str) and subject.strip():
            prompt_lines.append(
                "The following is a multiple choice question about "
                f"{subject.strip().replace('_', ' ')}."
            )
            prompt_lines.append("")

        prompt_lines.append(f"Question: {question}")
        prompt_lines.append("Choices:")
        for idx, choice in enumerate(choices):
            label = chr(ord("A") + idx)
            prompt_lines.append(f"{label}. {choice.strip()}")
        prompt_lines.append("Answer with the exact option text only.")
        prompt_lines.append("Answer:")
        return "\n".join(prompt_lines)

    if not choices:
        return f"Question: {question}\nAnswer:"

    prompt_lines = [f"Question: {question}", "Choices:"]
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        prompt_lines.append(f"{label}. {choice.strip()}")
    prompt_lines.append("Answer:")
    return "\n".join(prompt_lines)


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


def prepare_full_text_inputs(tokenizer, text: str, device: str) -> Dict[str, Any]:
    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    if input_ids.shape[1] < 1:
        raise ValueError("Text must tokenize to at least 1 token.")
    return {
        "text": text,
        "input_ids": input_ids,
    }


def prepare_generation_context_inputs(tokenizer, context: str, device: str) -> Dict[str, Any]:
    prefix_text = format_generation_context_prefix(context=context)
    return prepare_full_text_inputs(tokenizer=tokenizer, text=prefix_text, device=device)



def prepare_generation_question_prefix(tokenizer, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_question_prefix(question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def format_squad_context_prefill_prefix(context: str) -> str:
    return f"Passage: {context.strip()}\n"


def format_squad_question_prefix(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Select the shortest exact answer span from the passage below.\n"
        "Passage: "
    )


def prepare_text_inputs_with_offsets(tokenizer, text: str, device: str) -> Dict[str, Any]:
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.get("offset_mapping")
    if offset_mapping is None:
        raise ValueError("Tokenizer must provide offset mappings for extractive SQuAD evaluation.")
    input_ids = tokenized["input_ids"].to(device)
    if input_ids.shape[1] < 1:
        raise ValueError("Text must tokenize to at least 1 token.")
    return {
        "text": text,
        "input_ids": input_ids,
        "offset_mapping": [tuple(item) for item in offset_mapping[0].tolist()],
    }


def prepare_squad_extractive_inputs(tokenizer, context: str, question: str, device: str) -> Dict[str, Any]:
    normalized_context = context.strip()
    return {
        "context_text": normalized_context,
        "context_prefill": prepare_full_text_inputs(
            tokenizer=tokenizer,
            text=format_squad_context_prefill_prefix(normalized_context),
            device=device,
        ),
        "question_prefill": prepare_full_text_inputs(
            tokenizer=tokenizer,
            text=format_squad_question_prefix(question),
            device=device,
        ),
        "scoring_context": prepare_text_inputs_with_offsets(
            tokenizer=tokenizer,
            text=normalized_context,
            device=device,
        ),
    }


@torch.inference_mode()
def append_input_ids_to_past_with_hidden_states(
    model,
    past_key_values: PastKeyValues,
    input_ids: torch.Tensor,
) -> Dict[str, Any]:
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token.")

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    return {
        "past_key_values": outputs.past_key_values,
        "hidden_states": outputs.hidden_states[-1],
    }


def _is_valid_context_token(offset_mapping: List[Tuple[int, int]], context_text: str, token_idx: int) -> bool:
    start_char, end_char = offset_mapping[token_idx]
    if end_char <= start_char:
        return False
    return bool(context_text[start_char:end_char].strip())


def select_best_span_from_hidden_states(
    question_hidden_states: torch.Tensor,
    context_hidden_states: torch.Tensor,
    context_text: str,
    offset_mapping: List[Tuple[int, int]],
    max_answer_tokens: int,
) -> Dict[str, Any]:
    if question_hidden_states.ndim != 3 or context_hidden_states.ndim != 3:
        raise ValueError("question_hidden_states and context_hidden_states must have shape [batch, seq, hidden].")
    if question_hidden_states.shape[0] != 1 or context_hidden_states.shape[0] != 1:
        raise ValueError("This extractive scorer expects batch size 1.")
    if max_answer_tokens < 1:
        raise ValueError("max_answer_tokens must be >= 1.")

    question_hidden = question_hidden_states[0].float()
    context_hidden = context_hidden_states[0].float()
    if context_hidden.shape[0] != len(offset_mapping):
        raise ValueError(
            f"context_hidden length {context_hidden.shape[0]} must match offset_mapping length {len(offset_mapping)}"
        )

    query_window = min(8, question_hidden.shape[0])
    start_query = F.normalize(question_hidden[-1], dim=0)
    end_query = F.normalize(question_hidden[-query_window:].mean(dim=0), dim=0)
    normalized_context = F.normalize(context_hidden, dim=-1)

    start_scores = normalized_context @ start_query
    end_scores = normalized_context @ end_query

    best_score = float("-inf")
    best_span = None
    best_char_span = (0, 0)

    valid_token_indices = [
        idx
        for idx in range(len(offset_mapping))
        if _is_valid_context_token(offset_mapping, context_text, idx)
    ]
    if not valid_token_indices:
        return {
            "text": "",
            "start_token_idx": -1,
            "end_token_idx": -1,
            "char_start": 0,
            "char_end": 0,
            "score": float("-inf"),
        }

    valid_token_index_set = set(valid_token_indices)
    for start_idx in valid_token_indices:
        max_end_idx = min(len(offset_mapping) - 1, start_idx + max_answer_tokens - 1)
        for end_idx in range(start_idx, max_end_idx + 1):
            if end_idx not in valid_token_index_set:
                continue
            score = float((start_scores[start_idx] + end_scores[end_idx]).item())
            if best_span is None or score > best_score:
                best_score = score
                best_span = (start_idx, end_idx)
                best_char_span = (offset_mapping[start_idx][0], offset_mapping[end_idx][1])

    if best_span is None:
        return {
            "text": "",
            "start_token_idx": -1,
            "end_token_idx": -1,
            "char_start": 0,
            "char_end": 0,
            "score": float("-inf"),
        }

    char_start, char_end = best_char_span
    return {
        "text": context_text[char_start:char_end].strip(),
        "start_token_idx": int(best_span[0]),
        "end_token_idx": int(best_span[1]),
        "char_start": int(char_start),
        "char_end": int(char_end),
        "score": best_score,
    }


@torch.inference_mode()
def predict_squad_extractive_answer(
    model,
    past_key_values: PastKeyValues,
    question_input_ids: torch.Tensor,
    scoring_context_input_ids: torch.Tensor,
    scoring_context_offset_mapping: List[Tuple[int, int]],
    context_text: str,
    max_answer_tokens: int,
) -> Dict[str, Any]:
    question_outputs = append_input_ids_to_past_with_hidden_states(
        model=model,
        past_key_values=past_key_values,
        input_ids=question_input_ids,
    )
    scoring_context_outputs = append_input_ids_to_past_with_hidden_states(
        model=model,
        past_key_values=question_outputs["past_key_values"],
        input_ids=scoring_context_input_ids,
    )
    return select_best_span_from_hidden_states(
        question_hidden_states=question_outputs["hidden_states"],
        context_hidden_states=scoring_context_outputs["hidden_states"],
        context_text=context_text,
        offset_mapping=scoring_context_offset_mapping,
        max_answer_tokens=max_answer_tokens,
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

    if spec.answer_mode == "mmlu":
        choices = example.get("choices", None)
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("MMLU example must contain exactly 4 choices.")

        return build_text_candidate_token_ids(
            tokenizer,
            {
                chr(ord("A") + idx): choice
                for idx, choice in enumerate(choices)
            },
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


def compute_generation_exact_match(prediction: str, gold_answers: List[str]) -> float:
    normalized_prediction = normalize_qa_text(prediction)
    for gold_answer in gold_answers:
        if normalized_prediction == normalize_qa_text(gold_answer):
            return 1.0
    return 0.0


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
            "%s | cosine=%.6f | exact_match=%.6f | f1=%.6f | native_exact_match=%.6f | native_f1=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["exact_match"],
            row["f1"],
            row["native_exact_match"],
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
        ("MMLU", "MMLU/all/validation"),
    ]
    squad_dataset_key = "SQuAD/validation"

    logit_rows = {
        display_name: all_logit_results.get(dataset_key, {}).get(direction, {})
        for display_name, dataset_key in logit_dataset_keys
    }
    squad_row = all_generation_results.get(squad_dataset_key, {}).get(direction, {})

    translated_cosine_avg = _summary_mean([
        logit_rows["BoolQ"].get("cosine", float("nan")),
        logit_rows["PubMedQA"].get("cosine", float("nan")),
        logit_rows["MMLU"].get("cosine", float("nan")),
    ])
    translated_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("accuracy", float("nan")),
        logit_rows["PubMedQA"].get("accuracy", float("nan")),
        logit_rows["MMLU"].get("accuracy", float("nan")),
    ])
    native_accuracy_avg = _summary_mean([
        logit_rows["BoolQ"].get("native_accuracy", float("nan")),
        logit_rows["PubMedQA"].get("native_accuracy", float("nan")),
        logit_rows["MMLU"].get("native_accuracy", float("nan")),
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
        "| Method | Cosine Sim Avg | BoolQ | PubMedQA | MMLU | Acc Avg | SQuAD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| {target_model_id} (baseline) | N/A | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['MMLU'].get('native_accuracy', float('nan')))} | "
            f"{_format_summary_percent(native_accuracy_avg)} | "
            f"{_format_summary_float(squad_row.get('native_f1', float('nan')))} |"
        ),
        (
            f"| {alg} | "
            f"{_format_summary_float(translated_cosine_avg)} | "
            f"{_format_summary_percent(logit_rows['BoolQ'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['PubMedQA'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(logit_rows['MMLU'].get('accuracy', float('nan')))} | "
            f"{_format_summary_percent(translated_accuracy_avg)} | "
            f"{_format_summary_float(squad_row.get('f1', float('nan')))} |"
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
