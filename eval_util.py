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


def extract_question_and_answer(spec: HFDatasetSpec, example: Dict) -> Optional[Dict[str, Any]]:
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
    answer_mode: Optional[str] = None,
) -> str:
    question = question.strip()

    if answer_mode == "boolq":
        return (
            "Read the question and answer with exactly one word: yes or no.\n\n"
            f"Question: {question}\n"
            "Answer: "
        )

    if answer_mode == "pubmed_qa":
        return (
            "Read the question and answer with exactly one word: yes, no, or maybe.\n\n"
            f"Question: {question}\n"
            "Answer: "
        )

    if answer_mode == "mmlu":
        if not choices:
            raise ValueError("MMLU requires choices.")

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

        prompt_lines.append(
            "Answer with the correct option in the exact format '<letter>. <choice>'."
        )
        prompt_lines.append("Answer: ")
        return "\n".join(prompt_lines)

    # fallback / legacy path
    if choices:
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
        prompt_lines.append("Answer: ")
        return "\n".join(prompt_lines)

    return f"Question: {question}\nAnswer: "


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
    answer_mode: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    prefix_text = format_question_prefix(
        question,
        choices=choices,
        subject=subject,
        answer_mode=answer_mode,
    )
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def prepare_generation_prefix(tokenizer, context: str, question: str, device: str) -> Dict[str, torch.Tensor]:
    prefix_text = format_generation_prompt(context=context, question=question)
    return prepare_text_prefix(tokenizer=tokenizer, prefix_text=prefix_text, device=device)


def build_answer_candidates(
    spec: HFDatasetSpec,
    example: Dict[str, Any],
) -> Dict[str, List[str]]:
    if spec.answer_mode == "boolq":
        return {
            "yes": ["yes"],
            "no": ["no"],
        }

    if spec.answer_mode == "pubmed_qa":
        return {
            "yes": ["yes"],
            "no": ["no"],
            "maybe": ["maybe"],
        }

    if spec.answer_mode == "mmlu":
        choices = example.get("choices", None)
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError("MMLU example must contain exactly 4 choices.")

        normalized_choices = []
        for choice in choices:
            if not isinstance(choice, str) or not choice.strip():
                raise ValueError("Invalid MMLU choice.")
            normalized_choices.append(choice.strip())

        return {
            chr(ord("A") + idx): [f"{chr(ord('A') + idx)}. {choice}"]
            for idx, choice in enumerate(normalized_choices)
        }

    raise ValueError(f"Unsupported answer_mode: {spec.answer_mode}")


def build_candidate_token_ids(
    tokenizer,
    answer_candidates: Dict[str, List[str]],
) -> Dict[str, List[torch.Tensor]]:
    candidate_token_ids: Dict[str, List[torch.Tensor]] = {}

    for label, candidate_texts in answer_candidates.items():
        tokenized_variants: List[torch.Tensor] = []
        for text in candidate_texts:
            token_ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(token_ids) < 1:
                raise ValueError(f"Failed to tokenize answer candidate: {label} -> {text!r}")
            tokenized_variants.append(torch.tensor(token_ids, dtype=torch.long))
        candidate_token_ids[label] = tokenized_variants

    return candidate_token_ids


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
    total_logprob = token_log_probs.sum(dim=1).item()

    if normalize_by_length:
        return total_logprob / candidate_ids.shape[1]
    return total_logprob


def score_answer_choices(
    model,
    past_key_values,
    seed_token: torch.Tensor,
    choice_token_ids: Dict[str, List[torch.Tensor]],
    normalize_by_length: bool = True,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for label, variants in choice_token_ids.items():
        variant_scores = [
            score_candidate_logprob(
                model=model,
                past_key_values=past_key_values,
                seed_token=seed_token,
                candidate_token_ids=variant,
                normalize_by_length=normalize_by_length,
            )
            for variant in variants
        ]
        scores[label] = max(variant_scores)

    return scores


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


def summarize_generation_path_metrics(path_metrics: Dict[str, GenerationRunningAverage]) -> Dict[str, Dict[str, float]]:
    results = {}
    total_cosine_sum = 0.0
    total_exact_match_sum = 0.0
    total_f1_sum = 0.0
    total_native_exact_match_sum = 0.0
    total_native_f1_sum = 0.0
    total_count = 0

    for path_name, meter in path_metrics.items():
        path_result = meter.summary()
        results[path_name] = path_result

        total_cosine_sum += meter.cosine_sum
        total_exact_match_sum += meter.exact_match_sum
        total_f1_sum += meter.f1_sum
        total_native_exact_match_sum += meter.native_exact_match_sum
        total_native_f1_sum += meter.native_f1_sum
        total_count += meter.count

    if total_count == 0:
        results["AVG"] = {
            "cosine": float("nan"),
            "exact_match": float("nan"),
            "f1": float("nan"),
            "native_exact_match": float("nan"),
            "native_f1": float("nan"),
            "count": 0,
        }
    else:
        results["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "exact_match": total_exact_match_sum / total_count,
            "f1": total_f1_sum / total_count,
            "native_exact_match": total_native_exact_match_sum / total_count,
            "native_f1": total_native_f1_sum / total_count,
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


def summarize_generation_overall_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    active_directions,
) -> Dict[str, Dict[str, float]]:
    overall = {
        direction: {
            "cosine_sum": 0.0,
            "exact_match_sum": 0.0,
            "f1_sum": 0.0,
            "native_exact_match_sum": 0.0,
            "native_f1_sum": 0.0,
            "count": 0,
        }
        for direction in active_directions
    }

    for dataset_result in all_results.values():
        for direction in active_directions:
            row = dataset_result[direction]
            count = int(row["count"])
            overall[direction]["cosine_sum"] += float(row["cosine"]) * count
            overall[direction]["exact_match_sum"] += float(row["exact_match"]) * count
            overall[direction]["f1_sum"] += float(row["f1"]) * count
            overall[direction]["native_exact_match_sum"] += float(row["native_exact_match"]) * count
            overall[direction]["native_f1_sum"] += float(row["native_f1"]) * count
            overall[direction]["count"] += count

    summarized = {}
    total_cosine_sum = 0.0
    total_exact_match_sum = 0.0
    total_f1_sum = 0.0
    total_native_exact_match_sum = 0.0
    total_native_f1_sum = 0.0
    total_count = 0

    for direction in active_directions:
        count = overall[direction]["count"]
        if count == 0:
            summarized[direction] = {
                "cosine": float("nan"),
                "exact_match": float("nan"),
                "f1": float("nan"),
                "native_exact_match": float("nan"),
                "native_f1": float("nan"),
                "count": 0,
            }
        else:
            summarized[direction] = {
                "cosine": overall[direction]["cosine_sum"] / count,
                "exact_match": overall[direction]["exact_match_sum"] / count,
                "f1": overall[direction]["f1_sum"] / count,
                "native_exact_match": overall[direction]["native_exact_match_sum"] / count,
                "native_f1": overall[direction]["native_f1_sum"] / count,
                "count": count,
            }

        total_cosine_sum += overall[direction]["cosine_sum"]
        total_exact_match_sum += overall[direction]["exact_match_sum"]
        total_f1_sum += overall[direction]["f1_sum"]
        total_native_exact_match_sum += overall[direction]["native_exact_match_sum"]
        total_native_f1_sum += overall[direction]["native_f1_sum"]
        total_count += count

    if total_count == 0:
        summarized["AVG"] = {
            "cosine": float("nan"),
            "exact_match": float("nan"),
            "f1": float("nan"),
            "native_exact_match": float("nan"),
            "native_f1": float("nan"),
            "count": 0,
        }
    else:
        summarized["AVG"] = {
            "cosine": total_cosine_sum / total_count,
            "exact_match": total_exact_match_sum / total_count,
            "f1": total_f1_sum / total_count,
            "native_exact_match": total_native_exact_match_sum / total_count,
            "native_f1": total_native_f1_sum / total_count,
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


def log_generation_dataset_result(
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
            "%s | cosine=%.6f | exact_match=%.6f | f1=%.6f | native_exact_match=%.6f | native_f1=%.6f | count=%d",
            pretty_name,
            row["cosine"],
            row["exact_match"],
            row["f1"],
            row["native_exact_match"],
            row["native_f1"],
            int(row["count"]),
        )


def log_overall_result(
    logger: logging.Logger,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
    active_directions,
) -> None:
    logger.info("===== OVERALL AVERAGE ACROSS LOGIT-SCORE QA DATASETS =====")
    for direction in list(active_directions) + ["AVG"]:
        row = results[direction]
        pretty_name = (
            "AVG (weighted over all logit-score QA datasets and evaluated paths)"
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


def log_generation_overall_result(
    logger: logging.Logger,
    results: Dict[str, Dict[str, float]],
    model_a_id: str,
    model_b_id: str,
    active_directions,
) -> None:
    logger.info("===== OVERALL AVERAGE ACROSS GENERATION QA DATASETS =====")
    for direction in list(active_directions) + ["AVG"]:
        row = results[direction]
        pretty_name = (
            "AVG (weighted over all generation QA datasets and evaluated paths)"
            if direction == "AVG"
            else build_direction_pretty_name(direction, model_a_id, model_b_id)
        )
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
