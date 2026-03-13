from __future__ import annotations

import random
from typing import Dict, Iterable, Iterator, List


class _BaseDataset:
    def __init__(self, items: Iterable[Dict]) -> None:
        self._items = list(items)

    def shuffle(self, seed: int = 42, buffer_size: int | None = None):
        rng = random.Random(seed)
        items = list(self._items)
        rng.shuffle(items)
        return self.__class__(items)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self._items)


class FakeStreamingDataset(_BaseDataset):
    pass


class FakeMapDataset(_BaseDataset):
    pass


def load_dataset(dataset_path: str, dataset_name: str | None = None, split: str | None = None, streaming: bool = False):
    key = (dataset_path, dataset_name, split)

    if key == ("openwebtext", None, "train"):
        items = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Smoke tests should be fast, deterministic, and CPU-friendly."},
            {"text": "Tiny synthetic corpora are enough to exercise the training loop."},
        ]
        return FakeStreamingDataset(items)

    if key == ("google/boolq", None, "validation"):
        items = [
            {"question": "Is water wet?", "passage": "Water makes things wet.", "answer": True},
            {"question": "Is fire cold?", "passage": "Fire is hot.", "answer": False},
        ]
        return FakeMapDataset(items)

    if key == ("qiaojin/PubMedQA", "pqa_labeled", "train"):
        items = [
            {"question": "Does rest help recovery?", "context": {"contexts": ["Rest supports recovery."]}, "final_decision": "yes"},
            {"question": "Can rocks breathe?", "context": {"contexts": ["Rocks are not alive."]}, "final_decision": "no"},
        ]
        return FakeMapDataset(items)

    if key == ("cais/mmlu", "all", "validation"):
        items = [
            {
                "question": "Which letter comes first?",
                "choices": ["A", "B", "C", "D"],
                "answer": 0,
                "subject": "toy_subject",
            },
            {
                "question": "Which number is even?",
                "choices": ["1", "3", "4", "5"],
                "answer": 2,
                "subject": "toy_subject",
            },
        ]
        return FakeMapDataset(items)

    if key == ("rajpurkar/squad", None, "validation"):
        items = [
            {
                "question": "What color is the sky?",
                "context": "On a clear day, the sky looks blue.",
                "answers": {"text": ["blue"]},
            },
            {
                "question": "What do bees make?",
                "context": "Bees are known for making honey.",
                "answers": {"text": ["honey"]},
            },
        ]
        return FakeMapDataset(items)

    raise ValueError(
        f"Unsupported fake dataset request: dataset_path={dataset_path!r}, dataset_name={dataset_name!r}, split={split!r}, streaming={streaming!r}"
    )
