from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_PATH = REPO_ROOT / "tests" / "stubs"
if str(STUBS_PATH) not in sys.path:
    sys.path.insert(0, str(STUBS_PATH))

from transformers import TinyTokenizer


def test_tiny_tokenizer_supports_truncation_and_max_length() -> None:
    tokenizer = TinyTokenizer("tiny-a")
    encoded = tokenizer("abcdef", add_special_tokens=False, truncation=True, max_length=3)
    assert encoded.input_ids == tokenizer._encode_text("abc")


def test_tiny_tokenizer_exposes_small_model_max_length_for_smoke_eval() -> None:
    tokenizer = TinyTokenizer("tiny-a")
    assert tokenizer.model_max_length == 128
