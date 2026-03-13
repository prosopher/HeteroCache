from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn


class PreTrainedTokenizerBase:
    pass


class TinyTokenizer(PreTrainedTokenizerBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self._vocab_size = 128

    def _encode_text(self, text: str) -> List[int]:
        # Simple byte-ish tokenizer: stable, fast, and guaranteed to emit >=1 token
        if not text:
            return [self.eos_token_id]
        return [2 + (ord(ch) % (self._vocab_size - 2)) for ch in text]

    def __call__(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = True, verbose: bool = False):
        token_ids = self._encode_text(text)
        if add_special_tokens:
            token_ids = token_ids + [self.eos_token_id]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([token_ids], dtype=torch.long))
        return SimpleNamespace(input_ids=token_ids)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(chr((int(token_id) - 2) % 95 + 32))
        return "".join(chars)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_id: str) -> TinyTokenizer:
        return TinyTokenizer(model_id)


class PreTrainedModel(nn.Module):
    config: object


@dataclass
class TinyConfig:
    _name_or_path: str
    n_head: int = 2
    n_embd: int = 8
    n_layer: int = 2
    vocab_size: int = 128


class TinyCausalLM(PreTrainedModel):
    def __init__(self, model_id: str, torch_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = TinyConfig(_name_or_path=model_id)

        # Make weights deterministic per model id.
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(model_id)) % 10_000
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embed = nn.Embedding(self.config.vocab_size, self.config.n_embd)
            self.layers = nn.ModuleList(
                [nn.Linear(self.config.n_embd, self.config.n_embd) for _ in range(self.config.n_layer)]
            )
            self.cache_projs = nn.ModuleList(
                [nn.Linear(self.config.n_embd, self.config.n_embd) for _ in range(self.config.n_layer)]
            )
            self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.to(dtype=torch_dtype)

    def forward(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True):
        hidden = self.embed(input_ids)
        batch_size, seq_len, hidden_size = hidden.shape
        num_heads = self.config.n_head
        head_dim = hidden_size // num_heads

        new_past = []
        for layer_idx in range(self.config.n_layer):
            layer_hidden = torch.tanh(self.layers[layer_idx](hidden))

            if past_key_values is not None:
                past_key, past_value = past_key_values[layer_idx]
                past_flat = past_value.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hidden_size)
                past_summary = past_flat.mean(dim=1, keepdim=True)
                layer_hidden = layer_hidden + 0.1 * self.cache_projs[layer_idx](past_summary)

            key_flat = layer_hidden
            value_flat = layer_hidden + 0.01 * (layer_idx + 1)
            key = key_flat.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
            value = value_flat.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

            if past_key_values is not None:
                prev_key, prev_value = past_key_values[layer_idx]
                key = torch.cat([prev_key, key], dim=2)
                value = torch.cat([prev_value, value], dim=2)

            new_past.append((key, value))
            hidden = layer_hidden

        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits, past_key_values=tuple(new_past))


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model_id: str, torch_dtype: torch.dtype = torch.float32) -> TinyCausalLM:
        return TinyCausalLM(model_id=model_id, torch_dtype=torch_dtype)
