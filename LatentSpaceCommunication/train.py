# code1_train_translator.py
# 목표: Cross Attention Translator 학습
# - A=gpt2, B=gpt2-medium
# - WikiText 데이터로 2000 step 학습
# - 매 100 step 마다 validation reconstruction loss 표시
#
# transformers==4.35.2 전제

import math
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from translator_lib import (
    SharedSpaceKVTranslator,
    pack_past_key_values,
)

# -------------------
# 설정 (argparse 금지)
# -------------------
MODEL_A_NAME = "gpt2"
MODEL_B_NAME = "gpt2-medium"

SEED = 42
CONTEXT_LEN = 64          # toy: 짧게
BATCH_SIZE = 2
TRAIN_STEPS = 2000
VAL_EVERY = 100
VAL_BATCHES = 10          # validation은 일부 배치만

LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

# shared space / translator size (toy)
Q_DIM = 1536              # 24로 나누어 떨어짐(12,24 둘 다 OK)
D_MODEL = 256
N_HEADS = 8
DROPOUT = 0.0

SAVE_PATH = "translator_ckpt.pt"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lm_blocks(tokenizer, texts, block_size: int):
    # 텍스트 전체를 이어붙여 block으로 자르는 전형적인 방식
    joined = "\n\n".join([t for t in texts if t is not None])
    ids = tokenizer(joined, return_tensors=None)["input_ids"]
    # ids: List[int]
    n_blocks = len(ids) // block_size
    ids = ids[: n_blocks * block_size]
    blocks = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    return blocks


def collate_fn(batch):
    # batch: List[List[int]] length=CONTEXT_LEN
    input_ids = torch.tensor(batch, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def get_kv_from_model(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past = out.past_key_values
    K, V = pack_past_key_values(past)
    return K, V


@torch.no_grad()
def eval_recon_loss(translator, modelA, modelB, val_loader, device):
    translator.eval()
    total = 0.0
    n = 0

    for i, batch in enumerate(val_loader):
        if i >= VAL_BATCHES:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        K_A, V_A = get_kv_from_model(modelA, input_ids, attention_mask)
        K_B, V_B = get_kv_from_model(modelB, input_ids, attention_mask)

        loss = translator(K_A, V_A, K_B, V_B)
        total += float(loss.item())
        n += 1

    translator.train()
    return total / max(1, n)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_A_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # datasets
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]

    train_blocks = make_lm_blocks(tokenizer, train_texts, CONTEXT_LEN)
    val_blocks = make_lm_blocks(tokenizer, val_texts, CONTEXT_LEN)

    train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)

    # models (frozen)
    modelA = GPT2LMHeadModel.from_pretrained(MODEL_A_NAME).to(device)
    modelB = GPT2LMHeadModel.from_pretrained(MODEL_B_NAME).to(device)
    modelA.eval()
    modelB.eval()
    for p in modelA.parameters():
        p.requires_grad_(False)
    for p in modelB.parameters():
        p.requires_grad_(False)

    modelA.config.pad_token_id = tokenizer.eos_token_id
    modelB.config.pad_token_id = tokenizer.eos_token_id

    # specs
    a_layers = modelA.config.n_layer
    a_hidden = modelA.config.n_embd
    b_layers = modelB.config.n_layer
    b_hidden = modelB.config.n_embd

    translator = SharedSpaceKVTranslator(
        a_layers=a_layers, a_hidden=a_hidden,
        b_layers=b_layers, b_hidden=b_hidden,
        q_dim=Q_DIM, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT
    ).to(device)
    translator.train()

    opt = torch.optim.AdamW(translator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # infinite iterator
    train_iter = iter(train_loader)

    for step in range(1, TRAIN_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # target KV from frozen LMs
        with torch.no_grad():
            K_A, V_A = get_kv_from_model(modelA, input_ids, attention_mask)
            K_B, V_B = get_kv_from_model(modelB, input_ids, attention_mask)

        opt.zero_grad(set_to_none=True)
        loss = translator(K_A, V_A, K_B, V_B)
        loss.backward()

        nn.utils.clip_grad_norm_(translator.parameters(), GRAD_CLIP_NORM)
        opt.step()

        if step % 10 == 0:
            print(f"[train] step={step:4d} loss={loss.item():.6f}")

        if step % VAL_EVERY == 0:
            val_loss = eval_recon_loss(translator, modelA, modelB, val_loader, device)
            print(f"[valid] step={step:4d} recon_loss={val_loss:.6f}")

    # save
    ckpt = {
        "translator_state": translator.state_dict(),
        "config": {
            "MODEL_A_NAME": MODEL_A_NAME,
            "MODEL_B_NAME": MODEL_B_NAME,
            "Q_DIM": Q_DIM,
            "D_MODEL": D_MODEL,
            "N_HEADS": N_HEADS,
            "DROPOUT": DROPOUT,
            "a_layers": a_layers,
            "a_hidden": a_hidden,
            "b_layers": b_layers,
            "b_hidden": b_hidden,
        },
    }
    torch.save(ckpt, SAVE_PATH)
    print("saved:", SAVE_PATH)


if __name__ == "__main__":
    main()
