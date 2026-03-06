# code1_train_B2A.py
# 목표: gpt2-medium(B) -> gpt2(A) 단방향 translator 학습
# - WikiText 데이터로 2000 step
# - 매 100 step 마다 validation reconstruction loss 표시
# - argparse 금지
# - transformers==4.35.2

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from translator_lib import (
    OneWayKVTranslator_B2A,
    pack_past_key_values,
)

# -------------------
# 설정
# -------------------
MODEL_A_NAME = "gpt2"         # target
MODEL_B_NAME = "gpt2-medium"  # source

SEED = 42
CONTEXT_LEN = 64
BATCH_SIZE = 2

TRAIN_STEPS = 2000
VAL_EVERY = 100
VAL_BATCHES = 10

LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

# shared space dims (must be divisible by both 12 and 24)
Q_DIM = 1536
D_MODEL = 256
N_HEADS = 8
DROPOUT = 0.0

SAVE_PATH = "translator_B2A_ckpt.pt"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lm_blocks(tokenizer, texts, block_size: int):
    joined = "\n\n".join([t for t in texts if t is not None])
    ids = tokenizer(joined, return_tensors=None)["input_ids"]
    n_blocks = len(ids) // block_size
    ids = ids[: n_blocks * block_size]
    blocks = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    return blocks


def collate_fn(batch):
    input_ids = torch.tensor(batch, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def get_kv(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    K, V = pack_past_key_values(out.past_key_values)
    return K, V


@torch.no_grad()
def eval_val_loss(translator, modelA, modelB, val_loader, device):
    translator.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= VAL_BATCHES:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # source(B) + target(A)
        K_B, V_B = get_kv(modelB, input_ids, attention_mask)
        K_A, V_A = get_kv(modelA, input_ids, attention_mask)

        loss = translator(K_B, V_B, K_A, V_A)
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

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_blocks = make_lm_blocks(tokenizer, ds["train"]["text"], CONTEXT_LEN)
    val_blocks = make_lm_blocks(tokenizer, ds["validation"]["text"], CONTEXT_LEN)

    train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)

    # frozen LMs
    modelA = GPT2LMHeadModel.from_pretrained(MODEL_A_NAME).to(device).eval()
    modelB = GPT2LMHeadModel.from_pretrained(MODEL_B_NAME).to(device).eval()
    modelA.config.pad_token_id = tokenizer.eos_token_id
    modelB.config.pad_token_id = tokenizer.eos_token_id
    for p in modelA.parameters():
        p.requires_grad_(False)
    for p in modelB.parameters():
        p.requires_grad_(False)

    a_layers, a_hidden = modelA.config.n_layer, modelA.config.n_embd
    b_layers, b_hidden = modelB.config.n_layer, modelB.config.n_embd

    translator = OneWayKVTranslator_B2A(
        b_layers=b_layers, b_hidden=b_hidden,
        a_layers=a_layers, a_hidden=a_hidden,
        q_dim=Q_DIM, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT
    ).to(device).train()

    opt = torch.optim.AdamW(translator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_iter = iter(train_loader)

    for step in range(1, TRAIN_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            K_B, V_B = get_kv(modelB, input_ids, attention_mask)
            K_A, V_A = get_kv(modelA, input_ids, attention_mask)

        opt.zero_grad(set_to_none=True)
        loss = translator(K_B, V_B, K_A, V_A)
        loss.backward()
        nn.utils.clip_grad_norm_(translator.parameters(), GRAD_CLIP_NORM)
        opt.step()

        if step % 10 == 0:
            print(f"[train] step={step:4d} loss={loss.item():.6f}")

        if step % VAL_EVERY == 0:
            val_loss = eval_val_loss(translator, modelA, modelB, val_loader, device)
            print(f"[valid] step={step:4d} recon_loss={val_loss:.6f}")

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
