# code1_train_B2A.py
# gpt2-medium(B) -> gpt2(A) 단방향 translator 학습
# 논문 쪽 설정에 더 가깝게 수정:
# - 50k optimizer steps
# - warmup 2.5k + cosine decay
# - adapter heads=32, head_dim=64
# - translation_dim_factor=1
# - context len 128
# - effective batch size 256 via grad accumulation
#
# 주의:
# - reconstruction loss만 유지 (요청사항)
# - dataset은 WikiText 유지 (요청사항)
# - argparse 사용 안 함
# - transformers==4.35.2

import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from translator_lib import OneWayKVTranslator_B2A, pack_past_key_values

# -------------------
# 설정
# -------------------
MODEL_A_NAME = "gpt2"         # target
MODEL_B_NAME = "gpt2-medium"  # source

SEED = 42

# 논문 default prefix 길이에 맞춰 128
CONTEXT_LEN = 128

# paper-like effective batch
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 64   # 4 * 64 = 256 effective batch

TRAIN_STEPS = 50_000
VAL_EVERY = 100
VAL_BATCHES = 8

# paper-like optimizer schedule
LR_INIT = 1e-6
LR_PEAK = 1e-4
WARMUP_STEPS = 2_500
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01

# paper-like adapter size
ADAPTER_NUM_HEADS = 32
ADAPTER_HEAD_DIM = 64          # => adapter_dim = 2048
TRANSLATION_DIM_FACTOR = 1.0   # paper default
Q_DIM = 6144                   # implementation choice; divisible by 12 and 24
DROPOUT = 0.0

SAVE_PATH = "translator_B2A_paperlike_ckpt.pt"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_lm_blocks(tokenizer, texts, block_size: int):
    joined = "\n\n".join([t for t in texts if t is not None and len(t) > 0])
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
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    K, V = pack_past_key_values(out.past_key_values)
    return K, V


def compute_lr(step: int) -> float:
    # step is 1-based optimizer step
    if step <= WARMUP_STEPS:
        alpha = (step - 1) / max(1, WARMUP_STEPS - 1)
        return LR_INIT + alpha * (LR_PEAK - LR_INIT)

    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_PEAK * cosine


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def eval_val_loss(translator, modelA, modelB, val_loader, device):
    translator.eval()
    total = 0.0
    n = 0

    for i, batch in enumerate(val_loader):
        if i >= VAL_BATCHES:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

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

    train_loader = DataLoader(
        train_blocks,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_blocks,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )

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
        b_layers=b_layers,
        b_hidden=b_hidden,
        a_layers=a_layers,
        a_hidden=a_hidden,
        q_dim=Q_DIM,
        adapter_num_heads=ADAPTER_NUM_HEADS,
        adapter_head_dim=ADAPTER_HEAD_DIM,
        translation_dim_factor=TRANSLATION_DIM_FACTOR,
        dropout=DROPOUT,
    ).to(device)

    translator.train()

    optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=LR_INIT,
        weight_decay=WEIGHT_DECAY,
    )

    train_iter = iter(train_loader)

    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        running_micro_loss = 0.0

        for micro_idx in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                K_B, V_B = get_kv(modelB, input_ids, attention_mask)
                K_A, V_A = get_kv(modelA, input_ids, attention_mask)

            loss = translator(K_B, V_B, K_A, V_A)
            running_micro_loss += float(loss.item())

            (loss / GRAD_ACCUM_STEPS).backward()

        nn.utils.clip_grad_norm_(translator.parameters(), GRAD_CLIP_NORM)

        lr = compute_lr(step)
        set_optimizer_lr(optimizer, lr)
        optimizer.step()

        if step % 10 == 0:
            avg_micro_loss = running_micro_loss / GRAD_ACCUM_STEPS
            print(
                f"[train] step={step:6d} "
                f"loss={avg_micro_loss:.6f} "
                f"lr={lr:.8f}"
            )

        if step % VAL_EVERY == 0:
            val_loss = eval_val_loss(translator, modelA, modelB, val_loader, device)
            print(f"[valid] step={step:6d} recon_loss={val_loss:.6f}")

    ckpt = {
        "translator_state": translator.state_dict(),
        "config": {
            "MODEL_A_NAME": MODEL_A_NAME,
            "MODEL_B_NAME": MODEL_B_NAME,
            "Q_DIM": Q_DIM,
            "ADAPTER_NUM_HEADS": ADAPTER_NUM_HEADS,
            "ADAPTER_HEAD_DIM": ADAPTER_HEAD_DIM,
            "TRANSLATION_DIM_FACTOR": TRANSLATION_DIM_FACTOR,
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
