# Latent Space Communication via K-V Cache Alignment (Toy)

논문: Latent Space Communication via K-V Cache Alignment  
- arXiv: https://arxiv.org/abs/2601.06123

## 간단 소개
이 레포는 위 논문의 핵심 아이디어(공유 latent space Σ를 통해 서로 다른 모델의 KV cache를 번역)를 Toy Example로 구현합니다.

- 서로 다른 아키텍처(레이어 수/차원이 다른) HuggingFace 사전학습 LM 2개 사용
  - 기본값:
    - Model A: openai-community/gpt2 (12 layers, 768 hidden)
    - Model B: openai-community/gpt2-medium (24 layers, 1024 hidden)
- 베이스 LM은 freeze, Cross-Attention 기반 어댑터만 학습
- 손실은 Suffix LM Loss 없이 KV reconstruction loss(MSE)만 사용
- 양방향 변환(A->B, B->A)을 동시에 학습

## 파일 구성
- train.py: KV 번역 어댑터 학습 (A->B + B->A, reconstruction loss only)
- infer.py : 학습된 어댑터로 KV 변환 추론 및 “translated cache vs true cache”의 next-token logits 유사도 출력

## 설치
```console
pip install torch transformers datasets
```

## 학습
shared_dim_q는 두 모델의 레이어 수(n_layer)로 나누어 떨어져야 합니다.
기본값 1536은 12와 24로 나누어 떨어집니다.

```console
python train.py \
  --max_steps 300 \
  --batch_size 2 \
  --seq_len 64 \
  --shared_dim_q 1536 \
  --save_path kv_align_adapters.pt
```

## 추론 (KV 변환 데모)
prefix에서 뽑은 KV를 번역해서 target 모델에 넣었을 때(pred)와,
원래 target 모델의 KV를 넣었을 때(true)의 next-token logits cosine similarity를 출력합니다.

```console
python infer.py \
  --ckpt kv_align_adapters.pt \
  --prompt "Paris is the capital of"
```

출력 예:
- [A->B] cosine(logits_pred, logits_true) = 0.XXXX
- [B->A] cosine(logits_pred, logits_true) = 0.XXXX

## 주의/한계 (Toy)
- 논문의 모든 학습 레시피/규제항/스케일을 완전 재현하지 않습니다.
- 목적은 “KV cache 정렬/번역이 가능한 구조 + bidirectional reconstruction”을 빠르게 실험하는 것입니다.
