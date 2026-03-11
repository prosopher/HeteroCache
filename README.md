# HeteroCache

## 설치
```console
pip install -r requirements.txt
```

## 학습
```console
python train.py heterocache
python train.py heterocache --default-config-path configs/train_heterocache_toy.json
python train.py heterocache --default-config-path configs/train_heterocache_toy.json --top-layers-to-translate 3
python train.py lsc
python train.py lsc --default-config-path configs/train_lsc_toy.json
```

## 평가
```console
python eval.py
python eval.py --default-config-path configs/eval_toy.json
```
