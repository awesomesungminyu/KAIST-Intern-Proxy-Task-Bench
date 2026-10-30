# T2-R1-11 · Permuted MNIST

| 항목 | 내용 |
|---|---|
| Tier | 2 |
| 구분 | Transformer 승 |
| 입력 | 이미지 시퀀스 |
| 지표 | `accuracy` ↑ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/real.py](../../proxy_tasks/final16/real.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

[Drive 배포본과 압축 해제 안내](../../docs/DATA.md)를 따른다. 준비 후 `data/T2-R1-11/` 안에 `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`가 있어야 한다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T2-R1-11
```

## 실행

```bash
python -m validation check --task T2-R1-11 --model cs_transformer
python -m validation run --task T2-R1-11 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T2-R1-11-smoke
```
