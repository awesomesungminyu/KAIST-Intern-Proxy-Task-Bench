# T1-R0-04 · Parity

| 항목 | 내용 |
|---|---|
| Tier | 1 |
| 구분 | Transformer 패 |
| 입력 | 토큰 |
| 지표 | `mean_best_val_accuracy` ↑ |
| 평가 | `diagnostic_validation_bins` / `independent_best_epoch_per_bin` |
| 구현 | [proxy_tasks/formal_languages/runner.py](../../proxy_tasks/formal_languages/runner.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

[Drive 배포본과 압축 해제 안내](../../docs/DATA.md)를 따른다. 준비 후 `data/T1-R0-04/` 안에 `train_corpus.pk`, `val_corpus_bins.pk`가 있어야 한다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T1-R0-04
```

## 실행

```bash
python -m validation check --task T1-R0-04 --model cs_transformer
python -m validation run --task T1-R0-04 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T1-R0-04-smoke
```
