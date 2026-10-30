# T1-R2-01 · Positional Histogram

| 항목 | 내용 |
|---|---|
| Tier | 1 |
| 구분 | Transformer 승 |
| 입력 | 토큰 |
| 지표 | `slot_accuracy` ↑ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/generated/histogram.py](../../proxy_tasks/final16/generated/histogram.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

실행 시 고정된 생성기를 사용하므로 데이터 다운로드가 필요 없다.

## 실행

```bash
python -m validation check --task T1-R2-01 --model cs_transformer
python -m validation run --task T1-R2-01 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T1-R2-01-smoke
```
