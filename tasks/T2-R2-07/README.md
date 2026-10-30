# T2-R2-07 · Beijing PM2.5\*

| 항목 | 내용 |
|---|---|
| Tier | 2 |
| 구분 | Transformer 승 |
| 입력 | 연속값 |
| 지표 | `mse_ctx_id50_long50` ↓ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/r2/forecast.py](../../proxy_tasks/final16/r2/forecast.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

```bash
python -m validation check --task T2-R2-07 --model cs_transformer
python -m validation run --task T2-R2-07 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T2-R2-07-smoke
```
