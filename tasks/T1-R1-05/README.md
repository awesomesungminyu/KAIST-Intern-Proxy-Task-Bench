# T1-R1-05 · MAD Compression

| 항목 | 내용 |
|---|---|
| Tier | 1 |
| 구분 | Transformer 승 |
| 입력 | 토큰 |
| 지표 | `token_accuracy` ↑ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/synth.py](../../proxy_tasks/final16/synth.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

```bash
python -m validation check --task T1-R1-05 --model cs_transformer
python -m validation run --task T1-R1-05 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T1-R1-05-smoke
```
