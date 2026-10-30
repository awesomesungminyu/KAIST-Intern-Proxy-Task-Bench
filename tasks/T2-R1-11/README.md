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

```bash
python -m validation check --task T2-R1-11 --model cs_transformer
python -m validation run --task T2-R1-11 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T2-R1-11-smoke
```
