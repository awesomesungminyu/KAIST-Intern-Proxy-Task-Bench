# T1-R0-07 · RegBench

| 항목 | 내용 |
|---|---|
| Tier | 1 |
| 구분 | Transformer 승 |
| 입력 | 토큰 |
| 지표 | `next_token_acc` ↑ |
| 평가 | `reused_test_stream` / `final_step` |
| 구현 | [proxy_tasks/T1_R0_07/train.py](../../proxy_tasks/T1_R0_07/train.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

실행 시 고정된 생성기를 사용하므로 데이터 다운로드가 필요 없다.

## 실행

```bash
python -m validation check --task T1-R0-07 --model cs_transformer
python -m validation run --task T1-R0-07 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T1-R0-07-smoke
```
