# T2-R4-02 · Speech Commands (raw waveform)\*

S5 확정: Transformer/LSTM 3시드 승 방향 확인. 다른7개 구조는 기존 seed0 결과를 유지한다.

| 항목 | 내용 |
|---|---|
| Tier | 2 |
| 구분 | Transformer 승 |
| 입력 | 연속값 |
| 지표 | `accuracy` ↑ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/speech.py](../../proxy_tasks/final16/speech.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

[Drive 배포본과 압축 해제 안내](../../docs/DATA.md)를 따른다. 준비 후 `data/T2-R4-02/` 안에 `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy`가 있어야 한다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T2-R4-02
```

## 실행

```bash
python -m validation check --task T2-R4-02 --model cs_transformer
python -m validation run --task T2-R4-02 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T2-R4-02-smoke
```
