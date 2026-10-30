# T2-R2-02 · Lipophilicity (SMILES)\*

| 항목 | 내용 |
|---|---|
| Tier | 2 |
| 구분 | Transformer 패 |
| 입력 | SMILES 토큰 |
| 지표 | `rmse_valid` ↓ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/lipophilicity.py](../../proxy_tasks/final16/lipophilicity.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 데이터 준비: [DATA.md](../../docs/DATA.md).

## 데이터

[Drive 배포본과 압축 해제 안내](../../docs/DATA.md)를 따른다. 준비 후 `data/T2-R2-02/` 안에 `Lipophilicity.csv`, `split.npz`가 있어야 한다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T2-R2-02
```

## 실행

```bash
python -m validation check --task T2-R2-02 --model cs_transformer
python -m validation run --task T2-R2-02 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/T2-R2-02-smoke
```
