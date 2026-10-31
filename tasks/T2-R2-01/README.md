# T2-R2-01 · SNLI (byte-level)

예비(다양성 초과). 기존 seed0 결과를 그대로 재사용한다.

| 항목 | 내용 |
|---|---|
| Tier | 2 |
| 구분 | Transformer 패 |
| 입력 | 바이트 토큰 |
| 지표 | `accuracy` ↑ |
| 평가 | `validation` / `final_step` |
| 구현 | [proxy_tasks/final16/snli.py](../../proxy_tasks/final16/snli.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 최종 점수: [v3 표](../../results/final_pool_table_20261031_v3.md).

## 데이터

[DATA.md](../../docs/DATA.md)의 추가 업로드 목록과 전처리 규칙을 따른다. 준비 후 `data/T2-R2-01/`에 `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy`, `meta.json`가 필요하다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T2-R2-01
python -m validation check --task T2-R2-01 --model cs_transformer
python -m validation run --task T2-R2-01 --model cs_transformer --profile smoke --seeds 0
```

## 기존 결과 재사용

기준선: 0.33824426; 시드 수: 1; 판정: 예비 / Transformer 패.

| 구조 | 원 점수 |
|---|---:|
| cs_transformer | 0.53972769 |
| cs_gqa | 0.53850842 |
| cs_linear_attention | 0.4904491 |
| cs_gla | 0.50314975 |
| cs_retnet | 0.59825242 |
| cs_mamba | 0.63706565 |
| cs_s4d | 0.5966267 |
| cs_hyena | 0.65891081 |
| cs_lstm | 0.63300145 |
