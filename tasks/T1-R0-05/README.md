# T1-R0-05 · Tomita 5

선정(교체). 사유: Modular Arithmetic 3시드 동률(LSTM seed 2 수렴 실패). 라운드0의9개 구조·3시드 결과를 재사용했다.

| 항목 | 내용 |
|---|---|
| Tier | 1 |
| 구분 | Transformer 패 |
| 입력 | 토큰 |
| 지표 | `mean_best_val_accuracy` ↑ |
| 평가 | `diagnostic_validation_bins` / `independent_best_epoch_per_bin` |
| 구현 | [proxy_tasks/formal_languages/runner.py](../../proxy_tasks/formal_languages/runner.py) |

실행 설정: [protocol.json](../../validation/protocol.json). 최종 점수: [v3 표](../../results/final_pool_table_20261031_v3.md).

## 데이터

[DATA.md](../../docs/DATA.md)의 추가 업로드 목록과 전처리 규칙을 따른다. 준비 후 `data/T1-R0-05/`에 `train_corpus.pk`, `val_corpus_bins.pk`가 필요하다.

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T1-R0-05
python -m validation check --task T1-R0-05 --model cs_transformer
python -m validation run --task T1-R0-05 --model cs_transformer --profile smoke --seeds 0
```

## 기존 결과 재사용

기준선: 0; 시드 수: 3; 판정: 선정(교체) / Transformer 패.

| 구조 | 원 점수 |
|---|---:|
| cs_transformer | 0.210747354 |
| cs_gqa | 0.203224206 |
| cs_linear_attention | 0.221709656 |
| cs_gla | 0.0786981922 |
| cs_retnet | 0.327314815 |
| cs_mamba | 0.310934744 |
| cs_s4d | 0.497580467 |
| cs_hyena | 0.126421958 |
| cs_lstm | 0.999669312 |
