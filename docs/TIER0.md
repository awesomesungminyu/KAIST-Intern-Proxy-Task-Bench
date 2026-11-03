# Tier 0 — v4 최종 16개 CPU 재검증 (2026-11-03)

학습 없는 지표로는 구조 순위를 못 맞힘: 이번 16개 과제·9구조 조건에서 Bonferroni 보정 후 유의한 예측 지표는 0개였다. 지표별 평균 Spearman ρ 최고는 l2_norm의 +0.3054(약 0.31)다. 이 결론은 현재 학습 순위 스냅샷에 대한 것으로 모든 조건에서 예측이 불가능하다는 뜻은 아니다.

## 계산 범위와 지표

구조 9개 × 과제 16개 × 초기화 시드 3개(0·1·2), 총 432개 조합에서 다음 10종을 계산했다. GPU 및 학습 업데이트 없이 과제별 실제 학습 배치(B=16)를 사용했다. 구조·초기화 시드 간 배치 시드를 13으로 고정해 초기화 변동을 분리했다.

| 구분 | 지표 | 적용·제외 이유 |
|---|---|---|
| NAS-Bench-Suite-Zero 부분집합 | grad_norm, snip, synflow, jacov, nwot, l2_norm, plain, params, flops | 이전 구현·방향을 유지한 9종 |
| 추가 지표 | hidden_cov | 마지막 block 출력의 샘플 간 상관 고유값을 사용하는 기존 해석 구현 |
| 제외 | grasp, fisher | 이전 문서의 2차 미분·커널 호환성 사유 유지 |
| 제외 | epe-nas | 분류 라벨 전제로 모든 과제에 일관되게 적용하기 어려움 |
| 제외 | zen-score | BatchNorm을 사용하는 CNN 전제 |

nwot는 이전 naswot와 같은 구현이다. hidden_cov와 시퀀스 모델용 synflow 확장을 원논문 방법의 정확한 복제로 해석하지 않는다. flops는 지원 연산의 forward 계수이며 모든 사용자 정의 연산의 실제 하드웨어 연산량은 아니다. 모델 설정은 실제 과제 설정을 따랐다(RegBench d128·8층, Parity/Tomita 5 d32·2층, 나머지 d128·2층).

## 학습 후 순위와 비교

과제별 초기화 3시드 평균 지표를 구조 9개의 학습 후 점수 순위와 비교해 Spearman ρ 및 Kendall τ를 구했다. 학습 점수는 지표 방향을 맞췄다.

- A(확정, 5개): RegBench, MAD Compression, Parity, Tomita 5, enwik8. 9구조 모두 학습 3시드 평균을 사용했다.
- B(잠정, 11개): 나머지 과제. 구조별로 확보된 학습 시드의 평균을 사용했다. Transformer·LSTM 두 구조는 16개 모두 3시드가 있지만, 다른 7구조는 부족한 칸이 있다.

n=9에서 보정 전 |ρ| 임계값은 약 0.68~0.70이다. 정확 양측 순열검정(9!=362,880)을 사용하고, 통계량별로 지표 10종 × 과제 16개 = 160회 Bonferroni 보정했다. 보정 후 Spearman 및 Kendall 유의 칸은 모두 0개다. 전체 과제의 평균 ρ에는 잠정 B 순위가 섞여 있으며 평균 자체에 대한 별도 유의성 주장은 하지 않는다.

## 초기화 안정성과 결론

plain과 synflow는 각각 15/16 과제에서 초기화에 따라 구조 순위가 흔들렸다. 나머지 변동 과제 수는 grad_norm 7/16, snip 8/16, nwot 9/16, hidden_cov 10/16이다. 원 보고서에 사용한 변동 정의와 표준편차(ddof=0, 표본 표준편차도 별도 기록)를 유지한다.

이전 실행은 enwik8 입력에서 측정한 지표 벡터를 다른 과제에 재사용했다. 이번에는 v4의 실제 과제별 입력으로 다시 측정했지만, 보정 후 예측 지표가 없다는 결론은 유지됐다. 따라서 현재 하네스에서는 학습 없는 지표를 학습 후 구조 순위의 대체물로 사용하지 않는다. 9구조 전체의 3시드 결과가 완성되면 잠정 상관을 다시 확인해야 한다.

## 결과와 출처

[실행 보고서](../results/tier0_v4_20261103/REPORT.md), [상관 CSV](../results/tier0_v4_20261103/correlation.csv), [ρ 히트맵](../results/tier0_v4_20261103/rho_heatmap.png), [Transformer·LSTM 3시드 확인 표](../results/seed3_confirm_20261103/confirmation.md).

Krishnakumar et al., [NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies](https://papers.neurips.cc/paper_files/paper/2022/hash/b3835dd49b7d5bb062aecccc14d8a675-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2022 Datasets and Benchmarks Track. 이 논문은 지표 출처이며 위 v4 실험 결과의 출처는 이 저장소의 실행 보고서다.
