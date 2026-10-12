# proxy tasks — 구조 변경 검증용 소규모 벤치마크

Transformer 계열 구조 변경을 큰 모델로 검증하기 전에 빠르게 돌려볼 수 있는
6개 proxy 태스크를 하나의 실행 인터페이스로 모았다.

## 빠른 시작

    ./setup.sh zinc              # 태스크 하나만 준비 (인자 없으면 전체)
    ./run.sh zinc gine 0         # <task> <model> <seed>

## 태스크

| 태스크 | 내용 | 실행 |
|---|---|---|
| `zinc` | ZINC-12K 분자 그래프 회귀 (GCN / GINE / GraphGPS) | `./run.sh zinc gine 0` |
| `g2p` | 자소→음소 변환 | `./run.sh g2p <model> 0` |
| `regbench` | 정규언어 인식 | `./run.sh regbench <model> 0` |
| `parity` | Parity 언어 인식 | `./run.sh parity full` |
| `tomita5` `tomita6` | Tomita 문법 5 / 6 | `./run.sh tomita5 full` |
| `fflm` | 형식언어 LM 재현 실험 | `./run.sh fflm` |

모델 선택지와 상세는 각 디렉터리의 README.md 참조.

## 설계 원칙

- **태스크는 독립적이다.** 서로 import 하지 않고 `common/`(seed 고정, GPU 가드)만
  공유한다. 한 태스크만 떼어내 쓸 수 있다.
- **설치도 태스크 단위다.** `./setup.sh zinc` 는 ZINC 에 필요한 것만 준비한다.
- **데이터·체크포인트는 커밋하지 않는다.** `setup.sh` 와 각 태스크의
  `prepare_data` 로 복원한다.

## 구조

    .
    |- run.sh                    통합 실행 진입점
    |- setup.sh                  태스크별 환경·데이터 준비
    |- environment.md            검증된 환경, 학습 시간 해석 시 주의
    |- third_party/              외부 저장소 고정 커밋 + 패치
    `- proxy_tasks/
       |- common/                seed 고정, GPU 가드 (공용)
       |- zinc/ g2p/ regbench/   자체 구현
       |- formal_languages/      parity / tomita 하네스
       |- fflm/                  형식언어 LM 재현 패키지
       |- results/<task>/        history.json
       `- analysis/              표·그래프 생성

## 결과 — ZINC-12K

train 10,000 / val 1,000 / test 1,000, Test MAE (낮을수록 좋음).
공통 설정: 4 layers, batch 128, lr 1e-3, AdamW, ReduceLROnPlateau
(factor 0.5, patience 10, min_lr 1e-5), L1 loss.
seed 0/1/2 의 mean ± std (ddof=0). Test MAE 는 validation MAE 가 가장 낮은
체크포인트로 측정하며, test 데이터로 하이퍼파라미터를 고르지 않는다.

| 모델 | 구조 | 파라미터 | 200 epoch | 500 epoch |
|---|---|---|---|---|
| GCN | GNN (edge feature 미사용) | 110,201 | 0.4300 ± 0.0034 | 0.4320 ± 0.0069 |
| GINE | GNN (edge feature 사용) | 163,791 | 0.3219 ± 0.0045 | 0.2820 ± 0.0182 |
| GraphGPS | GNN + Transformer 하이브리드 | 168,181 | 실험 진행 중 | 실험 진행 중 |

GraphGPS 는 순수 Transformer 가 아니다. 각 레이어가 local message
passing(GINEConv) 과 global multi-head self-attention 을 결합한다.
파라미터 예산을 GINE(163,791) 에 ±10% 안으로 맞추기 위해 hidden dimension 을
60 으로 두었다(GINE 은 110). 파라미터 수는 맞췄지만 표현 폭은 다르다 —
둘을 동시에 맞추는 것은 구조상 불가능하므로 해석 시 고려할 것.

학습 시간 비교 전에 environment.md 를 읽을 것. 동시 실행 여부에 따라
같은 설정에서도 epoch 당 시간이 2.7배까지 차이났다.

## 참고 문헌

- Dwivedi, Joshi, Laurent, Bengio, Bresson, *Benchmarking Graph Neural Networks*, arXiv:2003.00982
- Hu, Liu, Gomes, Zitnik, Liang, Pande, Leskovec, *Strategies for Pre-training Graph Neural Networks*, ICLR 2020
- Rampasek, Galkin, Dwivedi, Luu, Wolf, Beaini, *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022, arXiv:2205.12454
- Bhattamishra, Ahuja, Goyal, *On the Ability and Limitations of Transformers to Recognize Formal Languages*, EMNLP 2020, arXiv:2009.11264
