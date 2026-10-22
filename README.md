# proxy tasks — 구조 변경 검증용 소규모 벤치마크

EvoScientist가 제안한 아키텍처를 15개 task ID에서 검증하는 독립 저장소다.
모델 추가 방법은 [ADDING_A_MODEL](docs/ADDING_A_MODEL.md), 실제 데이터 경로와
체크섬은 [DATA_INVENTORY](docs/DATA_INVENTORY.md)를 따른다.

## 실행 인터페이스

저장소 루트에서 `python -m validation`을 사용한다. 실행 명령, 프로토콜,
evaluation_sha256, 소스·입력 해시, 환경, 실패 상태를 공통 JSON에 기록한다.

```bash
cp .env.example .env
# 경로를 확인하고 셸에 export (.env는 자동 로드되지 않음)
set -a; source .env; set +a
python -m validation list
python -m validation check --task ett --model linear
# 학습 권한과 GPU 예산이 있는 세션에서만 실행
python -m validation run --task ett --model linear \
  --profile smoke --seeds 0 --out artifacts/ett-smoke
```

full은 `--profile full --seeds 0 1 2`로 명시한다. smoke는 성능 근거가 아니다.
기존 파일럿과 acceptance 결과를 섞지 않는다. 실행·비교 계약과 해석 한계는
[EvoScientist 가이드](docs/EVOSCIENTIST.md)를 따른다. **`run.sh`는 레거시**이며
공통 provenance를 기록하지 않는다. 신규 자동화는 validation을 사용한다.

## 환경과 데이터 준비

공통 태스크는 torch 2.5.1 + CUDA, scifar는 torchvision 0.20.1을 사용한다.
requirements에는 기존 설치 버전의 torchvision/torchaudio도 고정했다.
`mqar`, `mad_tsc`는 프로토콜 runtime에 지정한 기존
`/data/envs/hexa-mad/bin/python` (torch 2.2.2)을 기본으로 쓴다.
`--python`으로 명시한 인터프리터가 우선하며 torch 버전과 CUDA를 점검한다.
EvoScientist 환경에 ML 패키지를 설치할 필요가 없다.

```bash
# 새 환경 준비가 필요한 경우에만 실행 (패키지 설치/다운로드 포함)
./setup.sh g2p zinc
# FFLM은 기존 검증된 패키지에서 복원한다
python scripts/restore_fflm_assets.py \
  --source /workspace/exp/2026-10-12/E3-FFLM/20261012T062553Z_26331
```

현재 서버에서는 `/data/proxy-task-bench/data/<task>/`로 데이터를 복사하고 원본은
보존했다. `.env.example`이 이 경로를 지정한다. 새 서버에서 데이터가 없으면
[인벤토리](docs/DATA_INVENTORY.md)의 태스크별 확보 절차를 따른다.
setup.sh는 TFL/zoology/MAD 소스를 고정 커밋으로 준비하지만 FFLM/TFL corpus,
ETT CSV, sc_raw 캐시를 자동 다운로드하지 않는다.

| 환경변수 | 코드 기본값 | 내용 |
|---|---|---|
| PROXY_DATA_ROOT | `<repo>/data` | 태스크별 입력 데이터. 이 서버 .env.example: `/data/proxy-task-bench/data` |
| PROXY_OUTPUT_ROOT | `<repo>/artifacts` | 실행 산출물 |
| PROXY_VENDOR_ROOT | `<repo>/third_party` | tfl, zoology, mad-lab, fflm_vendor |

경로는 `proxy_tasks/common/paths.py`가 정한다. validation의 `--data-root`는
해당 태스크 폴더 override다. 직접 trainer의 `--root`/`--data_dir`도 우선한다.
FFLM vendor는 데이터 루트와 분리하며 `--fflm-assets`로 원 패키지를 직접 쓸 수도 있다.
알려진 레거시 예외: `proxy_tasks/run_zinc_gps.sh`의 `runs/...` 출력 경로는 하드코딩되어 있다.

## 태스크

모든 예시는 저장소 루트의 `python -m validation check --task <ID> --model <이름>`에
쓸 모델 이름을 보여준다. 입력·출력 shape와 등록 위치는 [모델 계약 표](docs/ADDING_A_MODEL.md#태스크별-계약)를 따른다.

| 태스크 | 내용 / 등록 모델 | 필요한 데이터 | 예제 모델 |
|---|---|---|---|
| regbench | PFA stream; transformer, lstm | 실행 시 생성 | transformer |
| g2p | CMUDict encoder-decoder; transformer, lstm | vocab.json, train/val/test.tsv | transformer |
| zinc | 그래프 회귀; gcn, gine, gps | metadata.json, PyG processed 파일 | gine |
| parity | Parity; san_abs, san_mask, lstm | TFL train/validation corpus | san_abs |
| tomita5 | Tomita-5; 위와 같음 | TFL train/validation corpus | san_abs |
| tomita6 | Tomita-6; 위와 같음 | TFL train/validation corpus | san_abs |
| fflm | Flip-Flop LM; transformer, lstm | manifest와 일치하는 data/vendor | lstm |
| ett | 96→96 다변량 예측; linear, transformer, naive | ETTh1.csv (직접 CLI는 ETTm1도 지원) | linear |
| scifar | RGB sequential CIFAR-10; transformer, lstm | CIFAR-10 Python batches | transformer |
| sc_raw | 16000-sample Speech Commands; transformer, lstm | 원 파형 x/y.npy 캐시 | lstm |
| mqar | 8-pair associative recall; transformer, linear_attention, gated_linear_attention, gqa2 | pinned zoology 생성기 | transformer |
| mad_tsc | MAD selective copying; 레시피 16종 | MAD baseline 캐시 | mh_attention_swiglu |
| jelassi_copy | 문자열 전사 + 길이 일반화 (Jelassi et al. 2024); transformer (Hard-ALiBi), lstm, mamba | 실행 시 생성 | transformer |
| iclreg | in-context 선형 회귀, **연속 벡터 입력** (Garg et al. 2022); transformer, lstm | 실행 시 생성 | transformer |
| tier0 | 학습 없이 초기 모델의 다중 지표; mixer 11종 | 균일 랜덤 토큰 생성 | transformer |

신규 태스크는 `proxy_tasks/<task>/README.md`에 설정과 한계를 적었다.
mad_tsc는 jelassi_copy와 다른 과제다. jelassi_copy는 Tier 1 win 합격, iclreg는 등록했으나
Tier 1 win 미확정이다 (판정 수치: [Tier 프레임워크](docs/TIER_FRAMEWORK.md#tier-1-win-신규-편입-판정-2026-10-22)).
태스크 ID가 `copy`가 아닌 이유는 [jelassi_copy README](proxy_tasks/jelassi_copy/README.md)에 있다.
Tier 간 Spearman과 공통 아키텍처 집합 작업은 ρ 측정 시점까지 보류했다.
해결해야 할 구조적 문제는 [Tier 프레임워크](docs/TIER_FRAMEWORK.md)에 보존했다.

## 구조

```text
validation/                 공통 CLI, protocols/v1.json, 결과 schema
proxy_tasks/
  common/                   seed, CUDA 가드, 경로, 모델 레지스트리
  regbench/ g2p/ zinc/       자체 구현
  formal_languages/         parity/tomita 상류 연결
  fflm/                     재구성 Flip-Flop LM
  ett/ scifar/ sc_raw/       실제 데이터 태스크
  mqar/ mad_tsc/             합성 태스크, pinned 외부 코드
  jelassi_copy/ iclreg/      합성 태스크, 자체 구현 (실행 시 생성)
  tier0/                    mixer 레지스트리와 zero-cost evaluator
third_party/                커밋 SHA와 패치 (checkout/vendor는 Git 제외)
experiments/                원 실험 코드·설정 보존 (결과·데이터 제외)
docs/                       모델 추가, 데이터 인벤토리, 연동·Tier 가이드
artifacts/                  실행 산출물 (Git 제외)
run.sh                      레거시 실행기
setup.sh                    환경/소스/일부 데이터 준비
```

태스크는 독립적이며 common만 공유한다. 데이터·체크포인트·50MB 이상 파일은
커밋하지 않는다. 모델만 바꾼 비교에서도 파라미터 예산이 자동으로 맞춰지지 않는다.
아래는 기존 실험 기록이며 새 프로토콜의 acceptance 결과가 아니다.

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
