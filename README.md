# proxy tasks — 구조 변경 검증용 소규모 벤치마크

Transformer 계열 구조 변경을 큰 모델로 검증하기 전에 빠르게 돌려볼 수 있는
6개 proxy 태스크를 하나의 실행 인터페이스로 모았다.

## EvoScientist 검증 인터페이스

자동 검증에는 `python -m validation`을 사용한다. 7개 task ID에 대해
평가 설정·입력 해시·소스 버전·실패 상태를 공통 JSON으로 기록한다.

```bash
python -m validation list
python -m validation check --task regbench --model transformer
python -m validation run --task regbench --model transformer \
  --profile smoke --seeds 0 --out artifacts/regbench-smoke
```

전체 실험은 `--profile full`을 명시해야 시작한다. smoke는 성능 근거가 아니다.
데이터·외부 코드 준비, 기준/후보 비교, EvoScientist 실행 지시와 평가 한계는
[연동 가이드](docs/EVOSCIENTIST.md)를 따른다.
기존 결과를 새 프로토콜의 실측 결과로 자동 가져오지 않는다.

## 기존 태스크 실행기

    ./setup.sh zinc              # 태스크 하나만 준비 (인자 없으면 전체)
    ./run.sh zinc gine 0         # <task> <model> <seed>

## 첫 실행

clone 직후에는 `<repo>/data`가 비어 있다. **`PROXY_DATA_ROOT`를 설정하고 그 아래에
데이터를 두지 않으면 g2p·zinc·fflm은 데이터를 찾지 못하고 경로와 함께 실패한다.**
regbench는 데이터를 실행 시 생성하므로 준비가 필요 없다.

1. 경로 설정 파일을 만든다. 값은 환경마다 바꿔야 한다.

       cp .env.example .env              # 값 수정
       set -a; source .env; set +a       # 자동으로 읽지 않으므로 셸에 export

2. `$PROXY_DATA_ROOT/<task>/`에 데이터를 둔다. 새로 받으려면 `./setup.sh g2p zinc`를 쓴다.
   이 서버에는 이미 받아 둔 데이터가 `<archive>/proxy_tasks/<task>/data/` 구조로 있으므로
   태스크별로 링크한다. 아카이브 경로를 `PROXY_DATA_ROOT`에 직접 넣으면 구조가 달라 찾지 못한다.

       A=/workspace/AI4Science_2026summer_archive/smyu/proxy_tasks
       mkdir -p "$PROXY_DATA_ROOT"
       ln -s "$A/g2p/data"  "$PROXY_DATA_ROOT/g2p"
       ln -s "$A/zinc/data" "$PROXY_DATA_ROOT/zinc"

3. FFLM은 `setup.sh`로 준비되지 않는다. 검증된 패키지에서 따로 복원한다
   (데이터는 `$PROXY_DATA_ROOT/fflm`, vendor는 `$PROXY_VENDOR_ROOT/fflm_vendor`로 간다).

       python scripts/restore_fflm_assets.py \
         --source /workspace/exp/2026-10-12/E3-FFLM/20261012T062553Z_26331

4. 확인한다.

       python -m validation check --task g2p --model transformer

## 데이터·산출물 경로

모든 경로는 `proxy_tasks/common/paths.py` 한 곳에서 정한다. 데이터가 수백 MB라
다른 디스크에 둘 수 있고 서버와 맥에서 위치가 다르므로 환경변수로 옮길 수 있다.
설정하지 않아도 기본값으로 동작한다.

| 환경변수 | 기본값 | 내용 |
|---|---|---|
| `PROXY_DATA_ROOT` | `<repo>/data` | 태스크별 입력 데이터 (`<root>/<task>/`) |
| `PROXY_OUTPUT_ROOT` | `<repo>/artifacts` | 체크포인트·history·로그 (`<root>/<task>/<model>-s<seed>/`) |
| `PROXY_VENDOR_ROOT` | `<repo>/third_party` | 외부 저장소·패키지 (`tfl/`, `fflm_vendor/`) |

`--root`, `--data_dir`, `--out_dir`처럼 명시적으로 넘긴 인자가 항상 우선한다.
FFLM은 데이터를 `$PROXY_DATA_ROOT/fflm`에, vendor 패키지를 `$PROXY_VENDOR_ROOT/fflm_vendor`에
둔다(`scripts/restore_fflm_assets.py`가 이 위치로 복원한다). 기존 패키지 폴더를
`--fflm-assets`로 직접 지정해도 된다.

알려진 미해결: `proxy_tasks/run_zinc_gps.sh`는 아직 `--out_dir runs/...`를 하드코딩해서
`proxy_tasks/runs/`에 쓴다. 경로 통합 작업이 zinc 관련 변경을 금지했기 때문에 고치지 않았다.

## 태스크

| 태스크 | 내용 | 실행 |
|---|---|---|
| `zinc` | ZINC-12K 분자 그래프 회귀 (GCN / GINE / GraphGPS) | `./run.sh zinc gine 0` |
| `g2p` | 자소→음소 변환 | `./run.sh g2p <model> 0` |
| `regbench` | 정규언어 인식 | `./run.sh regbench <model> 0` |
| `parity` | Parity 언어 인식 | `./run.sh parity full` |
| `tomita5` `tomita6` | Tomita 문법 5 / 6 | `./run.sh tomita5 full` |
| `fflm` | 형식언어 LM 재현 실험 | `./run.sh fflm` |
| `tier0` | Zero-cost proxy: 시퀀스 믹서 11종을 학습 없이 init 상태로 스코어링 | `cd proxy_tasks && python -m tier0.tier0_proxy` |

모델 선택지와 상세는 각 디렉터리의 README.md 참조. 태스크들을 Tier 0–3으로
묶는 구조와 아직 Tier 간 비교를 할 수 없는 이유는
[Tier 프레임워크](docs/TIER_FRAMEWORK.md)를 볼 것.

## 설계 원칙

- **태스크는 독립적이다.** 서로 import 하지 않고 `common/`(seed 고정, GPU 가드, 경로)만
  공유한다. 한 태스크만 떼어내 쓸 수 있다.
- **설치도 태스크 단위다.** `./setup.sh zinc` 는 ZINC 에 필요한 것만 준비한다.
- **데이터·체크포인트는 커밋하지 않는다.** `setup.sh` 와 각 태스크의
  `prepare_data` 로 복원한다.

## 구조

    .
    |- run.sh                    통합 실행 진입점
    |- setup.sh                  태스크별 환경·데이터 준비
    |- environment.md            검증된 환경, 학습 시간 해석 시 주의
    |- third_party/              외부 저장소 고정 커밋 + 패치 (클론본은 커밋하지 않음)
    |- data/                     입력 데이터 기본 위치 (커밋하지 않음)
    |- artifacts/                산출물 기본 위치 (커밋하지 않음)
    |- docs/                     연동 가이드, Tier 프레임워크
    `- proxy_tasks/
       |- common/                seed 고정, GPU 가드, 경로 (공용)
       |- zinc/ g2p/ regbench/   자체 구현
       |- formal_languages/      parity / tomita 하네스
       |- fflm/                  형식언어 LM 재현 패키지
       |- tier0/                 zero-cost proxy 하니스
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
