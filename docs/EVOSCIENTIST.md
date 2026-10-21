# EvoScientist에서 프록시 검증 실행하기

이 저장소는 EvoScientist의 셸 실행 도구에서 호출하는 **독립 검증 저장소**다.
EvoScientist 내부 API나 에이전트 구조에 의존하지 않는다. 기준 모델과 후보 모델은
각각 소스 버전이 고정된 별도 체크아웃에서 실행하고, 결과 JSON을 비교한다.

EvoScientist venv와 GPU 학습용 Python은 분리한다. 서버에서 이미 검증한 ML
환경은 태스크별 protocol `runtime`이 선택한다. mqar/mad_tsc는 기존 torch 2.2.2
인터프리터를 쓰고 나머지는 현재 Python (torch 2.5.1)을 쓴다.
`--python /절대경로/python`으로 덮어쓸 수 있다. EvoScientist venv에 torch를
설치하거나 기존 서버 환경을 업그레이드할 필요가 없다.

## 공통 명령

저장소 루트에서 실행한다. `python -m validation` 자체는 표준 라이브러리만 사용한다.

```bash
python -m validation list
python -m validation check --task regbench --model transformer

# 먼저 동작 확인. smoke 결과는 성능 비교에 사용할 수 없다.
python -m validation run --task regbench --model transformer \
  --profile smoke --seeds 0 --timeout 180 \
  --out artifacts/regbench-baseline-smoke

# 평가 설정은 validation/protocols/v1.json으로 고정한다.
python -m validation run --task regbench --model transformer \
  --profile full --seeds 0 1 2 --timeout 3600 \
  --out artifacts/regbench-baseline-full

# 후보 체크아웃에서도 동일 태스크·프로토콜·데이터·seed로 실행한 뒤:
python -m validation compare \
  --baseline /absolute/baseline/artifacts/regbench-baseline-full \
  --candidate /absolute/candidate/artifacts/regbench-candidate-full
```

기존 출력 폴더 재사용은 거부한다. 새 폴더를 지정해야 이전 결과가 섞이지 않는다.
태스크는 순차 실행하며, 협조적 GPU 잠금과 실행 전 사용량 검사를 한다.
다른 프로그램이 사용하는 GPU를 중단하지 않는다. 잠금은 이 도구를 쓰는 같은
OS 사용자끼리 적용되며 GPU 사용량 검사는 완전한 자원 예약을 보장하지 않는다.

`check`는 파일·해시·런타임 점검이다. 실제 학습 가능 여부는 `smoke`로 확인한다.
전체 실험은 명시적으로 `--profile full`을 지정할 때만 시작한다. 기본 제한 시간은
seed당 1800초이며, 긴 실험에서는 예산에 맞게 `--timeout`을 지정한다.
시간 제한은 학습 프로세스와 그 자식에 적용된다. 입력 점검 시간은 포함하지 않는다.

## 태스크와 해석 범위

| ID | 모델 | 보고 지표 | 해석 |
|---|---|---|---|
| regbench | transformer, lstm | 마지막 step의 next-token accuracy | 반복 관측한 test stream이므로 개발용 지표 |
| g2p | transformer, lstm | 마지막 epoch의 validation word accuracy | CMUDict 재구성 split, 최종 test 아님 |
| zinc | gcn, gine, gps | 최저 validation MAE | trainer가 test split을 로드·평가하지 않음 |
| parity | san_abs, san_mask, lstm | bin별 최고 validation accuracy의 평균 | bin마다 다른 epoch일 수 있는 진단값 |
| tomita5 | san_abs, san_mask, lstm | 위와 같음 | 위와 같음 |
| tomita6 | san_abs, san_mask, lstm | 위와 같음 | 위와 같음 |
| fflm | transformer, lstm | ID read error rate | 개발용 ID만 평가하며 sparse/dense OOD는 남겨 둠 |
| ett | linear, transformer, naive | best val_mse (min) | ETTh1, 최대20 epoch/조기 종료, naive epoch0 |
| scifar | transformer, lstm | best val_accuracy (max) | 10 epoch, budget_limited |
| sc_raw | transformer, lstm | final val_accuracy (max) | 2048 updates, epoch0, budget_limited |
| mqar | transformer, linear_attention, gated_linear_attention, gqa2 | best_valid_accuracy (max) | 최대30 epoch, >=0.99 조기 종료, 별도 test 없음 |
| mad_tsc | MAD recipe 16종 | final val_accuracy (max) | 단일 baseline, upstream test는 개발 데이터 |
| tier0 | mixer 11종 | 5 proxies + 3 controls | 학습 없음, 지표별 기술통계만 |

숫자의 증감만 보고 자동으로 가설을 채택하지 않는다. `compare`는 기술통계만
출력한다. 3개 미만 seed는 pilot으로 표시하고, 단일 seed·smoke·시간 제한 종료를
반복 검증 완료로 취급하지 않는다. 서로 다른 태스크를 하나의 평균 점수로 합치지 않는다.
현재 인터페이스는 기존 baseline을 실행하거나 등록된 모델 구현을 교체하는 방식이다.
새 이름은 models.py 레지스트리와 프로토콜 models에 등록한다. CLI는 자동 갱신된다.
[태스크별 모델 계약과 예제](ADDING_A_MODEL.md)를 따른다.

## 자산 준비 및 서버에서 재사용

데이터·학습 결과·체크포인트·vendor 의존성은 Git에 넣지 않는다. 검증 시 실제
입력 파일의 SHA-256과 평가 코드의 해시를 기록한다. 자산이 없으면 다운로드나
학습을 임의로 시작하지 않고 경로와 함께 실패한다.

G2P/ZINC는 기존 준비 명령을 쓰거나 `--data-root`로 이미 준비한 폴더를 지정한다.
G2P는 train/val/test 단어 집합의 교집합도 검사한다.

```bash
python -m validation check --task g2p --model transformer \
  --data-root /workspace/AI4Science_2026summer_archive/smyu/proxy_tasks/g2p/data
python -m validation check --task zinc --model gine \
  --data-root /workspace/AI4Science_2026summer_archive/smyu/proxy_tasks/zinc/data
```

형식언어 태스크는 `third_party/tfl.commit`의 커밋과 패치, 준비된 pickle 데이터가
필요하다. 기존 `bash setup.sh parity tomita5 tomita6`는 소스만 준비하므로 데이터
유무를 별도로 확인한다. 데이터 생성법은 해당 upstream 문서를 따른다. 이 서버의
원본 준비본은 `/workspace/tfl`이며 실제 데이터명은 `Tomita-5`, `Tomita-6`다.
현재는 고정·패치된 소스를 third_party/tfl에 준비했고 데이터는
`$PROXY_DATA_ROOT/{parity,tomita5,tomita6}`에 복사했다. --tfl-root는 소스만,
--data-root는 해당 태스크의 복사 데이터 폴더를 선택한다.
검증 실행은 별도 staging 폴더에 소스·데이터를 연결하므로 기존 체크포인트를 덮어쓰지 않는다.
고정 upstream의 LSTM은 GPU 길이 텐서를 `pack_padded_sequence`에 넘기는 문제가
있다. 검증 runner는 해당 프로세스 안에서 길이 텐서만 CPU로 옮기는 호환 처리를
적용한다. 입력·가중치·packed 값은 바꾸지 않으며 공유 `/workspace/tfl` 소스도 수정하지 않는다.

```bash
python -m validation check --task tomita5 --model san_abs --tfl-root /workspace/tfl
```

FFLM은 `package_manifest.json`으로 고정된 data/vendor 56개 파일이 필요하다.
`setup.sh`가 이 데이터를 생성한다는 이전 안내는 잘못되었다. 저장된 패키지를
직접 참조하거나, 아래 스크립트로 전체 해시를 먼저 확인하고 복원한다.
변경된 대상 파일은 덮어쓰지 않는다. 복원할 패키지가 없는 새 서버에서는 먼저
동일 manifest에 맞는 자산을 확보해야 하며, 현재 자동 다운로드 도구는 제공하지 않는다.

```bash
python scripts/restore_fflm_assets.py \
  --source /workspace/exp/2026-10-12/E3-FFLM/20261012T062553Z_26331
python -m validation check --task fflm --model transformer

# 복사 없이 기존 패키지를 직접 사용할 수도 있다.
python -m validation run --task fflm --model lstm --profile smoke --seeds 0 \
  --fflm-assets /workspace/exp/2026-10-12/E3-FFLM/20261012T062553Z_26331 \
  --out artifacts/fflm-lstm-smoke
```

FFLM 검증 실행은 `--evaluation-splits id`를 지정해 ID만 평가한다. 기존
`run_fflm.py` 기본값과 `run.sh fflm`은 여전히 ID/sparse/dense 전체를 평가하므로
자동 구조 탐색에는 `python -m validation`을 사용한다.

FFLM full 예산은 LSTM 500step, Transformer 10000step이다. 따라서 두 구조의
직접 비교는 이 CLI에서 예산 불일치로 거부된다. 동일 구조의 기준/후보를 비교하거나,
예산을 바꾸는 연구라면 새 프로토콜 버전을 만든다. smoke는 축소 모델과 8개 평가
ID 시퀀스를 사용하므로 본 실험 성능으로 해석할 수 없다.

### 최종 테스트는 후보를 확정한 뒤

`full`은 고정된 전체 **학습 예산의 개발 검증**이다. 최종 테스트 완료를 의미하지 않는다.
평가할 후보의 소스 해시, 데이터 해시, 선택된 체크포인트, seed와 평가 횟수를
먼저 고정한 뒤 별도 최종 평가를 수행한다. ZINC의 `native/.../best_model.pt`,
FFLM의 `native/.../final.pt`가 저장된다. 기존 `run.sh`는 학습을 다시 시작하는
명령이므로 체크포인트 평가 전용 명령으로 사용하지 않는다. 이 버전에는 저장된
체크포인트만 읽는 최종 테스트 CLI가 포함되지 않는다.

## 결과 계약

`validation/result.schema.json`이 seed별 결과 형식을 정의한다.

```text
artifacts/<experiment>/
  summary.json
  <task>-<model>-s<seed>-<unique>/
    manifest.json       실행 명령·설정·소스/데이터 해시·환경
    result.json         done / failed / timeout, 실제 지표 또는 오류
    train.log           원본 stdout/stderr
    native/             원래 trainer의 history·체크포인트·결과
```

`wall_seconds`는 입력 점검과 프로세스 시작, 학습·평가·사후 검사를 포함한다.
순수 학습 시간으로 부르지 않는다. 원래 trainer가 따로 측정한 시간은 `native/`를
참조하며, 서로 다른 측정 구간의 시간을 비교하지 않는다.

비교는 같은 태스크·프로토콜 해시·평가 코드·데이터·환경·예산·seed 집합일 때만
허용한다. 모델 소스의 차이는 허용하지만 평가/데이터/옵티마이저 변경은 거부한다.
FFLM과 자체 태스크/Tier 0에서는 `models*.py`, 형식언어에서는
`src/components/`가 후보 구조 변경 지점이다. 나머지를 수정하는 연구는 새 프로토콜이 필요하다.
실행 중 소스나 입력이 바뀌면 결과를 실패로 처리한다.

## EvoScientist에게 줄 실행 지시 예시

아래 내용을 작업 지시로 사용하고 경로·모델·시간 예산을 채운다. EvoScientist가
실제로 접근할 수 있는 작업공간에 기준·후보 체크아웃과 데이터 경로를 먼저 제공한다.
실행 격리가 있는 환경에서는 외부 서버의 `/workspace`가 자동으로 보이지 않는다.
검증 실행기의 subprocess 로그와 JSON을 근거로 답하도록 한다.

> 이 저장소의 docs/EVOSCIENTIST.md와 validation/protocols/v1.json을 읽고 후보 구조를
> 검증하라. 지정된 ML Python으로 check 후 smoke를 실행하라. 승인된 GPU 시간 예산
> 안에서만 full을 실행하고, 기준/후보의 데이터·평가 코드·seed·학습 예산을 유지하라.
> 비교에는 validation compare만 사용하고, 실패·미측정·smoke를 성능 근거로 쓰지 마라.
> 원본 로그, result.json, 비교 JSON의 위치와 평가 한계를 함께 보고하라.
> 환경변수의 API 키를 출력하거나 저장소에 복사하지 마라.

EvoScientist 자체는 [공식 저장소](https://github.com/EvoScientist/EvoScientist)를 따른다.
이 변경은 EvoScientist 실행, LLM 호출 또는 전체 학습을 자동으로 시작하지 않는다.

## 검증 명령과 프로토콜 변경

```bash
python -m unittest discover -s tests -v
```

평가 방식·분할·예산을 바꿀 때는 `v1.json`을 덮어써서 과거 결과와 섞지 않는다.
새 JSON 및 버전을 만들고 실행기가 새 버전을 명시적으로 선택하도록 함께 변경한다.
이번 v1에서는 G2P greedy decoding이 호출 전 train/eval 모드를 복원하도록 수정했다.
기존 코드는 검증 도중 train 모드로 전환해 이후 배치에 dropout이 적용될 수 있었다.
따라서 예전 G2P 결과는 v1 기준 결과로 자동 재사용하지 않는다.
ZINC에는 검증 전용 옵션을 추가했고, FFLM에는 평가 split 선택 옵션을 추가했다.
기존 직접 실행기의 기본 전체 평가 동작은 유지한다.

## 신규 태스크 실행 계약

[데이터 인벤토리](DATA_INVENTORY.md)의 복사본을 .env로 지정한다. 새 패키지 설치는
필요하지 않다. MQAR/MAD의 --zoology-root / --mad-root는 고정 checkout을 덮어쓴다.

각 budget_rule은 counter key, full/smoke 예산, 조기 종료 허용/검증 조건, 지표 위치를
명시한다. 새 native trainer는 history.json 최상위 지표를 기록한다. validation은
ETT/MQAR의 정당한 조기 종료와 ETT naive(epoch0)를 허용하고 불완전 예산을 거부한다.
ETT/scifar/sc_raw의 --final_test는 검색 중 전달하지 않으며 test 평가 결과를 거부한다.

MAD는 upstream train.train을 그대로 호출하며 단일 selective-copying baseline만
실행한다 (공식 sweep 전체가 아님). 모든 모델 seed가 data seed12345 캐시를 쓰도록
run 내부에 symlink를 만든다. upstream test는 반복 모니터링하므로 개발 지표다.
MAD 레시피 추가는 상류 layer registry/config 및 패치가 필요하다. 등록된 16종 전체의
Turing 실행을 보증하지 않는다. SDPA로 kernel_exact=false다.

Tier 0는 outer runner가 seeds 0/1/2를 반복하고 child는 각각 한 번 실행한다.
full은 d128/4layers/L256/probe128/batch8, smoke는 --quick이다. compare는
지표 정의(name/direction/role)를 확인하고 지표별 평균/표준편차만 낸다.
mlp_sal 상수 등 v2의 한계를 프로토콜 limitations에 보존했다.

모델/경로·레지스트리 편입으로 evaluation_sha256이 바뀌었다. 기존 acceptance
(smoke)와 새 결과를 비교하거나, 보존한 파일럿을 acceptance로 반입하지 않는다.
