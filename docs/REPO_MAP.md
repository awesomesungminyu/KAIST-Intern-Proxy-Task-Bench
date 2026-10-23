# 저장소 지도

디렉터리별 역할과 지켜야 할 규칙을 모은 문서다. 2026-10-23, 커밋 `b45b67f` 기준으로 작성했다.
태스크별 요약은 [TASK_INDEX](TASK_INDEX.md)를 본다.

## 먼저 읽을 주의 두 가지

1. **`proxy_tasks/common/*.py`와 각 태스크의 `train.py`(그리고 `models*.py`가 아닌 모든 `.py`)는 `evaluation_sha256`에 들어간다.**
   공백, 주석, docstring, import 순서, 포맷터 적용만 바꿔도 해시가 바뀌고, 이전 결과와 `compare`할 수 없게 된다.
   아키텍처 변경은 `models.py`(또는 `models_*.py`)에서만 한다.
2. **데이터, 실행 결과, 체크포인트는 Git에 커밋하지 않는다.** 데이터는 `$PROXY_DATA_ROOT`,
   결과·체크포인트는 `$PROXY_OUTPUT_ROOT`(기본 `artifacts/`)에 둔다. 50MB 이상 파일도 커밋하지 않는다.

## 최상위 구조 (깊이 3, Git 추적 파일 기준)

```text
.github/workflows/validation.yml   CI: 표준 라이브러리 계약 테스트 + validation list
.env.example                       PROXY_* 경로 예시 (.env는 자동 로드되지 않음)
README.md                          실행 인터페이스, 환경, 태스크 표, 구조
HANDOFF.md                         2026-10-21 인계 기록 (일부 수치·상태는 이후 커밋보다 오래됨)
environment.md                     검증 환경과 학습 시간 해석 주의
requirements.txt                   루트 의존성 (torch 2.5.1 고정)
run.sh                             레거시 실행기 (provenance 기록 없음)
setup.sh                           venv, 외부 소스 고정 커밋 checkout, 일부 데이터 준비
docs/                              모델 추가, 데이터 인벤토리, EvoScientist 연동, Tier 프레임워크, 검증 기록, 이 지도
validation/                        python -m validation (list/check/run/compare)
  runner.py                        실행기와 해시(evaluation/source/data) 계산
  torch_compat.py                  TFL LSTM 길이 텐서 CPU 호환 처리
  protocols/v1.json                태스크별 모델·지표·예산·한계 (평가 계약)
  result.schema.json               seed별 result.json 형식
proxy_tasks/                       정식 태스크. 작업 디렉터리로 쓰인다 (cd proxy_tasks && python -m <task>.train)
  common/                          seed, CUDA 가드, 경로, 모델 레지스트리 (모든 native 태스크가 공유)
  <task>/                          태스크별 코드 (아래 표)
  results/                         태스크 5개의 .gitkeep만 있음 (history.json은 커밋되어 있지 않음)
  scripts/                         regbench/g2p/zinc용 레거시 스모크·병렬 예시 스크립트
  requirements.txt                 초기 3개 태스크용 의존성 (루트 requirements.txt와 버전 표기가 다름)
  run_zinc_gps.sh, mon_gps.sh      ZINC GraphGPS 레거시 실행·모니터 (runs/ 경로 하드코딩)
scripts/restore_fflm_assets.py     FFLM data/vendor를 manifest 해시로 검증 후 복원
tests/                             unittest 5개 파일, 49개 테스트 (CI는 unittest discover)
third_party/                       외부 저장소의 고정 커밋(*.commit)과 패치(*.patch)만 추적. checkout은 Git 제외
experiments/                       탐색 작업 원본 보존. 그대로 실행되지 않으며 성능 근거가 아님
  mqar/                            MQAR·flip-flop·parity·wikitext2 탐색 스크립트, harness, configs, 환경 스냅샷
  mad_official_repro/              MAD 공식 벤치마크 재현 스크립트와 패치
  quick_suite/, tier2_compare/, tier2_followup30/, scraw/   ETT·sCIFAR·SC-Raw 파일럿 원본
```

Git에서 제외되는 주요 경로: `artifacts/`, `/data/`, `proxy_tasks/*/data/`, `third_party/{tfl,zoology,mad-lab,fflm_vendor}/`,
`proxy_tasks/fflm/{vendor,runs,smoke_runs,benchmark_runs}/`, `proxy_tasks/{runs,logs}/`, `.env`, `__pycache__/`.

## 태스크 디렉터리 구성

| 디렉터리 | task ID | 모델 등록 | 데이터 로더 | 학습/실행 진입점 | 설정 위치 | 문서 |
|---|---|---|---|---|---|---|
| `regbench/` | regbench | `models.py` | `data.py`, `pfa.py` (생성) | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `g2p/` | g2p | `models.py` | `data.py`, `prepare_data.py` (다운로드) | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `zinc/` | zinc | `models.py`, `models_gps.py` | PyG `ZINC` (`train.py` 안), `prepare_data.py` | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `formal_languages/` | parity, tomita5, tomita6 | `models.py` (TFL 인자표) | 상류 TFL | `runner.py` (상류 main 래퍼), `run_*.sh` | 프로토콜, `*_cmds.txt`, `patches.md` | README, TASK_CARD |
| `fflm/` | fflm | `models.py` | `run_fflm.py` 안 (생성 + 고정 평가 배열) | `run_fflm.py`, `start_server.sh` | `package_manifest.json`, 프로토콜 | README(영문), TASK_CARD |
| `ett/` | ett | `models.py` | `data.py` | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `scifar/` | scifar | `models.py` | `data.py` | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `sc_raw/` | sc_raw | `models.py` | `data.py` | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `mqar/` | mqar | `models.py` | 없음 (`train.py`가 zoology 생성기 호출) | `train.py` | `train.py`의 `CONFIG` + 프로토콜 | README, TASK_CARD |
| `mad_tsc/` | mad_tsc | `models.py` (MAD 레시피) | `data.py` (캐시 이름, 로그 파서) | `train.py` (상류 `train.train` 호출) | `train.py`의 `MADConfig` + 프로토콜 | README, TASK_CARD |
| `jelassi_copy/` | jelassi_copy | `models.py` | `data.py` (생성) | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `iclreg/` | iclreg | `models.py` | `data.py` (생성) | `train.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |
| `tier0/` | tier0 | `models.py` | `tier0_proxy.py` 안 (랜덤 토큰) | `tier0_proxy.py` | CLI 인자 + 프로토콜 | README, TASK_CARD |

`__init__.py`는 `fflm/`, `formal_languages/`를 제외한 태스크 디렉터리에 있다.

## evaluation_sha256에 들어가는 파일

`validation/runner.py`의 `evaluation_identity`가 태스크별로 계산한다. 아래 파일 중 하나라도 바이트가 바뀌면 해시가 바뀐다.

| 대상 | 들어가는 파일 |
|---|---|
| 모든 태스크 | `validation/protocols/v1.json`, `validation/*.py` |
| native / mad / tier0 어댑터 (regbench, g2p, zinc, ett, scifar, sc_raw, mqar, iclreg, jelassi_copy, mad_tsc, tier0) | `proxy_tasks/<task>/*.py` 중 이름이 `models`로 시작하지 않는 파일 (`train.py`, `data.py` 등), `proxy_tasks/common/*.py` |
| mqar 추가 | 고정 zoology checkout의 `zoology/data` |
| mad_tsc 추가 | 고정 mad-lab checkout의 `train.py`, `mad/data`, `mad/model/pl_model_wrapper.py`, `mad/metrics.py`, `mad/configs.py`, `mad/paths.py`, `configs/tasks` |
| fflm | `proxy_tasks/fflm/run_fflm.py`의 AST (최상위 `LSTM`, `model_for` 정의 제외) |
| parity / tomita5 / tomita6 | `proxy_tasks/formal_languages/runner.py`, 고정 TFL checkout의 `src/**/*.py` 중 `components/` 밖 파일 |

`models*.py`는 해시에서 빠진다. 별도로 기록되는 `source.sha256`(source_identity)은 `proxy_tasks/`와 `validation/`의 모든 `.py`를 포함하며,
`compare`는 모델 소스 차이만 허용하고 평가·데이터·옵티마이저 차이는 거부한다 ([EVOSCIENTIST](EVOSCIENTIST.md#결과-계약)).

해시를 확인할 때는 레포의 함수를 그대로 쓴다.

```bash
# 저장소 루트에서. 외부 checkout은 PROXY_VENDOR_ROOT 또는 기본 third_party/를 쓴다.
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import argparse, json
from validation import runner
vendor = runner.proxy_paths.THIRD_PARTY
for task in sorted(json.loads(runner.PROTOCOL.read_text())["tasks"]):
    args = argparse.Namespace(task=task, tfl_root=vendor / "tfl",
                              zoology_root=vendor / "zoology", mad_root=vendor / "mad-lab")
    print(task, runner.evaluation_identity(args))
PY
```

## 데이터·결과·체크포인트 규칙

- 데이터: `$PROXY_DATA_ROOT/<task>/` (코드 기본값 `<repo>/data`, 이 서버 `.env.example`은 `/data/proxy-task-bench/data`). 커밋하지 않는다.
  복사 체크섬은 [data_checksums.json](data_checksums.json), 준비법은 [DATA_INVENTORY](DATA_INVENTORY.md).
- 실행 결과: `python -m validation run --out artifacts/<새 폴더>`. 기존 출력 폴더 재사용은 거부된다. `artifacts/`는 커밋하지 않는다.
- 체크포인트: `*.pt`, `*.pth`, `*.ckpt`는 `.gitignore`로 제외한다.
- 외부 코드: `third_party/`에는 `*.commit`과 `*.patch`만 커밋한다. checkout과 vendor는 `setup.sh`, `scripts/restore_fflm_assets.py`로 복원한다.
- 비밀정보: `.env`, `.env.*`는 제외하고 `.env.example`만 커밋한다.
- `.gitignore`가 다루지 않는 데이터 확장자(`*.npy`, `*.pk` 등)가 있으므로, 데이터를 저장소 안에 둘 때는 `data/` 아래에 둔다
  ([CLEANUP_PROPOSALS](CLEANUP_PROPOSALS.md) 참조).

## 진입점 정리

| 목적 | 명령 | 비고 |
|---|---|---|
| 태스크 목록 | `python -m validation list` | torch 없이 동작 |
| 환경·자산 점검 | `python -m validation check --task <ID> --model <이름>` | CUDA가 보여야 통과 |
| 실행 | `python -m validation run ... --profile smoke\|full --seeds ... --out ...` | 학습 권한과 GPU 예산이 있는 세션에서만 |
| 비교 | `python -m validation compare --baseline ... --candidate ...` | 같은 프로토콜·평가 해시·데이터·예산·seed일 때만 |
| 계약 테스트 | `python -m unittest discover -s tests -v` | CI와 같은 명령 |
| 레거시 | `./run.sh`, `proxy_tasks/scripts/*.sh`, `formal_languages/run_*.sh` | provenance 미기록. 신규 자동화에 쓰지 않음 |
