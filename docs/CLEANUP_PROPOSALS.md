# 정리 후보 제안서

2026-10-23, 커밋 `b45b67f` 기준 점검 결과다. **아무것도 실행하지 않았다.** 제안만 적는다.

## 전제

- 파일 이동, 이름 변경, 리팩터링은 공통 아키텍처 집합 브랜치(`codex/common-arch-set`, 각 태스크 `models.py` 수정 중)가
  머지된 뒤에 한다. 이 브랜치는 이 clone의 `git branch -a`에 보이지 않아 실제 diff를 확인하지 못했다.
  아래의 "충돌 가능성"은 그 브랜치가 `models.py`를 고친다는 사실만으로 판단했다.
- `proxy_tasks/common/*.py`, 태스크의 `models*.py`가 아닌 `.py`, `validation/*.py`, `validation/protocols/v1.json`을 바꾸면
  `evaluation_sha256`이 바뀐다 ([REPO_MAP](REPO_MAP.md#evaluation_sha256에-들어가는-파일)). 해시가 바뀌면 기존 결과와 `compare`할 수 없으므로
  그런 항목은 다음 프로토콜 버전(재기준 실행)과 묶어서 처리한다.
- 표기: **sha** = evaluation_sha256 영향, **충돌** = 공통 집합 브랜치와의 충돌 가능성, **작업량** = S(1시간 이내) / M(반나절) / L(하루 이상).

## 우선순위 상위 5개

| # | 제안 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| 1 | 오래된 문서 수치·상태 갱신 (task 수 13→15, 테스트 38→49, iclreg/jelassi_copy "미구현" 문구) | `HANDOFF.md`, `proxy_tasks/README.md` 1행, `docs/INTEGRATION_VERIFICATION_20261021.md` | 없음 | 없음 | S |
| 2 | 문서에 있으나 동작하지 않는 명령·옵션 정정 | `proxy_tasks/README.md`의 `zinc/prepare_data.py --backend dgl` (코드에는 `--root`만 있음), `--model {transformer,lstm}` / ZINC `{gcn,gine}` (`gps` 누락) | 없음 | 없음 | S |
| 3 | 신규 태스크를 태스크 표에 추가 (jelassi_copy, iclreg 행 누락) | `docs/EVOSCIENTIST.md` "태스크와 해석 범위" 표, `docs/DATA_INVENTORY.md` 표 | 없음 | 낮음 | S |
| 4 | `.gitignore`에 데이터·캐시 확장자 규칙 추가 (기존 규칙은 유지하고 추가만) | `*.npy`, `*.npz`, `*.pk`, `*.pkl`, `*.safetensors`, `*.bin`, `.pytest_cache/`, `.ipynb_checkpoints/`가 무시되지 않음 (`git check-ignore`로 확인). sc_raw·mad_tsc는 `.npy`, TFL은 `.pk`를 쓴다 | 없음 | 없음 | S |
| 5 | 공통 집합 머지 뒤 `TASK_CARD.md`와 기존 태스크 README 통합, README 형식 통일 | 13개 태스크 README가 두 가지 템플릿(초기 3개 vs 신규)과 영문(`fflm/README.md`)으로 나뉨. 카드와 README 내용 일부 중복 | 없음 | 있음 (README의 모델 목록) | M |

## 전체 목록

### A. 문서 불일치

| # | 내용 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| A1 | task 수·테스트 수가 오래됨 | `proxy_tasks/README.md` "13 task ID", `HANDOFF.md` "13개 task ID", "38 tests" (현재 15개, 49 tests) | 없음 | 없음 | S |
| A2 | iclreg/jelassi_copy를 "미구현"으로 기술 | `HANDOFF.md` "iclreg/Jelassi copy는 미구현으로 유지" (현재 `fb2bc74` 이후 구현·등록) | 없음 | 없음 | S |
| A3 | 존재하지 않는 CLI 옵션 | `proxy_tasks/README.md` → `zinc/prepare_data.py --backend dgl`. `zinc/prepare_data.py`의 인자는 `--root` 하나 | 없음 | 없음 | S |
| A4 | 모델 선택지 누락 | `proxy_tasks/README.md` "ZINC은 `{gcn,gine}`" (`gps` 등록됨) | 없음 | 낮음 | S |
| A5 | 신규 태스크 표 누락 | `docs/EVOSCIENTIST.md` 표 13행, `docs/DATA_INVENTORY.md` 표 13행 (jelassi_copy, iclreg 없음) | 없음 | 낮음 | S |
| A6 | 결과 보관 위치 안내가 실제와 다름 | `regbench/`, `g2p/`, `zinc/` README "../results/<task>/ 에 history.json 으로 보관". 실제 기본 출력은 `$PROXY_OUTPUT_ROOT/<task>/<model>-s<seed>` (`common/paths.py`), `proxy_tasks/results/`에는 `.gitkeep`만 있음 | 없음 | 없음 | S |
| A7 | 지표 설명 부재·불일치 | `regbench/README.md`, `g2p/README.md` "평가지표: 학습 로그 참조" (프로토콜에는 `next_token_acc`, `val_word_acc`). `zinc/README.md`는 직접 실행 기준 "Test MAE"를 먼저 적음 | 없음 | 없음 | S |
| A8 | 표준 라이브러리를 외부 패키지로 표기 | `g2p/README.md` "외부 패키지: `torch`, `urllib`" | 없음 | 없음 | S |
| A9 | 형식언어 레거시 실행 경로가 저장소 밖 파일에 의존 | `run.sh parity` → `run_parity_mini.sh`의 `RUNNER=${RUNNER:-$EXP_DIR/runner.py}` (기본 `/workspace/exp/2026-10-12/runner.py`), 저장소의 `formal_languages/runner.py`가 아님. `formal_languages/README.md` 그리드에 있는 SAN-Rel은 등록 모델이 아님 | 없음 (`.sh`, README) | 없음 | S |
| A10 | 루트 README의 "예제 모델"과 이번에 정한 기준 모델이 다름 | `README.md` 태스크 표 예제 모델 zinc=gine, fflm=lstm, ett=linear, sc_raw=lstm. 기준 모델은 zinc=gps, 그 외 transformer | 없음 | 낮음 | S |
| A11 | 의존성 표기가 두 벌 | `requirements.txt` (torch==2.5.1, torch-geometric==2.8.0) vs `proxy_tasks/requirements.txt` (torch>=2.1, torch_geometric>=2.5, tqdm) | 없음 | 없음 | S |
| A12 | 테스트 실행기 안내 | 문서·CI는 `python -m unittest discover -s tests`. 서버의 어느 Python 환경에도 pytest가 없음. pytest를 표준으로 쓰려면 dev 의존성 명시가 필요 | 없음 | 없음 | S |

### B. 중복 코드

| # | 내용 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| B1 | `Position`, `Encoder` 클래스가 바이트 단위로 동일 | `ett/models.py`, `scifar/models.py`, `sc_raw/models.py` (세 파일의 두 클래스 md5 일치) | 없음 (`models*.py`) | **높음** | M |
| B2 | `configure_backends()`가 세 trainer에 동일 | `ett/train.py`, `scifar/train.py`, `sc_raw/train.py` (md5 일치). `common/`으로 옮기면 모든 native 태스크 해시가 바뀜 | **있음** | 낮음 | S (재기준 실행 별도) |
| B3 | 이름은 같고 구현이 다른 블록 | `CausalSelfAttention` (`regbench`, `iclreg`, `mqar`), `RMSNorm` (`tier0`, `jelassi_copy`). 태스크 간 import 금지 규칙에 따른 의도적 중복. 공통 집합이 정해진 뒤 판단 | 없음 | **높음** | L |
| B4 | 형식언어 보조 스크립트 복붙 | `formal_languages/monitor.py` / `monitor_t5.py` / `monitor_t6.py` (실험 ID와 로그 glob만 다름), `run_tomita5.sh` / `run_tomita6.sh` (데이터셋 이름만 다름) | 없음 (formal 해시는 `runner.py`만 포함. source.sha256은 바뀜) | 없음 | S |
| B5 | 학습 스크립트의 seed별 사본 | `experiments/mqar/train_mqar_*_seed789.py`, `*_seed101112.py` 등. 원본 보존 목적이므로 삭제보다 현 상태 유지 권장 | 없음 | 없음 | — |

### C. 죽은 코드·사용되지 않는 파일

| # | 내용 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| C1 | 호출되지 않는 함수 | `zinc/models.py`의 `build_model()` (`zinc/train.py`는 `MODELS.build` 사용. regbench·g2p의 `build_model`은 사용 중) | 없음 | **높음** | S |
| C2 | 평가 해시 계산의 무효 분기 | `validation/runner.py` `evaluation_identity`가 `run_fflm.py` AST에서 `LSTM`, `model_for`를 빼지만 두 정의는 이미 `fflm/models.py`로 옮겨져 `run_fflm.py`에 없음 | **있음** (모든 태스크) | 없음 | S (재기준 실행 별도) |
| C3 | 레거시 실행·모니터 스크립트 | `run.sh` (7개 태스크만 지원), `proxy_tasks/run_zinc_gps.sh` (`runs/` 하드코딩), `proxy_tasks/mon_gps.sh` (`logs/zinc_gps`), `proxy_tasks/scripts/run_smoke_tests.sh`, `run_parallel_example.sh` (regbench/g2p/zinc 전용). `legacy/`로 옮기거나 문서에 레거시로만 남김 | 없음 | 없음 | S |
| C4 | 빈 결과 디렉터리 | `proxy_tasks/results/{fflm,formal_languages,g2p,regbench,zinc}/.gitkeep`만 있음. `.gitignore` 주석("results/ 의 history.json 만 커밋한다")과 함께 정리 | 없음 | 없음 | S |
| C5 | 저장소 밖 기본 경로 하드코딩 | `formal_languages/runner.py`, `monitor*.py`, `run_*.sh`의 `EXP_DIR` 기본값 `/workspace/exp/2026-10-12`, `REPO` 기본값 `/workspace/tfl`. `runner.py` 수정은 형식언어 3개 태스크 해시를 바꿈 | `runner.py`만 **있음** | 없음 | S |

### D. 태스크 디렉터리 구조 불일치

| # | 내용 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| D1 | 진입점 이름이 제각각 | `train.py` (10개), `fflm/run_fflm.py`, `tier0/tier0_proxy.py`, `formal_languages/runner.py`. validation 어댑터가 경로를 하드코딩하므로 이름을 바꾸면 `validation/runner.py`도 바뀜 | **있음** | 중간 | M |
| D2 | `__init__.py` 없음 | `fflm/`, `formal_languages/` | 없음 (fflm은 `run_fflm.py` AST, 형식언어는 `runner.py`만 해시. source.sha256은 바뀜) | 낮음 | S |
| D3 | 데이터 로더 위치가 다름 | `mqar/`에 `data.py` 없음 (생성기 호출이 `train.py` 안), `zinc/`는 `train.py` 안에서 PyG 로드, `fflm/`은 `run_fflm.py` 안 | **있음** | 낮음 | M |
| D4 | 디렉터리 하나가 task ID 세 개 담당 | `formal_languages/` ↔ `parity`, `tomita5`, `tomita6` | 이름 변경 시 **있음** | 낮음 | M |
| D5 | 모델 보조 파일 | `zinc/models_gps.py`만 분리되어 있음. `models*.py` 규칙 덕분에 해시에는 영향 없음. 공통 집합 이후 패턴 통일 여부 결정 | 없음 | **높음** | S |

### E. 네이밍 불일치

| # | 내용 | 근거 | sha | 충돌 | 작업량 |
|---|---|---|---|---|---|
| E1 | 같은 모델 ID, 다른 구현 | `transformer` (regbench `TinyGPT`, tier0, jelassi_copy Hard-ALiBi 등), `mamba` (tier0, jelassi_copy, mad_tsc 레시피), `lstm` | 프로토콜 ID 변경 시 **있음** | **높음** | 공통 집합 작업 범위 |
| E2 | `linear`의 의미가 태스크마다 다름 | ett `linear` = DLinear, mqar `linear_attention` | 프로토콜 변경 시 **있음** | **높음** | S |
| E3 | CLI 플래그 표기 혼재 | 밑줄형 `--data_dir`, `--max_len` (native trainer) vs 하이픈형 `--data-seed` (fflm), `--seq-len` (tier0), `--data-root` (validation) | 해당 trainer **있음** | 낮음 | M |
| E4 | 출력 폴더 이름 규칙이 세 가지 | `common/paths.py` `<model>-s<seed>`, validation native `<model>_seed<seed>`, ett 직접 실행 `<dataset>-<model>-s<seed>` | **있음** | 낮음 | M |
| E5 | 직접 CLI 기본 seed가 제각각 | 0 (대부분), mqar 456, mad_tsc 12345, fflm 1729, tier0 1234. validation은 0/1/2를 넘기므로 영향은 직접 실행에만 있음. 문서화만 권장 | 바꾸면 **있음** | 낮음 | S |

## 부록 1: 문서 간 모순 표 (2026-10-23 점검)

| 주제 | 문서 A | 문서 B / 코드 | 판단 |
|---|---|---|---|
| task ID 수 | `README.md`: 15개 | `proxy_tasks/README.md`, `HANDOFF.md`, `INTEGRATION_VERIFICATION_20261021.md`: 13개 | 코드·프로토콜은 15개. 인계·검증 문서는 그 시점 기록 |
| 테스트 수 | `HANDOFF.md`, `INTEGRATION_VERIFICATION_20261021.md`: 38 | `VERIFICATION_20261021.md`: 23 / 현재 실행: 49 | 시점 차이. 현재 49 |
| iclreg / jelassi_copy | `HANDOFF.md`: 미구현 | `README.md`, `TIER_FRAMEWORK.md`, 코드: 구현·등록 | 코드 기준 구현됨 |
| ZINC 모델 | `proxy_tasks/README.md`: gcn, gine | 프로토콜·레지스트리: gcn, gine, gps | 문서 누락 |
| ZINC 다운로드 대체 경로 | `proxy_tasks/README.md`: `--backend dgl` | `zinc/prepare_data.py`: 해당 옵션 없음 | 문서 오류 |
| 결과 보관 | regbench/g2p/zinc README: `../results/<task>/` | `common/paths.py`: `$PROXY_OUTPUT_ROOT` | 문서가 오래됨 |
| 형식언어 runner | `formal_languages/README.md`: 저장소 `runner.py`가 기록 | `run_*.sh`: `$EXP_DIR/runner.py` (저장소 밖) | 레거시 경로 의존 |
| 기본 의존성 | `requirements.txt`: torch==2.5.1 | `proxy_tasks/README.md`·`proxy_tasks/requirements.txt`: torch>=2.1, cu121 권장 | 두 벌 공존 |
| 태스크 표 | `README.md`: 15행 | `EVOSCIENTIST.md`, `DATA_INVENTORY.md`: 13행 | 신규 2개 누락 |

## 부록 2: 점검 기록 (수정 없음)

- `evaluation_sha256` (`b45b67f`, 15개 task): 작업 전후 동일. 계산 방법은 [REPO_MAP](REPO_MAP.md#evaluation_sha256에-들어가는-파일).
- 테스트: `CUDA_VISIBLE_DEVICES="" /opt/conda/bin/python -m unittest discover -s tests` → 49 tests, OK, skip 1
  (`test_torch_compat.PackedSequenceTests.test_cuda_lengths_work_without_changing_packed_values`, CUDA 필요로 미실행).
  pytest는 서버의 어느 환경에도 설치되어 있지 않아 같은 테스트를 unittest로 실행했다.
- 린터·포맷터: 저장소에 설정 파일(pyproject, setup.cfg, .flake8, ruff, pre-commit 등)이 없어 실행하지 않았다.
- `python -m validation check` 15개 task (기준 모델): 모두 데이터·외부 커밋·manifest 점검을 통과한 뒤
  `CUDA unavailable`에서 멈췄다. CUDA가 필요한 마지막 단계는 GPU 필요로 미실행.
- 직접 trainer `--help` 14개 (`mqar`, `mad_tsc`는 hexa-mad Python): 모두 종료 코드 0.
- Git 추적 파일 크기 상위: `docs/data_checksums.json` 161,054 B가 최대. 50MB 이상 파일 없음.
- 추적 중인 체크포인트·데이터·`__pycache__`·`.env` 파일: 없음. 로컬 `.env`는 무시 규칙으로 제외되어 있음.
- 실행하지 않은 것: [DATA_INVENTORY](DATA_INVENTORY.md)의 전체 체크섬 재검증 스크립트 (약 8.8 GiB 읽기, 공유 디스크 부하로 제외), smoke/full 학습.
