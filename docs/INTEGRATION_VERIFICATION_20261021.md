# 태스크 편입 검증 기록 — 2026-10-21

학습은 실행하지 않았다. 신규 태스크와 Tier 0 편입, 데이터 준비, 명령·파일 계약을
검증한 기록이다. 신규 성능 결과나 acceptance 실행을 뜻하지 않는다.

## 실행 없이 검증한 범위

| 점검 | 결과 |
|---|---|
| `python -m unittest discover -s tests` | **38 tests, OK**, skip 없음 |
| `python -S -m validation list` | 표준 라이브러리만으로 13 task ID 출력 |
| 기존 7종 + ett/scifar/sc_raw/mqar/mad_tsc/tier0 `check` | **13/13 ready** (각 프로토콜 첫 번째 모델) |
| 신규 5종 + tier0 `--help` | 6/6 종료 코드 0 (MQAR/MAD는 hexa-mad interpreter) |
| `common.paths.task_data/task_output` 직접 호출 | 모든 13개 이름으로 임시 루트에서 성공 |
| protocol models ↔ registry | torch 없이 AST/torch-free TFL mapping으로 일치 확인 |
| ETT CSV loader | 고정 SHA256 통과. ETTh1 8449/2785/2785, ETTm1 34369/11425/11425, sample x/y 각각 (96,7) |
| scifar loader | 45000/5000/10000. raw sample (3,1024); normalize 후 모델 계약 (1024,3) |
| sc_raw mmap loader | training36923 / validation4443 / testing4888; x float32, sample(16000,1), label 범위0..11 |
| MQAR generation | hexa-mad torch2.2.2로 8예제, input/label (8,64) |
| MAD cache | train(12800,256), development(1280,256), input/target int64 |
| MAD baseline construction | mh_attention_swiglu 407824 parameters, CPU에서 모델 생성만 (학습 없음) |
| MAD wrapper plumbing | 실제 config/model + **모의 train 함수**로 seed7→고정 data seed12345 연결과 history JSON/epoch 확인 |
| MAD epoch parser | 저장된 attention 공식 benchmark 완료 CSV 396개를 읽기만 해서 epoch 범위1..200 확인. 새 acceptance로 가져오지 않음 |
| Tier0 모델 분리 | 11종 모두 seed7/d32/2layers에서 전후 state_dict SHA256 동일, 기존 상태 크기 추정식 동일 |
| 데이터 복사 | 전체 source/destination SHA256 일치. FFLM은 커밋된 manifest 전부 검증 |
| Python/shell/schema | AST parse, `bash -n`, Draft202012 result schema 검사 통과 |

공통 ML interpreter: `/opt/conda/bin/python` (torch2.5.1+cu124).
MQAR/MAD: `/data/envs/hexa-mad/bin/python` (torch2.2.2).
처음 테스트 프로세스에서 CUDA 초기화 실패가 한 번 있었으나 이후 신규 프로세스의
13개 preflight와 최종 unittest에서는 재현되지 않았다. 설치/환경 업그레이드는 하지 않았다.
MAD import는 선택적 `causal_attention_cuda` 부재를 출력하지만 attention 기본 모델
생성은 성공했다. 필요한 extension이 없는 다른 레시피의 실행을 보증하지 않는다.

## 실패 처리를 확인한 계약 테스트

불완전 예산과 초과 예산, 조기 종료 조건 미충족, 최종 test 평가 결과,
Tier0 seed/config 불일치·NaN, 서로 다른 runtime torch 버전, Tier0 지표 정의 불일치를
거부했다. ETT naive는 최대 counter0으로 기록한다. Lightning의 최종 validate에
붙는 epoch N을 추가 학습 epoch로 세지 않는다.

기존 subprocess 성공/실패/timeout, G2P 평가 모드 복원, TFL lengths CPU 호환,
FFLM manifest 복원·덮어쓰기 거부, provenance 비교 거부 조건도 통과했다.
모의 subprocess/모의 MAD 수치는 임시 디렉터리에서만 생성했으며 성능 근거가 아니다.

## 자료 위치와 한계

- 실제 데이터: `/data/proxy-task-bench/data` (원본 보존).
- 복사 체크섬: [data_checksums.json](data_checksums.json). FFLM은 package_manifest.json.
- preflight 원본 JSON: `artifacts/integration-checks/checks.json` (Git 제외).
- 사용 가이드: [ADDING_A_MODEL](ADDING_A_MODEL.md), [DATA_INVENTORY](DATA_INVENTORY.md).

`check`는 파일/런타임/해시 점검이며 forward/backward나 수렴의 보증이 아니다.
실제 smoke/full 학습, 전체 MAD 16종 실행, Tier0 점수 재계산은 수행하지 않았다.
기존 결과 수치는 바꾸지 않았고, 파일럿을 acceptance로 반입하지 않았다.
Tier 간 Spearman과 공통 아키텍처 집합은 보류했다.

이번 3개 커밋 범위의 `git diff --check`는 통과했다. 원격 대비 전체 범위에는
이전 커밋에서 원문 보존한 experiments/ 및 third_party/ patch의 trailing whitespace
경고가 남는다. 패치와 원본을 바이트 단위로 보존하기 위해 수정하지 않았다.
전체 미푸시 범위에서 50MiB 이상 blob이 없음을 확인했다.
