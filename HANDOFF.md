# HANDOFF — EvoScientist 태스크 편입 완료, 푸시 승인 대기

갱신: 2026-10-21. 저장소 `/workspace/proxy-task-bench`, 브랜치
`codex/evoscientist-validation`, 원격
`git@github.com:awesomesungminyu/KAIST-Intern-Proxy-Task-Bench.git`.

## 현재 상태

기존 인계의 5.1~5.7 구현·데이터·문서·검증 작업을 완료했다.
**학습은 한 번도 실행하지 않았다.** 실제 smoke/full acceptance는 수행하지 않았으며
모의 결과와 기존 파일럿을 성능 근거로 쓰지 않았다. 원격 푸시는 아직 하지 않았다.

원격 추적 브랜치의 마지막 커밋은 `c994656`. 미푸시 커밋은 기존 3개와 이번 3개다.
이번 커밋 중 `9eb8ba9`는 신규 태스크 5개, `af5208f`는 validation/Tier0다.
마지막 문서·경로 커밋은 `git log`에서 확인한다.

## 완료한 작업

- ett/scifar/sc_raw/mqar 초안 마무리 + mad_tsc 래퍼/레시피 16종 편입.
  mad_tsc는 MAD selective copying **baseline 설정 하나**이며 396-run sweep이나
  Jelassi copy가 아니다. 상류 train.train 호출, FP32/SDPA, kernel_exact=false.
  모든 모델 seed가 data seed12345 캐시를 쓰도록 run 내부 symlink로 연결한다.
- 모든 신규 태스크 README와 모델 계약 작성. `docs/ADDING_A_MODEL.md`에
  입력/출력, factory 인자, 최소 예제, 외부 패치와 구조적 제한을 정리했다.
- validation 13개 task ID 등록. 태스크별 runtime/torch 버전/추가 패키지 명시.
  mqar/mad_tsc 기본 Python은 `/data/envs/hexa-mad/bin/python` (torch2.2.2).
  나머지는 현재 interpreter (torch2.5.1). --python이 우선한다.
- native 지표/예산 하드코딩을 protocol budget_rule로 대체. early stop 조건,
  naive epoch0, test 평가 금지, 데이터/외부 source 경로 전달을 점검한다.
- Tier0 모델을 models.py로 분리. registry는 family/state_elements를 함께 받는다.
  validation은 outer seed 반복(기본0/1/2), child는 seed당 한 번만 점수화.
  full d128/4layers/L256/probe128/batch8, smoke --quick.
  compare는 8개 지표 각각 mean/population std만 반환하며 단일 개선량을 만들지 않는다.
- run.sh를 레거시로 표시하고 README/EVOSCIENTIST의 기본 진입점을 validation으로 변경.
  torchvision0.20.1/torchaudio2.5.1을 requirements/setup에 명시 (기존 설치 버전).
  패키지 설치는 하지 않았다.
- 승인된 `/data/proxy-task-bench/data/<task>`로 데이터 **복사**, 원본 보존.
  실제 합계 약8.80GiB + FFLM51MiB. TFL은 사용되는 3언어만 복사해 예상16GB보다 작다.
  MAD benchmark cache 전체 + 원본 data.zip도 복사했다. 모든 원본/사본 SHA256 일치.
  FFLM vendor는 third_party/fflm_vendor에 manifest로 복원했다.
- TFL source는 `/workspace/tfl`의 로컬 clone에서 pinned SHA를 checkout하고 tfl.patch 적용.
  validation staging은 source와 복사된 task별 corpus를 연결한다.
- `.env.example`과 로컬 `.env`에 `/data/proxy-task-bench/data` 지정.
  `.env`는 Git 제외이며 자동 로드하지 않으므로 실행 전 export한다.

## 검증 결과

- `python -m unittest discover -s tests`: **38 tests, OK**, skip 없음.
- `python -S -m validation list`: torch 없이 13개 task 출력.
- 각 태스크 첫 번째 프로토콜 모델 `check`: **13/13 ready** (기존7종 포함).
- 신규5종 + Tier0 `--help`: 6/6 성공. task_data/task_output은 13개 이름 모두 직접 호출 성공.
- ETT/CIFAR/raw/MAD 실제 loader/shape 확인, MQAR는 torch2.2.2에서 8예제 생성 성공.
- Tier0 11종 전후 가중치 SHA256와 상태 추정식 동일 (seed7, d32, 2layers).
- MAD 기본 attention 모델 407824 params 생성 성공. 래퍼는 실제 config/model과
  모의 train 함수로 점검했다 (optimizer/forward/backward/학습 없음).
- MAD CSV epoch parser가 학습 후 validate의 epoch를 추가 학습으로 세지 않도록 수정.
  기존 완료 CSV396개 읽기 점검 + 회귀 테스트. 기존 결과를 acceptance에 반입하지 않음.
- Python AST, shell syntax, result schema, git diff whitespace 확인.

세부 기록: `docs/INTEGRATION_VERIFICATION_20261021.md`.
체크섬/준비법: `docs/DATA_INVENTORY.md`, `docs/data_checksums.json`.
preflight 원본: `artifacts/integration-checks/checks.json` (Git 제외).

처음 프로세스에서 CUDA 초기화 실패가 있었으나 이후 13종 check와 최종 테스트에서는
재현되지 않았다. MAD의 선택적 causal_attention_cuda extension은 없으므로 일부 레시피는
실패할 수 있다. 전체 16종이나 실제 학습의 실행 성공/수렴을 검증한 것은 아니다.

## 유지한 결정과 한계

- ETT validation 기본은 ETTh1로 명시. ETTm1은 직접 CLI --dataset으로 지원하며
  ETTh1 acceptance에 섞지 않는다. 두 CSV 모두 복사·검증했다.
- scifar 기본10epoch, sc_raw2048update/epoch0; 기존 budget_limited/미수렴 상태 유지.
- MQAR는 실제로 4종×3seed 완료 이력이 있었음 (대부분1seed라는 설명은 틀림).
- MAD official의 두 번째 모델은 FAILED, seed1개12345. SDPA로 kernel_exact=false.
- Tier0는 v2. mlp_sal 상수, snip↔mix_nuc rho0.82, ntk_cond 좌표 안정성 미확인,
  mix_nuc는 DSS++ 아님, 무작위 토큰으로 과제 의존성 없음. v1 결과는 CPU였다.
- zinc는 그래프, g2p는 encoder-decoder라 범용 causal mixer를 그대로 꽂을 수 없다.
- fflm full step은 lstm500 / 그 외10000. ETT naive는 학습 예산0.
  예산이 다른 모델의 matched-budget compare는 거부된다.
- iclreg/Jelassi copy는 미구현으로 유지. 공통 아키텍처 집합과 Tier 간 rho는 보류.
- `proxy_tasks/run_zinc_gps.sh`의 레거시 runs/... 경로는 아직 하드코딩되어 있다.
- 모델/경로 변경으로 evaluation_sha256이 바뀌어 기존 smoke acceptance와 compare 불가.

## 다음 단계 — 푸시만 승인 필요

인계 지침: **푸시 전에 git log / git status /
`git diff --stat origin/codex/evoscientist-validation..HEAD`를 보여주고 승인받기.**
승인이 오면 아래 명령으로 해당 브랜치만 일반 푸시한다. force push/main merge 금지.

```bash
GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
  git push origin codex/evoscientist-validation
```

로컬 커밋 사용자 정보는 명령 단위로 지정한다.
`git -c user.name="Sungmin Yu" -c user.email=smyu9150@gmail.com commit ...`.

학습 실행, 기존 결과 수치 변경, 파일럿 acceptance 반입, 미구현 과제 작성,
공통 아키텍처 집합 작업, 데이터/50MB 이상 파일 커밋, force push, main 머지는 금지다.
