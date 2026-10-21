# Raw Speech Commands

모델: transformer, lstm. 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md)를 따른다.

데이터: {training,validation,testing}_{x,y}.npy. float32 [N,16000], 12-class 정수 label. train 36923 / val 4443 / test 4888.
기본 위치는 `$PROXY_DATA_ROOT/sc_raw`다. 실제 서버 복사 경로·체크섬은
[데이터 인벤토리](../../docs/DATA_INVENTORY.md)를 참고한다. 직접 CLI는 다운로드하지 않는다.

런타임: torch 2.5.1. 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

예산과 선택: batch 2 × 2048 updates, 최종 validation 정확도. seed로 섞은 4096개 예제를 중복 없이 사용 (epoch 0회).

```bash
# 저장소 루트, .env를 export한 뒤 환경/자산 점검 (학습 없음)
python -m validation check --task sc_raw --model lstm
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task sc_raw --model lstm \
  --profile smoke --seeds 0 --out artifacts/sc_raw-smoke
# 직접 CLI 도움말 (proxy_tasks/에서, 위 ML 인터프리터 사용)
python -m sc_raw.train --help
```

smoke는 성능 근거가 아니다. full은 `--profile full --seeds 0 1 2`로 명시한다.
결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.
`--data-root`는 validation의 태스크 폴더 override, `--data_dir`는 직접 trainer의 override다.

validation은 test를 평가하지 않는다. 직접 CLI의 `--final_test`는 후보를 확정한 뒤 최종 보고에만 사용한다. 저장된 checkpoint만 평가하는 전용 CLI는 아직 없다.

한계: 기존 transformer 8.3% / lstm 8.6%는 무작위 수준·budget_limited. 원 실행 subset seed는 0이고 새 CLI는 --seed를 쓴다. 입력 패치화나 다운샘플링 없음. 길이 16000 attention은 무겁다. 기존 파일럿은 이 프로토콜의 acceptance 결과로 가져오지 않는다.
