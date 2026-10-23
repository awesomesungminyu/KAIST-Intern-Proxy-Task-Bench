# ETT 예측

모델: linear (DLinear), transformer, naive. 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md)를 따른다.

데이터: ETTh1.csv / ETTm1.csv. SHA256은 data.py에 고정. 입력/예측 각각 96시점, 7채널, train 통계로 표준화.
기본 위치는 `$PROXY_DATA_ROOT/ett`다. 실제 서버 복사 경로·체크섬은
[데이터 인벤토리](../../docs/DATA_INVENTORY.md)를 참고한다. 직접 CLI는 다운로드하지 않는다.

런타임: torch 2.5.1. 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

예산과 선택: validation MSE 최소 checkpoint; 최대 20 epoch, patience 3. naive는 epoch 0.

```bash
# 저장소 루트, .env를 export한 뒤 환경/자산 점검 (학습 없음)
python -m validation check --task ett --model linear
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task ett --model linear \
  --profile smoke --seeds 0 --out artifacts/ett-smoke
# 직접 CLI 도움말 (proxy_tasks/에서, 위 ML 인터프리터 사용)
python -m ett.train --help
```

smoke는 성능 근거가 아니다. full은 `--profile full --seeds 0 1 2`로 명시한다.
결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.
`--data-root`는 validation의 태스크 폴더 override, `--data_dir`는 직접 trainer의 override다.

validation은 test를 평가하지 않는다. 직접 CLI의 `--final_test`는 후보를 확정한 뒤 최종 보고에만 사용한다. 저장된 checkpoint만 평가하는 전용 CLI는 아직 없다.

한계: 과거 linear/transformer 3 seed, naive 1 seed. ETTh1과 ETTm1을 섞지 않는다. validation 프로토콜은 ETTh1만 사용한다. 기존 파일럿은 이 프로토콜의 acceptance 결과로 가져오지 않는다.
