# Sequential CIFAR-10

모델: transformer, lstm. 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md)를 따른다.

데이터: cifar/cifar-10-batches-py/. train 45000 / val 5000 (split seed 42) / test 10000. RGB [1024,3], augmentation 없음.
기본 위치는 `$PROXY_DATA_ROOT/scifar`다. 실제 서버 복사 경로·체크섬은
[데이터 인벤토리](../../docs/DATA_INVENTORY.md)를 참고한다. 직접 CLI는 다운로드하지 않는다.

런타임: torch 2.5.1 + torchvision 0.20.1. 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

예산과 선택: 10 epoch 중 validation 정확도 최대 checkpoint.

```bash
# 저장소 루트, .env를 export한 뒤 환경/자산 점검 (학습 없음)
python -m validation check --task scifar --model transformer
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task scifar --model transformer \
  --profile smoke --seeds 0 --out artifacts/scifar-smoke
# 직접 CLI 도움말 (proxy_tasks/에서, 위 ML 인터프리터 사용)
python -m scifar.train --help
```

smoke는 성능 근거가 아니다. full은 `--profile full --seeds 0 1 2`로 명시한다.
결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.
`--data-root`는 validation의 태스크 폴더 override, `--data_dir`는 직접 trainer의 override다.

validation은 test를 평가하지 않는다. 직접 CLI의 `--final_test`는 후보를 확정한 뒤 최종 보고에만 사용한다. 저장된 checkpoint만 평가하는 전용 CLI는 아직 없다.

한계: 기존 결과는 budget_limited·미수렴: transformer 52.4%, lstm 31.7%. 이 수치는 새 acceptance 결과가 아니다. 기존 파일럿은 이 프로토콜의 acceptance 결과로 가져오지 않는다.
