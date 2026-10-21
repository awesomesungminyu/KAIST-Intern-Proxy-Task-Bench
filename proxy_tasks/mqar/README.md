# Multi-query associative recall

모델: transformer, linear_attention, gated_linear_attention, gqa2. 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md)를 따른다.

데이터: 입력 파일 없음. third_party/zoology의 MQAR 생성기: vocab 8192, L64, KV 8개, train 20000 / valid 2000, data seed 123 / 2147483789.
기본 위치는 `$PROXY_DATA_ROOT/mqar`다. 실제 서버 복사 경로·체크섬은
[데이터 인벤토리](../../docs/DATA_INVENTORY.md)를 참고한다. 직접 CLI는 다운로드하지 않는다.

런타임: /data/envs/hexa-mad/bin/python (torch 2.2.2, pydantic/wandb/einops 필요). 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

예산과 선택: 최대 30 epoch, validation accuracy >=0.99 조기 종료; 최고 validation 정확도.

```bash
# 저장소 루트, .env를 export한 뒤 환경/자산 점검 (학습 없음)
python -m validation check --task mqar --model transformer
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task mqar --model transformer \
  --profile smoke --seeds 0 --out artifacts/mqar-smoke
# 직접 CLI 도움말 (proxy_tasks/에서, 위 ML 인터프리터 사용)
python -m mqar.train --help
```

smoke는 성능 근거가 아니다. full은 `--profile full --seeds 0 1 2`로 명시한다.
결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.
`--zoology-root` / 직접 `--zoology_root`로 생성기 checkout을 지정한다. 데이터는 실행 시 생성한다.

한계: 기존 harness_validation은 4종 × 3 seed (456,789,101112). linear/gated는 30 epoch 안에 0.99 미달, GQA-2 seed 표준편차 약 45%p. 별도 test split 없음. mixer는 causal이고 위치 정보를 내부에서 제공해야 한다. 기존 파일럿은 이 프로토콜의 acceptance 결과로 가져오지 않는다.
