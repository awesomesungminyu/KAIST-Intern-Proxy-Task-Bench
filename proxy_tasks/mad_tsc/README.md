# MAD selective copying

모델: MAD 레시피 16종 (models.py 참고). 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md)를 따른다.

데이터: MAD benchmark 캐시 중 data.py의 DATASET 디렉터리. train/test 각 inputs.npy, targets.npy; data seed 12345 고정. 모델 seed와 독립.
기본 위치는 `$PROXY_DATA_ROOT/mad_tsc`다. 실제 서버 복사 경로·체크섬은
[데이터 인벤토리](../../docs/DATA_INVENTORY.md)를 참고한다. 직접 CLI는 다운로드하지 않는다.

런타임: /data/envs/hexa-mad/bin/python (torch 2.2.2, pytorch-lightning 2.2.1 등). 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

예산과 선택: MAD upstream train.train을 감싼 baseline 하나: vocab16, L256, copy16, train12800/dev1280, dim128, lr5e-4, wd0, batch128, 최대200 epoch, FP32. 마지막 epoch 개발 정확도. smoke는 같은 데이터로 1 epoch.

```bash
# 저장소 루트, .env를 export한 뒤 환경/자산 점검 (학습 없음)
python -m validation check --task mad_tsc --model mh_attention_swiglu
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task mad_tsc --model mh_attention_swiglu \
  --profile smoke --seeds 0 --out artifacts/mad_tsc-smoke
# 직접 CLI 도움말 (proxy_tasks/에서, 위 ML 인터프리터 사용)
python -m mad_tsc.train --help
```

smoke는 성능 근거가 아니다. full은 `--profile full --seeds 0 1 2`로 명시한다.
결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.
`--data-root`는 validation의 태스크 폴더 override, `--data_dir`는 직접 trainer의 override다.

`--mad-root` / 직접 `--mad_root`로 pinned MAD checkout을 지정한다. baseline 캐시가 없으면 실패한다. 기존 로그를 재사용하지 않는다.

한계: 396-run MAD sweep이나 Jelassi copy가 아니다. upstream test는 반복 모니터링한 개발 데이터다. 정확도 >=0.999 시 upstream이 조기 종료할 수 있다. 과거 공식 benchmark는 mh_attention_swiglu 완료 / striped_mh_hyena_moe FAILED, seed12345 하나. 나머지 14종은 Turing 미검증. SDPA 패치로 kernel_exact=false. 필요한 CUDA extension이 없는 레시피는 실패할 수 있다. 기존 파일럿은 이 프로토콜의 acceptance 결과로 가져오지 않는다.
