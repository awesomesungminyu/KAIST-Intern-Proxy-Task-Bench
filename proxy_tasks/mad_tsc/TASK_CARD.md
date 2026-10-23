# mad_tsc — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | MAD selective copying baseline 설정 하나 (Poli et al. 2024 MAD, upstream `train.train` 호출) |
| Tier (문헌 기대 라벨) | Tier 1 win |
| 유형 | 합성 데이터 (고정 캐시), 시퀀스 |
| 입력 → 출력 | MAD `LanguageModel`: `tokens[B,L] → logits[B,L,V]`. 등록 단위는 MAD layer ID 목록(레시피) |
| 시퀀스 길이 | 256, 복사 토큰 16개 (`mad_tsc/train.py`의 `MADConfig`) |
| vocab | 16 |
| 지표 | `val_accuracy`, **높을수록 좋음**. split `reused_mad_test_development`, 선택 `final_epoch` |
| 예산 | full 최대 200 epoch (upstream이 정확도 0.999 이상에서 조기 종료 가능) / smoke `--smoke_test` (1 epoch) |
| 런타임 | `/data/envs/hexa-mad/bin/python` (torch 2.2.2, pytorch-lightning 등), CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`, 16종): `mh_gla_swiglu`, `mh_gla_moe`, `hyena_swiglu`, `hyena_moe`,
`mamba`, `mh_hyena_swiglu`, `mh_attention_moe`, `mh_attention_swiglu`, `mh_hyena_moe`, `striped_mamba_swiglu`,
`striped_mamba_moe`, `striped_hyena_swiglu`, `striped_mh_hyena_swiglu`, `striped_hyena_moe`, `striped_mh_hyena_moe`,
`striped_hyena_experts_moe`
기준 모델: `mh_attention_swiglu`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task mad_tsc --model mh_attention_swiglu
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task mad_tsc --model mh_attention_swiglu \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && /data/envs/hexa-mad/bin/python -m mad_tsc.train --help`

## 데이터

- 출처: MAD benchmark 캐시 중 `mad_tsc/data.py`의 `DATASET` 디렉터리 (train/test 각 `inputs.npy`, `targets.npy`). data seed 12345 고정이며 모델 seed와 독립이다.
- 다운로드: 자동 다운로드 없음. 이 서버에서는 `/data/mad-official-repro`의 캐시를 복사했다 ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)). 캐시가 없으면 실패한다.
- 라이선스: mad-lab 코드 MIT ([third_party/README](../../third_party/README.md)). 캐시 데이터의 라이선스 표기는 확인 불가.

## 알려진 한계

- MAD 전체 난이도·하이퍼파라미터 sweep이 아니며 Jelassi copy(`jelassi_copy`)도 아니다 (프로토콜 limitations).
- upstream test split을 Lightning이 반복 모니터링하므로 개발 데이터로 취급한다. 손대지 않은 final test가 없다 (프로토콜 limitations).
- 과거 공식 benchmark는 seed 하나였고 `striped_mh_hyena_moe`는 FAILED였다. 나머지 14종은 Turing에서 미검증이다 (프로토콜 limitations).
- FlashAttention을 PyTorch SDPA로 바꿨다 (`kernel_exact=false`). 필요한 CUDA extension이 없는 레시피는 실패할 수 있다 (프로토콜 limitations).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/mad_tsc-<model>/`

## 아키텍처 추가 규칙

레시피 등록은 `models.py`에서만 한다. 새 layer가 필요하면 `third_party/mad-lab`의 layer registry와 config에 추가하고
별도 패치로 보존해야 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#외부-구현을-쓰는-태스크)).
