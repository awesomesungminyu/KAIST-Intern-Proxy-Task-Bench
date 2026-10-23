# zinc — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | ZINC-12K 분자 그래프 회귀 |
| Tier (문헌 기대 라벨) | Tier 2 win |
| 유형 | 실제 데이터, **비시퀀스 (그래프)** |
| 입력 → 출력 | PyG batch `Data(x, edge_index, edge_attr, batch, …)` → `prediction[B]`. `num_params()` 제공 |
| 시퀀스 길이 | 해당 없음 (그래프) |
| vocab | 원자 종류·결합 종류 수를 데이터에서 읽어 `metadata.json`에 기록 (`zinc/prepare_data.py`). 현재 서버 기준 원자 21, 결합 4 |
| 지표 | `best_val` (validation MAE), **낮을수록 좋음**. split `validation`, 선택 `lowest_validation_mae_checkpoint` |
| 예산 | full 200 epoch, batch 128, lr 1e-3, `--validation_only` / smoke `--smoke_test` (1 epoch) |
| 런타임 | torch 2.5.1 + torch_geometric, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `gcn`, `gine`, `gps` (`gps` 구현은 `models_gps.py`)
기준 모델: `gps`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task zinc --model gps
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task zinc --model gps \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m zinc.train --help`

## 데이터

- 출처: PyTorch Geometric `ZINC(subset=True)` (`zinc/prepare_data.py`). 전체 250K `molecules.zip`을 받아 12K를 필터링한다.
- 다운로드: `cd proxy_tasks && python -m zinc.prepare_data`. 필요한 파일은 `metadata.json`, `subset/processed/{train,val,test}.pt` ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 라이선스: 확인 불가 (저장소에 기재 없음).

## 알려진 한계

- 구조 탐색 인터페이스는 validation MAE만 보고하며 test split을 로드·평가하지 않는다 (프로토콜 limitations).
- GCN은 결합 특징을 쓰지 않는다. GraphGPS는 GNN/attention 하이브리드다. 파라미터 예산은 자동으로 맞춰지지 않는다 (프로토콜 limitations).
- 그래프 입력이라 시퀀스 mixer를 그대로 꽂을 수 없다. 처리 방향은 미정이다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#2-zinc는-그래프-모델이라-시퀀스-믹서를-돌릴-수-없다)).
- `proxy_tasks/run_zinc_gps.sh`의 `runs/...` 출력 경로는 하드코딩된 레거시다 ([HANDOFF](../../HANDOFF.md)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/zinc-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다 (보조 구현 파일은 `models*.py` 이름을 쓴다. [ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
