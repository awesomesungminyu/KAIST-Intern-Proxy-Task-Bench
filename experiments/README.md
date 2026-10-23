# experiments — 탐색 작업 원본 보존

**이 디렉터리의 결과는 탐색 작업이며 성능 근거가 아니다. 정식 태스크는 `proxy_tasks/` 아래에 있다.**

리포 밖(특히 `/workspace` 밖의 `/data`)에 흩어져 있던 탐색용 스크립트와 설정을 유실 방지 목적으로
그대로 옮겼다. 원본 디렉터리 구조와 하드코딩 경로를 고치지 않았으므로 여기서 그대로 실행되지 않는다.
결과 파일·데이터·체크포인트·로그는 넣지 않았다. 결과는 아래 "원본 경로"에 남아 있다.

정식 태스크로 옮긴 것은 `proxy_tasks/<task>/` 에 있다. 이 사본은 그 이식이 원본과 같은지 대조하는 용도다.

## 목록

장치는 결과 파일의 `gpu` 필드가 아니라 코드와 로그로 확인했다.
"코드" = CUDA 가 없으면 실행을 거부하도록 짜여 있음. "기록" = 실제 사용한 `torch.device` 를 문자열로 저장함.
"로그" = 실행 로그에 GPU 사용 문구가 있음.

| 이름 | 원본 경로 | 실제 모델 | 결과 등급 | seed | 장치 | 알려진 한계 | 정식 태스크 |
|---|---|---|---|---|---|---|---|
| `mqar` | `/data/mqar` | transformer(RoPE), causal linear attention(ELU+1), gated linear attention(헤드별 forget gate), GQA-2 transformer | 4종 각각 full(최대 30 epoch, 정확도 0.99 도달 시 조기 종료) | 3 (456, 789, 101112) — `results/harness_validation` | GPU (기록: `device: cuda`) | 30 epoch 안에 linear·gated 는 0.99 미도달. GQA-2 의 seed 간 표준편차가 45%p 로 매우 큼. torch 2.2.2 환경 | `mqar` |
| `mqar` (hexa_t) | `/data/mqar/harness/hexa_t` | MAD 레시피 16종 정의, 실제 full 실행은 `mh_attention_swiglu`, `striped_mh_hyena_moe` | H1~H4 full | 3 | GPU (기록) | 16종 중 2종만 full. `mad-mqar-hexa_t.patch` 판본 사용 | 이식 안 함 |
| `mad_official_repro` | `/data/mad-official-repro` | MAD 공식 벤치마크: `mh_attention_swiglu`, `striped_mh_hyena_moe` | `mh_attention_swiglu` 완료(`mad_scores.csv`), `striped_mh_hyena_moe` **FAILED** (`status/FAILED`) | 1 (12345) | GPU (로그: Lightning `GPU available: True (cuda), used: True`) | FlashAttention 을 SDPA 로 바꿈(kernel_exact=false). selective copying 은 6개 MAD 과제 중 하나 | `mad_tsc` (t-SC 만) |
| `scraw` | `/workspace/exp/2026-10-16-scraw` | M5식 CNN | full 20 epoch | 1 (456) | GPU (코드) | 1 seed. `raw_cache_v1/*.npy`(2.8GB) 생성 스크립트 포함 | 이식 안 함 (학습 루프가 sc_raw 비교와 다름) |
| `quick_suite` | `/workspace/exp/2026-10-16-quick-suite` | ETT: DLinear / sCIFAR: sequential CNN | ETT 완료, sCIFAR `completed_time_limited` | 1 (456) | GPU (코드) | 시간 제한 실행 | 이식 안 함 (데이터 준비 스크립트만 참고) |
| `tier2_compare` | `/workspace/exp/2026-10-16-tier2-compare` | ETT: linear(DLinear), transformer, naive / sCIFAR: transformer, lstm | ETT 완료, sCIFAR `budget_limited` | ETT linear·transformer 3, naive 1 / sCIFAR 1 | GPU (코드) | sCIFAR 3·10 epoch 모두 미수렴 | `ett`, `scifar` |
| `tier2_followup30` | `/workspace/exp/2026-10-16-tier2-followup30` | SC-Raw: transformer, lstm | `budget_limited` | 1 (0) | GPU (코드) | 2,048 update(epoch 0회), 정확도 8.3%·8.6% 로 12-class 무작위 수준 | `sc_raw` |

## 넣지 않은 것

- 결과: `results/`, `runs/`, `status/`, `logs/`, `check-*/`, 요약·결과 JSON, 결과표 MD.
- MAD 단계별 학습 곡선 `results/*/logs/*/metrics.csv` 792개(합계 109MB). 원본 위치에만 있다.
- 외부 소스 사본: `/data/mqar/zoology-src`, `/data/mqar/vendor/mad-lab`, `/data/mad-official-repro/repo`.
  고정 커밋과 패치로 `third_party/` 에 기록했다.
- 데이터: SpeechCommands, sc_raw 캐시, CIFAR-10, ETT, MAD 벤치마크 데이터. `docs/DATA_INVENTORY.md` 참조.
