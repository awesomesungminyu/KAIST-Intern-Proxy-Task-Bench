> 자동화 진입점은 저장소 루트의 `python -m validation`이다 (13 task ID).
> 아래 직접 실행/레거시 스크립트 안내는 기존 실험을 보존한다.
> [모델 추가](../docs/ADDING_A_MODEL.md) · [데이터 준비](../docs/DATA_INVENTORY.md) · [공통 실행](../README.md).

# Proxy Tasks: RegBench / G2P (CMUDict) / ZINC-12K

세 개의 소규모 proxy 검증 과제 구현. Track 1 (RegBench, G2P, ZINC-12K) 재현용 코드.
대상 서버: TITAN RTX 24GB 1장 (sm_75, Turing) — bf16/FlashAttention-2 미지원.

## 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에 `torch`, `torch_geometric`을 포함. CUDA 버전에 맞는 torch를
먼저 설치해두는 걸 권장 (예: `pip install torch --index-url https://download.pytorch.org/whl/cu121`),
그 다음 `pip install -r requirements.txt`로 나머지 설치.

## 데이터 용량 (실측 기준, 다운로드 파일만)

| task | 다운로드 용량 | 비고 |
|---|---|---|
| RegBench | 0 (생성형, 파일 없음) | PFA를 코드로 즉석 생성 |
| G2P (CMUDict) | 약 3.5MB | `cmusphinx/cmudict` GitHub raw에서 확인 (3,618,488 bytes) |
| ZINC-12K | 약 59MB~380MB (로더에 따라 다름) | 아래 "ZINC 주의사항" 참고 |

### ZINC 주의사항

두 가지 official 경로가 있고 용량이 다름:
1. `benchmarking-gnns` 공식 스크립트: `data.dgl.ai`에서 12K 전용으로 이미 준비된
   `ZINC.pkl`을 받음 (문서 표기 58.9MB).
2. PyTorch Geometric `ZINC(subset=True)`: Dropbox의 전체 250K `molecules.zip`을
   통째로 받은 뒤 인덱스 파일로 12K만 필터링 (파일 자체 용량은 이 방식이 더 큼).

이 구현은 **PyG 경로**를 쓴다 (표준 라이브러리라 유지보수가 쉬움). Dropbox가 막힌
네트워크에서는 실패할 수 있으니, 막히면 `zinc/prepare_data.py --backend dgl`로
전환하는 옵션을 넣어뒀다 (아래 참고).

## 실행 순서 (권장)

```bash
# 0. GPU 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1. RegBench — 단독 실행으로 1 epoch 런타임 측정
python -m regbench.train --smoke_test          # 수십 초짜리 스모크 테스트
python -m regbench.train --seed 0 --model transformer

# 2. G2P
python -m g2p.prepare_data                      # 다운로드 + split 생성 (1회만)
python -m g2p.train --seed 0 --model transformer

# 3. ZINC-12K (가장 무거우니 마지막)
python -m zinc.prepare_data                      # PyG 다운로드 + 전처리 (1회만)
python -m zinc.train --seed 0 --model gine
```

또는 `bash scripts/run_smoke_tests.sh` 한 번으로 세 task 스모크 테스트를 순서대로 확인.

각 task의 `train.py`는 `--seed`, `--model {transformer,lstm}` (ZINC은 `{gcn,gine}`,
`gine`은 bond type edge feature까지 쓰는 GIN 변형, `gcn`은 edge feature를 무시하는
plain GCN),`--out_dir`을 받는다. 3-seed 반복은 쉘에서 `for s in 0 1 2; do ...; done`으로.

## 병렬 실행 (TITAN RTX 24GB 1장)

모델이 전부 작아서 (RegBench ~0.4M, G2P ~2M, ZINC ~500K 파라미터) GPU 메모리는
문제 없음. 병목은 GPU 유틸리제이션과 CPU dataloader일 수 있음. 권장 확인 순서:

1. 각 task를 **단독으로** 1 epoch 돌려서 런타임을 잰다 (`scripts/run_smoke_tests.sh` 또는
   위 실행 순서의 1회 실행).
2. 같은 task의 seed 3개를 `&`로 동시에 띄워보고, 순차 실행 대비 총 시간이
   줄어드는지 비교한다: `bash scripts/run_parallel_example.sh regbench transformer`
3. 이득이 있으면 seed들을 동시 실행, 없으면 순차 실행.

`nvidia-smi dmon`으로 GPU-Util을 보면서 판단하면 됨 (지속적으로 90%+ 면 병렬 이득 적음).

## Turing(sm_75) 주의사항

- bf16 하드웨어 미지원 → AMP는 `torch.cuda.amp.autocast(dtype=torch.float16)` +
  `GradScaler` 사용. 모델이 작으니 fp32로도 충분히 빠름 (기본값 fp32).
- FlashAttention-2 미지원 → `torch.nn.functional.scaled_dot_product_attention`
  (PyTorch 기본 SDPA, math/efficient 커널 자동 선택)만 사용, 별도 flash-attn 패키지 불필요.

## 검증되지 않은 부분 (직접 확인 필요)

- RegBench 하이퍼파라미터는 원 논문(Akyürek et al. 2024)과 후속 논문(TVP, 2025)의
  수치가 다름 (vocab/state/edge 수, 학습 DFA 풀 크기 고정 여부). 기본값은
  `regbench/pfa.py`의 `PFAConfig`에 두 세트를 주석으로 남겨뒀으니 팀 방침에 맞게 선택.
- G2P split은 Yao & Zweig(2015)/Yolchuyeva(2019) 논문의 정확한 split 파일을
  안정적으로 구할 수 없어서, 현재 유지보수되는 `cmusphinx/cmudict`에서
  **결정론적 랜덤 split(90/5/5, seed 고정)**으로 대체함. 논문 수치(106,837/2,670/12,000)와
  단어 집합이 정확히 같지 않을 수 있음 — 이 부분은 "정확한 재현"이 아니라
  "동등한 난이도의 재구성"으로 취급할 것.
