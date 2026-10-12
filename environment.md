# 실행 환경

## 검증된 환경

모든 실험 결과가 아래 환경에서 측정되었다.

| 항목 | 값 |
|---|---|
| GPU | NVIDIA TITAN RTX 24GB (sm_75, Turing) |
| CUDA | 12.4 (driver 575.64.03) |
| Python | 3.11.10 |
| PyTorch | 2.5.1+cu124 |
| PyTorch Geometric | 2.8.0.post1 |

## 로컬(macOS) 작업

setup.sh 는 macOS 에서 CPU/MPS 빌드 torch 를 설치한다. 코드 편집, 문서 작성,
결과 분석은 로컬에서 가능하지만 학습 실행은 CUDA 서버에서 한다.
결과는 git 으로 주고받는다.

## 재현성

common/seed.py 가 random / numpy / torch / torch.cuda 시드를 고정하고
cudnn.deterministic=True, cudnn.benchmark=False 를 설정한다
(속도보다 재현성 우선).

common/gpu_guard.py 는 GPU 를 못 잡으면 CPU 로 조용히 떨어지지 않고 즉시
종료한다. CUDA_VISIBLE_DEVICES 가 빈 문자열이면 실행을 거부하므로 run.sh 는
값이 없을 때 0 을 쓴다.

## 학습 시간 해석 시 주의

기록된 학습 시간(초)은 GPU 모델뿐 아니라 동시 실행 여부에 크게 좌우된다.

실측 예 (ZINC GINE, 동일 설정):

| 조건 | epoch 당 |
|---|---|
| 단독 실행 (200 epoch run) | 약 3.7 s |
| 여러 seed 동시 실행 (500 epoch run) | 약 9.9 s |

같은 코드, 같은 GPU 인데 2.7배 차이가 난다. 모델 간 학습 시간을 비교하려면
동일한 동시 실행 조건에서 측정한 값끼리 비교해야 한다.
formal_languages/run_*.sh 는 기본 P=8 병렬이므로 시간 비교 목적이면 P=1 로 둘 것.

측정 구간은 epoch 루프 직전부터 루프 종료 후까지다 (검증 평가 포함, 데이터
로딩과 모델 생성 제외). history.json 의 total_time_sec 는 최종 test 평가까지
포함한다.
