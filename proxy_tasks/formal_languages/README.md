# parity / tomita5 / tomita6 — 형식언어 인식

외부 저장소 Transformer-Formal-Languages (satwik77, EMNLP 2020, MIT) 를
고정 커밋으로 클론하고 패치해서 쓴다. 코드를 이 레포에 복사하지 않으므로
setup.sh 를 먼저 실행해야 한다.

## 실행

저장소 루트에서:

    ./setup.sh parity            # tfl 클론 + 패치 (tomita 도 같은 환경)
    ./run.sh parity  pilot       # 2-epoch 파일럿, 소요 시간 추정
    ./run.sh parity  full        # 전체 그리드
    ./run.sh parity  summary     # runs.jsonl 을 표로
    ./run.sh tomita5 full
    ./run.sh tomita6 full

## 구성

| 파일 | 역할 |
|---|---|
| runner.py | tfl main.py 를 감싸 시간과 결과를 runs.jsonl 로 기록 |
| run_*.sh | 그리드 정의 + 병렬 실행 (P 로 동시 실행 수 조절) |
| *_cmds.txt | 실제로 실행된 설정 목록 (재현용) |
| summarize.py | runs.jsonl 을 표로 |
| monitor*.py | 진행 상황 확인 |
| patches.md | tfl 수정 내역과 이유 |
| env.captured.txt | 실행 당시 tfl 커밋 해시와 pip freeze |

## 그리드

SAN(절대 PE / PE 없음), SAN-Rel(상대 PE), RNN(LSTM) 을
d_model 16·32 와 depth 1·2 조합으로. 100 epoch, RMSprop, batch 32.

## 주의

run_*.sh 기본값은 P=8 동시 실행이다. 학습 시간을 비교할 목적이면 P=1 로
단독 실행할 것 — 동시 실행 시 GPU 경합으로 시간이 크게 늘어난다
(저장소 루트의 environment.md 참조).
