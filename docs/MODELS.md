# 모델

공통 pre-RMSNorm + SwiGLU 백본에서 mixer만 바꾼 9종을 사용한다.

`cs_transformer`, `cs_gqa`, `cs_linear_attention`, `cs_gla`, `cs_mamba`, `cs_hyena`, `cs_lstm`, `cs_s4d`, `cs_retnet`

구현은 `proxy_tasks/common_set/`에 있다. 새 모델은 `mixers.py`의 factory와 `NAMES`, `validation/protocol.json`의 모델 목록에 등록한다.
Part 과제는 `proxy_tasks/final16/catalog.py`에서 생성하고 각 Part의 `build()`가 모델을 만든다. RegBench factory는 `proxy_tasks/T1_R0_07/models.py`, Parity adapter는 `proxy_tasks/common_set/tfl.py`다.

RegBench는 폭 128·8층·2헤드, Parity는 폭 32·2층·4헤드, 나머지 과제는 폭 128·2층·4헤드를 유지한다. 입력·출력 헤드와 평가 방법은 과제에 따라 다르다.
믹서 내부 폭은 Transformer 블록 파라미터 대비 ±10%에 맞춘다. 논문의 전체 설정을 재현한 대형 모델은 아니다.
RetNet은 CUDA에서 FLA 0.3.1 커널, CPU 검사에서 PyTorch reference를 사용한다. Hyena·S4D 원본 라이선스는 `proxy_tasks/common_set/licenses/`에 있다.
