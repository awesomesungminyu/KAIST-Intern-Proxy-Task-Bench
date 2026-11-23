# 모델

공통 pre-RMSNorm + SwiGLU 백본에서 mixer만 바꾼 9종을 사용한다.

`cs_transformer`, `cs_gqa`, `cs_linear_attention`, `cs_gla`, `cs_mamba`, `cs_hyena`, `cs_lstm`, `cs_s4d`, `cs_retnet`

구현은 `proxy_tasks/common_set/`에 있다. 새 모델은 `mixers.py`의 factory와 `NAMES`, `validation/protocol.json`의 공통 및 과제별 모델 목록에 등록한다. 기준·후보 비교에서는 양쪽 프로토콜이 같아야 하므로, 새 이름을 추가했다면 양쪽에 동일하게 등록한다.
Part 과제는 `proxy_tasks/final16/catalog.py`에서 생성하고 각 Part의 `build()`가 모델을 만든다. RegBench factory는 `proxy_tasks/T1_R0_07/models.py`, Parity adapter는 `proxy_tasks/common_set/tfl.py`다.

RegBench는 폭 128·8층·2헤드, Parity는 폭 32·2층·4헤드, 나머지 과제는 폭 128·2층·4헤드를 유지한다. 입력·출력 헤드와 평가 방법은 과제에 따라 다르다.
믹서 내부 폭은 Transformer 블록 파라미터 대비 ±10%에 맞춘다. 논문의 전체 설정을 재현한 대형 모델은 아니다.
RetNet은 CUDA에서 FLA 0.3.1 커널, CPU 검사에서 PyTorch reference를 사용한다. Hyena·S4D 원본 라이선스는 `proxy_tasks/common_set/licenses/`에 있다.

## 임시 후보 플러그인

`python -m validation precheck --mixer /path/candidate.py:CandidateMixer` 또는 `suite --mixer ...`를 사용한다. 기존 9개 기준 모델의 `mixers.py`와 `protocol.json`은 수정하지 않는다. 임시 모델 ID는 파일 바이트와 클래스 이름의 SHA256 앞 8자리인 `cand_<hash>`다. 새 프로세스는 `PROXY_CANDIDATE_MIXER` 환경변수로 같은 후보를 등록한다.

```python
import torch.nn as nn
from proxy_tasks.common_set.mixers import CausalSelfAttention

class CandidateMixer(nn.Module):
    def __init__(self, d_model, num_heads, l_max):
        super().__init__()
        self.core = CausalSelfAttention(d_model, num_heads)
    def forward(self, x):
        return self.core(x)  # float32 [B,T,D] -> [B,T,D]
```

생성자 인자 3개는 키워드로 전달한다. mixer 내부의 위치 정보·인과적 상태만 후보가 담당하며 RMSNorm, SwiGLU, residual, embedding, readout, 깊이, 데이터, 예산은 하니스가 유지한다. 단일 파일에 보조 코드를 포함하고, 외부 파일·네트워크·API·전역 RNG 재설정에 의존하지 않는다. precheck가 임의 import 코드를 보안 격리해 주지는 않는다.

RegBench d128·8층·2헤드, Parity/Tomita 5 d32·2층·4헤드, 나머지 d128·2층·4헤드를 그대로 따른다. **현재 저장소의 파라미터 매칭은 mixer 단독이 아니라 동일 MLP·RMSNorm을 포함한 Block 파라미터 ±10%다.** 후보에도 같은 규칙을 적용하며 precheck.json에 Block 비율과 mixer 단독 비율을 모두 기록한다. mixer-only ±10%를 충족했다는 주장과 구분한다.

예시는 [candidate_mixer.py](../examples/candidate_mixer.py), 출처 있는 대체 아이디어와 대조군은 [research_mixers.py](../examples/research_mixers.py)다. 인과성 검사는 여러 길이·절단점·train/eval 모드의 유한 표본 검사이며 모든 입력에 대한 수학적 증명은 아니다. CPU precheck 실패는 suite를 차단한다. GPU 최대 길이 메모리/속도 검사는 CUDA 복구 후 별도로 측정한다.
