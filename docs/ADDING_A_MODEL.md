# 새 아키텍처 추가하기

EvoScientist의 변경 지점은 각 태스크의 `models.py`다. 평가·데이터·옵티마이저는
그대로 두고 모델만 바꾼다. 같은 모델 이름의 구현을 후보 체크아웃에서 교체하거나,
`@register_model("새_이름")`을 추가한다. 새 이름은 `validation/protocols/v1.json`의
해당 `models`에도 등록한다. CLI 선택지는 레지스트리에서 자동 생성된다.
모델 목록을 바꾸면 프로토콜 해시도 바뀌므로 **기준과 후보 모두 같은 프로토콜로
새로 실행**해야 한다. 과거 파일럿을 새 acceptance 결과로 가져오지 않는다.

## 태스크별 계약

`B`: batch, `T`: 토큰/시점 수, `D`: hidden width, `V`: vocabulary.
정수 토큰은 int64, 연속 입력은 float32다. logits에 softmax를 적용하지 않는다.

| 태스크 / 등록 파일 (`proxy_tasks/` 아래) | 생성 함수 인자 | 입력 → 출력 / 추가 계약 |
|---|---|---|
| regbench / `regbench/models.py` | `vocab_size, max_len` | `forward(tokens[B,T]) → logits[B,T,V]`, causal. `num_params()` 제공 |
| g2p / `g2p/models.py` | `src_vocab, tgt_vocab, max_len` | `forward(src[B,S], tgt_in[B,T]) → logits[B,T,Vt]`; trainer가 greedy decoding 수행 (BOS/EOS/PAD=1/2/0). `num_params()` 제공 |
| zinc / `zinc/models.py` | `num_atom_types, num_bond_types, hidden_dim=None`, gps는 `use_lap_pe, lap_pe_k`도 전달 | PyG batch `Data(x, edge_index, edge_attr, batch, …) → prediction[B]`. `num_params()` 제공 |
| fflm / `fflm/models.py` | `smoke=False, flash=True` | `tokens[B,T] → logits[B,T,5]`; 토큰 w/r/i/0/1 |
| ett / `ett/models.py` | 없음 | 표준화된 과거 `[B,96,7] → 다음 96시점 [B,96,7]`; MSE |
| scifar / `scifar/models.py` | 없음 | train 통계로 정규화한 RGB raster `[B,1024,3] → logits[B,10]`; cross-entropy |
| sc_raw / `sc_raw/models.py` | 없음 | 1초 원 파형 `[B,16000,1] → logits[B,12]`; cross-entropy, 다운샘플링/패치화 없음 |
| mqar / `mqar/models.py` | `d_model, num_heads` | **mixer** `[B,T,D] → [B,T,D]`, causal 필수. 위치 정보는 mixer 내부에 구현 |
| tier0 / `tier0/models.py` | `d` | **mixer** `[B,L,d] → [B,L,d]`, causal. `family`, `state_elements(cfg)`도 등록 |
| parity / tomita5 / tomita6 / `formal_languages/models.py` | 이름과 TFL CLI 인자 목록 | TFL의 `src/components/` 모델 계약을 따름 (아래) |
| mad_tsc / `mad_tsc/models.py` | 없음 (MAD layer ID 목록 반환) | MAD `LanguageModel`: `tokens[B,L] → logits[B,L,V]`. MAD registry/config/recipe를 함께 사용 |

직접 실행은 `cd proxy_tasks && python -m <task>.train --help`로 확인한다.
모델 생성 시 다운로드·학습·외부 서비스 호출·전역 RNG 재설정을 하지 않는다.
고정 seed로 생성된 가중치, 학습 예산, split, 선택 지표가 비교의 기준이다.

## 최소 등록 예제

아래는 `ett/models.py`에 넣을 수 있는 최소 인터페이스 예제다. 좋은 예측 모델이라는
주장은 없으며, 등록 방법만 보여준다. `nn`과 `register_model`은 해당 파일에 이미 있다.

```python
class LastStepProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(7, 7)

    def forward(self, x):
        return self.proj(x[:, -1:, :]).expand(-1, 96, -1)

@register_model("last_step_projection")
def _last_step_projection():
    return LastStepProjection()
```

학습할 파라미터가 없는 ETT 모델은 `NO_TRAINING`에도 이름을 추가하고,
프로토콜 `budget_rule.no_training_models`에도 추가한다 (epoch 0).
naive는 실효 최대 예산 0으로 기록되어 학습 모델과의 matched-budget compare는 거부된다.
scifar/sc_raw도 인자 없는 생성 함수를 등록하되 위 표의 출력 shape를 지킨다.

regbench/g2p/zinc는 기존 factory와 같은 키워드 인자를 받아 새 클래스로 넘긴다.
예를 들어 regbench는 다음 형태다.

```python
@register_model("candidate")
def _candidate(vocab_size, max_len):
    return CandidateLM(vocab_size=vocab_size, max_len=max_len)
```

`CandidateLM` 구현은 같은 파일에 두고 `forward`와 `num_params()`를 제공한다.
G2P의 greedy decoding과 PAD/EOS 종료, 평가 전후 train/eval 모드 보존은
기존 trainer가 담당하므로 모델은 forward 계약을 유지한다. ZINC의 PyG 입력을 토큰 텐서로 임의 변환하면 같은 평가 계약이 아니다.

MQAR는 LM 전체 대신 mixer만 등록한다. 공통 `TransformerLM`이 embedding,
LayerNorm, GELU FFN, weight tying과 초기화를 담당한다.

```python
@register_model("candidate")
def _candidate(d_model, num_heads):
    return CandidateCausalMixer(d_model=d_model, num_heads=num_heads)
```

미래 토큰을 바꿔도 앞쪽 출력이 바뀌지 않는지 확인해야 한다. 일반 양방향 encoder를
그대로 넣으면 정답 누출이다. 원본 attention mixer는 RoPE를 내부에서 처리한다.

Tier 0에서는 상태 크기 분석식을 모델과 함께 등록한다.

```python
@register_model("candidate", family="rnn",
                state_elements=lambda c: 2 * c.d_model)
def _candidate(d):
    return CandidateRecurrentMixer(d)
```

분석식은 **레이어 하나의 상태 원소 수**다. evaluator가 layer 수와 dtype bytes(기존 v2는
2)를 곱한다. 실제 FP32 메모리 측정값이 아니다. LM/Block과 지표 구현은 고정한다.
validation은 seed당 한 번씩 점수화하고 기본 0/1/2를 반복한다. 기존 직접 CLI의
기본 seed는 1234/1235/1236이며 `--seed-start`로 바꿀 수 있다.

## 외부 구현을 쓰는 태스크

**parity/tomita**: `formal_languages/models.py`의 `register_model(name, tfl_args)`는
상류 CLI 인자 변환표다. 새 구조는 `third_party/tfl/src/components/`에 구현하고
`src/model.py`에서 선택하도록 연결한 뒤, 변경을 `third_party/tfl.patch`에 기록한다.
`src/model.py`가 바뀌면 evaluation 해시도 바뀌므로 기준/후보에 같은 dispatcher
변경을 적용해야 한다. TFL 모델 출력과 hidden-state 반환 방식은 해당 상류 구현을
따른다. 현재 공개 코드에 없는 SARNN은 사용할 수 없다.

**mad_tsc**: 새 layer를 `third_party/mad-lab/mad/model/layers/`에 구현하고
`mad/registry.py`의 `layer_registry`에 module/config/shorthand를 등록한다.
`configs/layers/<layer>.yml`도 추가하고 별도 `third_party` 패치로 보존한다.
`mad_tsc/models.py`에 layer ID의 순서 목록을 반환하는 레시피를 추가한다.
현 레시피 16개는 보존된 `experiments/mqar/configs/hexa_t/mad16_recipes.json`에서 옮겼다.
`mad.commit`의 고정 SHA와 SDPA 패치를 유지한다. 등록된 레시피라도 필요한 CUDA
extension이 없으면 실패할 수 있다. 모델 등록은 Turing 지원을 보장하지 않는다.

## 그대로 연결할 수 없는 경우

- ZINC는 그래프 입력이다. 시퀀스 mixer 하나를 직접 꽂을 수 없다.
- G2P는 encoder-decoder다. causal mixer 외에 encoder, decoder와 decoding 인터페이스가 필요하다.
- FFLM full step 예산은 `run_fflm.py`에서 lstm=500, 나머지=10000으로 정한다.
  새 이름을 추가하면 프로토콜 `steps`에도 같은 예산을 기록한다. 서로 다른 예산 비교는 거부된다.
- sc_raw의 길이는 16000이다. 메모리를 줄이려고 입력을 임의 패치화하면 평가 변경이다.
- 공통 아키텍처 집합과 Tier 간 Spearman 작업은 ρ 측정 시점까지 보류다.
  iclreg와 Jelassi et al. copy는 구현되어 있지 않다.

## 확인 순서와 결과 해석

```bash
python -m validation list
python -m unittest discover -s tests
python -m validation check --task ett --model candidate
# 학습 권한과 GPU 예산이 있는 세션에서만 실행:
python -m validation run --task ett --model candidate --profile smoke --seeds 0 \
  --out artifacts/ett-candidate-smoke
```

`check`는 파일·환경·해시를 점검한다. smoke는 실행 확인이며 성능 근거가 아니다.
학습을 허용하지 않은 세션에서는 `run`을 실행하지 않는다. full 결과만 동일 데이터,
프로토콜, evaluation 해시, 환경, 예산과 seed로 `compare`한다. Tier 0는 지표별
기술통계만 제공하고 단일 개선량을 계산하지 않는다. test는 후보를 확정한 뒤에만
평가하며, 이 인터페이스는 저장된 checkpoint만 읽는 최종 test CLI를 제공하지 않는다.
