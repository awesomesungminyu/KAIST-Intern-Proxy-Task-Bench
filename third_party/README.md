# 외부 저장소

## Transformer-Formal-Languages

- 원저장소: https://github.com/satwik77/Transformer-Formal-Languages
- 라이선스: MIT
- 논문: Bhattamishra, Ahuja, Goyal, *On the Ability and Limitations of
  Transformers to Recognize Formal Languages*, EMNLP 2020, arXiv:2009.11264
- 고정 커밋: tfl.commit 참조

parity, tomita5, tomita6 가 이 저장소를 쓴다. 코드를 복사하지 않고 setup.sh 가
고정 커밋으로 클론한 뒤 tfl.patch 를 적용한다.

## 패치 내용 (tfl.patch)

모델, 손실, 평가 로직은 변경하지 않았다. 실행 환경 호환 목적의 수정만 있다.
자세한 내용은 ../proxy_tasks/formal_languages/patches.md 참조.

1. src/components/sa_rnn.py — 원저장소에 누락된 파일의 스텁 (SARNN 전용, 미사용)
2. src/components/transformers.py — TransfoXL import 를 try/except 로.
   transformers>=4.40 에서 제거됨. SAN-Rel(relative PE) 에만 영향
3. src/model.py — .view( 를 .reshape( 로 2곳. torch 2.x 비연속 텐서 대응, 수치 동일
4. src/utils/helper.py — CUDA 없을 때 set_device 생략 (CPU 실행용)

## zoology (HazyResearch)

- 원저장소: https://github.com/HazyResearch/zoology
- 라이선스: Apache-2.0
- 고정 커밋: zoology.commit 참조. 패치 없음.

mqar 가 MQAR 데이터 생성기(`zoology.data.multiquery_ar.MQARConfig`)만 쓴다.
setup.sh 가 `$PROXY_VENDOR_ROOT/zoology` 에 클론한다. zoology 패키지 import 가
pydantic·wandb·einops 를 요구하므로 mqar 는 이 의존성이 있는 별도 환경
(서버의 `/data/envs/hexa-mad`, torch 2.2.2)에서 돈다. 프로토콜의 `runtime` 항목 참조.

## mad-lab (MAD, Poli et al. 2024)

- 원저장소: https://github.com/athms/mad-lab
- 라이선스: MIT
- 고정 커밋: mad.commit 참조

mad_tsc 가 MAD 의 selective copying(t-SC) 과제·모델 레시피·학습기를 쓴다.
setup.sh 가 `$PROXY_VENDOR_ROOT/mad-lab` 에 클론하고 mad-turing-sdpa.patch 를 적용한다.

### 패치

- mad-turing-sdpa.patch — `mad/model/layers/attention.py` 한 파일.
  FlashAttention-2 가 Turing(sm_75)을 지원하지 않아 `flash_attn.modules.mha.MHA` 를
  PyTorch SDPA 로 바꿨다. 과제 설정·모델 폭·레이어 레시피·옵티마이저 sweep·평가는 그대로다
  (원 실행 기록: protocol_exact=true, kernel_exact=false, hardware_exact=false).
- mad-mqar-hexa_t.patch — 같은 파일을 다르게 고친 판본. 탐색 작업
  (experiments/mqar 의 hexa_t 하니스)이 쓴 것을 기록용으로 보존한다. 정식 태스크는 쓰지 않는다.
