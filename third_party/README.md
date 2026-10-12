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
