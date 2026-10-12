# 레포 패치 (satwik77/Transformer-Formal-Languages, 2026-10-12)
1. src/components/sa_rnn.py 신규 스텁 — 레포에 파일 누락, SARNN 모델 전용(미사용)
2. src/components/transformers.py — TransfoXL import를 try/except로. transformers>=4.40에서 삭제됨. SAN-Rel(relative PE)만 사용 불가
3. src/model.py — .view( → .reshape( 2곳. torch 2.x 비연속 텐서 대응, 수치 동일
모델·손실·평가 로직 변경 없음
4. src/utils/helper.py — CUDA 없을 때 set_device 생략 (CPU 실행용, 장치 선택만 변경)
