# 최종 v4 검증 — 2026-11-02

- 전체 테스트: `python -m unittest discover -s tests -v` — 17개 통과.
- CPU 1스텝: `python scripts/smoke_tasks.py --data-root /data/proxy-task-bench/final16-eval-data/data --tfl-root /workspace/AI4Science_2026summer_archive/sungminyu/proxy-bench/tfl --device cpu` — 16/16 통과. 유한 손실·기울기·optimizer 갱신 확인, 평가·체크포인트 없음.
- `python -m validation data --data-root /data/proxy-task-bench/final16-eval-data/data` — 16/16 ready.
- 16개 ID, 티어별 승 4·패 4, 생성형 5·파일 필요 11개. SNLI 제거, Lipophilicity 복구(82bf268), Speech Commands만 S5, EMNIST만 다양성 초과 예비.
- additions 압축 해제 후 4파일의 CSV 해시 일치, Tomita 5·EMNIST `prepare` 성공.
- small의 Lipophilicity.csv로 `prepare` 성공. 생성된 scaffold split은 서버 기존 split.npz와 인덱스까지 동일(3,360/420/420).
- 5개 묶음 `sha256sum -c SHA256SUMS.txt` 모두 OK. additions 31,513,733 B, SHA256SUMS.txt 518 B.
- 최초 테스트는 RDKit 미설치로 1건 실패. RDKit 설치 후 전체 재실행 통과. 실행 환경 PyTorch 2.5.1+cu124, NumPy 2.4.6, RDKit 2026.9.1. NumPy는 requirements.txt의 권장 범위(<2)와 달라 해당 권장 환경 전체 검증을 뜻하지 않는다.
- full 학습 실험·성능 재평가·push는 하지 않았다. Drive 파일은 서버 사본을 준비했으며 실제 업로드하지 않았다.
