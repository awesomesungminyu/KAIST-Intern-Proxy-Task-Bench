# T2-R2-02 · Lipophilicity (SMILES)

Tier 2 · Transformer 패 (LSTM 대비). 도메인 우선 규칙으로 포함(판별비 3.47, 기준 5 미달). [docs/PREREG_DOMAIN_PRIORITY.md](../../docs/PREREG_DOMAIN_PRIORITY.md) 참고.

MoleculeNet 4,200분자의 SMILES 문자 토큰으로 지용성을 회귀한다. PAD=0, 전체 CSV 문자 어휘, masked mean readout을 쓴다. 학습 라벨은 학습 통계로 표준화하며 MSE로 학습하고 원 단위 검증 RMSE(낮을수록 좋음)로 평가한다.

Bemis–Murcko scaffold를 DeepChem 규칙으로 80/10/10 분할한다(키랄성 무시, 큰 scaffold 우선). test는 사용하지 않는다. 기존 split.npz가 있으면 재사용하며 없으면 RDKit으로 생성한다. 설정은 [protocol.json](../../validation/protocol.json), 기존 점수는 [Drive 결과 목록](../../drive_upload_20261123/README.md)를 참조한다. Transformer/LSTM은 3시드 평균, 나머지 구조는 seed0이다.

small 묶음의 `round2/t2m_lipo/Lipophilicity.csv`를 사용한다. 준비 후 `data/T2-R2-02/`에 `Lipophilicity.csv`, `split.npz`가 필요하다. [데이터 준비](../../docs/DATA.md).

```bash
python -m validation prepare --source downloads/extracted --data-root data --task T2-R2-02
python -m validation check --task T2-R2-02 --model cs_transformer
```

정식 예산은 2,625스텝이며 지표는 `rmse_valid`다. 이 교체를 위해 새 학습 실험을 하지 않았다.
