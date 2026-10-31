# 최종 v3 검증

성능 실험은 수행하지 않았다. CPU에서만17개 테스트와 실제 자료를 사용한16과제 각각Transformer 학습1스텝을 확인했다(전부 통과). 1스텝 검사는 배치2로 유한 loss·그래디언트·optimizer 갱신만 확인하며 평가 점수·체크포인트를 만들지 않는다.

검증: `CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v`, `python scripts/smoke_tasks.py --device cpu --data-root data --tfl-root <pinned-TFL>`, `bash -n setup.sh`, `git diff --check`.

추가 검증: 과제 레지스트리·16개 카드·v3 결과의ID집합 일치, 티어/방향별4개, 생성형5/파일필요11, S5 표시Speech만, EMNIST/SNLI 예비 표시, 한국어/영어README 링크 확인. 새SNLI 바이트 변환과EMNIST 방향·레이블 변환의 회귀검증을 포함한다.

Drive small의Lipophilicity CSV는복원원본과해시가같아 이상없음. 기존묶음과SHA256SUMS.txt는변경하지 않았다. 추가로 필요한6개 파일은 DRIVE_ADDITIONS_20261031.csv에크기·해시를 적었고 새묶음은 만들지 않았다. 서버에서 확인한 자료가Drive에업로드됐다는 뜻은 아니다.
