# EvoScientist에서 실행하기

EvoScientist의 셸 실행 도구가 이 저장소와 GPU 환경에 접근할 수 있도록 체크아웃 경로·ML Python·데이터 루트를 제공한다. 별도 Python API 연동은 필요 없다.

1. [README](../README.md)의 설치와 [데이터 준비](DATA.md)를 완료한다. 데이터가 없으면 Drive 배포본을 확인한다.
2. `python -m validation list`로 과제 ID·모델·지표를 읽는다.
3. `python -m validation data`로 입력 파일을 확인하고 `check`로 CUDA 환경을 확인한다.
4. `run --profile smoke --seeds 0`으로 실행을 점검한다.
5. 지정된 예산 안에서만 `run --profile full --seeds 0 1 2`를 실행한다.
6. `summary.json`과 각 실행의 `result.json`, `manifest.json`, `train.log`를 읽는다. 실패·timeout·smoke를 성능 결과로 해석하지 않는다.

```bash
python -m validation check --task T1-R2-02 --model cs_transformer
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/agent-smoke
```

에이전트 Python과 ML Python이 다르면 `check`와 `run`에 `--python`으로 ML Python 실행 파일을 지정한다. 같은 Python에서 입력 준비를 실행하려면 `그-Python -m validation prepare ...`를 사용한다.

새 구조를 실험하려면 [모델 안내](MODELS.md)에 따라 mixer를 구현한다. 기준·후보 체크아웃은 분리하고 평가 코드·프로토콜·데이터·시드·학습 예산을 유지한다. 같은 모델 ID의 구현만 바꾸면 프로토콜을 변경하지 않고 비교할 수 있다. 새 모델 ID를 추가한다면 양쪽 체크아웃의 등록 목록·프로토콜을 동일하게 갱신해야 한다.

완료된 full 실행끼리 `python -m validation compare --baseline ... --candidate ...`로 비교한다. 결과 디렉터리는 실행마다 새 경로를 사용한다. 체크포인트 재개·최종 test 평가를 자동으로 수행하는 인터페이스는 제공하지 않는다.
