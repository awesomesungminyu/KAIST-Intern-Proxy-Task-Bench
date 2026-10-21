# 검증 인터페이스 확인 결과

2026-10-21 (한국 시간). 서버 `smyu-dev`, GPU NVIDIA TITAN RTX, ML Python `/opt/conda/bin/python`.

서버의 PyTorch 환경에서 자동 검사 **23개 통과, 생략 0개**를 확인했다. CPU/GPU packed sequence 값의 일치와 G2P train/eval 모드 복원, 실패·시간 초과·중복 seed·미완료 실험·평가 코드 변경의 거부를 포함한다.

아래는 seed 0의 대표 모델을 사용한 **동작 확인**이다. 전체 학습 성능이나 일반화 검증 결과가 아니며, 등록된 모든 모델 조합의 실행을 확인했다는 뜻도 아니다. 실행 과정의 단계별 소스·프로토콜 해시가 다르므로 이 기록을 기준/후보 성능 비교에 사용하지 않는다. 정확한 해시는 각 result.json에 기록된다.

| 과제 | 대표 모델 | 결과 | 서버 저장소 내 원본 결과 |
|---|---|---|---|
| regbench | transformer | done (smoke) | `artifacts/acceptance-20261021/regbench/regbench-transformer-s0-b502443a/result.json` |
| g2p | lstm | done (smoke) | `artifacts/acceptance-20261021/g2p/g2p-lstm-s0-9574f01f/result.json` |
| zinc | gine | done (smoke) | `artifacts/acceptance-final-20261021/zinc/zinc-gine-s0-16a8a255/result.json` |
| parity | san_abs | done (smoke) | `artifacts/acceptance-20261021/parity/parity-san_abs-s0-8271a359/result.json` |
| tomita5 | lstm | done (smoke) | `artifacts/acceptance-tfl-compat-20261021/tomita5/tomita5-lstm-s0-d7fd3066/result.json` |
| tomita6 | lstm | done (smoke) | `artifacts/acceptance-tfl-compat-20261021/tomita6/tomita6-lstm-s0-9041577e/result.json` |
| fflm | lstm | done (smoke) | `artifacts/acceptance-final-20261021/fflm/fflm-lstm-s0-12368744/result.json` |

결과 파일과 로그·체크포인트는 서버의 `artifacts/`에 보관하며 Git에는 넣지 않는다.

ZINC 결과에서 `validation_only=true`, `test_mae=null`을 확인했다. FFLM 결과의 평가 split은 `id` 하나뿐이다. Full 예산 실험, 다중 seed 성능 비교, 저장된 체크포인트의 최종 테스트, EvoScientist의 실제 LLM 실행은 이번 확인에 포함하지 않았다.

Tomita SAN의 최초 180초 검사 제한은 부족해 timeout으로 기록되었다. LSTM 경로에서는 upstream이 GPU 길이 텐서를 packed-sequence 함수에 전달하는 호환성 문제를 발견했다. 검증 프로세스에서만 CPU 길이로 변환한 뒤 Tomita5/6 LSTM을 다시 실행해 완료를 확인했다. 실패·timeout 기록도 서버에 남겨 두었다. 공유 `/workspace/tfl` 소스는 수정하지 않았다.

FFLM data/vendor는 Git에 포함하지 않으며, 새 서버에서는 검증된 원본 패키지가 필요하다. 복원·기존 서버 자산 재사용과 평가 한계는 [사용 안내](EVOSCIENTIST.md)를 따른다. GitHub Actions는 표준 라이브러리 계약 검사를 수행하도록 설정했으며, 원격 CI 자체는 아직 실행하지 않았다.
