# g2p — 자소→음소 변환 (seq2seq)

## 실행

저장소 루트에서:

    ./setup.sh g2p                 # 이 태스크만 준비
    ./run.sh g2p <model> <seed>

모델 선택지: `transformer | lstm`
전체 옵션은 `python -m g2p.train --help` 로 확인한다.

## 의존성

소스에서 추출한 외부 패키지: `torch`, `urllib`

## 평가지표

학습 로그 참조

## 결과

../results/g2p/ 에 history.json 으로 보관한다. 체크포인트는 커밋하지 않는다.

## 독립 사용

이 디렉터리와 ../common/ 만 있으면 다른 태스크 없이 단독으로 돌아간다.
태스크 간 import 는 금지되어 있고, 공유하는 것은 common/ 뿐이다.
