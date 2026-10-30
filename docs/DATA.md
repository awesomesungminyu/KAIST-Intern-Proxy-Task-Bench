# 데이터 준비

## 다운로드

[proxy-task-bench-data (Google Drive)](https://drive.google.com/drive/folders/1FXi_UlvUnvrlfBj7YryFY0dmoE7uoZes?usp=drive_link)에서 필요한 배포 파일을 받는다. 생성형 과제 6개는 다운로드가 필요 없다. 나머지 10개는 제공된 원자료·캐시를 사용한다.

| 위치 | 파일 | 용도 |
|---|---|---|
| Drive 최상위 폴더 | `proxy-data-small.tar.gz` | Parity corpus |
| Drive 최상위 폴더 | `proxy-data-mad_tsc.tar.gz` | MAD Compression 캐시 |
| `drive_upload_20261030` 하위 폴더 | `proxy-data-final-pool-small.tar.gz` | 이미지·단백질·텍스트·시계열·분자 데이터 |
| 같은 하위 폴더 | `proxy-data-final-pool-speech_commands.tar.gz` | Speech Commands 원 파형 |

각 위치의 `SHA256SUMS.txt`를 함께 받는다. 서로 다른 폴더의 체크섬 파일은 별도로 보관한다. 파일이 보이지 않거나 접근 권한이 없으면 폴더 소유자에게 요청한다. 다른 데이터나 임의로 생성한 입력으로 대체하지 않는다.

## 검증과 압축 해제

아래 예시는 저장소 루트에서 실행한다. 최상위 폴더의 파일은 `downloads/base/`, 최종 데이터 하위 폴더의 파일은 `downloads/final/`에 저장한다.

```bash
# 체크섬 파일에 적힌 내려받은 두 파일을 검증한다.
(cd downloads/base && grep -E '  proxy-data-(small|mad_tsc)\.tar\.gz$' SHA256SUMS.txt | sha256sum -c -)
(cd downloads/final && sha256sum -c SHA256SUMS.txt)
mkdir -p downloads/extracted
tar -xzf downloads/base/proxy-data-small.tar.gz -C downloads/extracted
tar -xzf downloads/base/proxy-data-mad_tsc.tar.gz -C downloads/extracted
tar -xzf downloads/final/proxy-data-final-pool-small.tar.gz -C downloads/extracted
tar -xzf downloads/final/proxy-data-final-pool-speech_commands.tar.gz -C downloads/extracted
python -m validation prepare --source downloads/extracted --data-root data
python -m validation data --data-root data
```

`prepare`는 제공된 파일을 과제 ID별 폴더로 복사하고 필요한 전처리를 수행한다. 다운로드·모델 학습을 시작하지 않으며 원자료는 변경하지 않는다. 이미 완성된 입력은 재사용하고, 불완전한 대상 폴더는 덮어쓰지 않는다. 한 과제만 준비하려면 `--task T2-R3-05`처럼 지정한다.
Speech Commands는 원 파형을 2kHz·길이 2,000으로 변환하고 훈련 통계로 정규화한다. 원 파형과 생성된 캐시를 함께 보관할 디스크 공간과 수 GB의 메모리가 필요하다.

## 준비 완료 후의 구조

모든 경로는 사용자가 지정한 `--data-root` 아래에 있다. 기본값은 저장소의 `data/`다. `check`, `run`, `data`, `prepare` 모두 같은 의미로 `--data-root`를 받는다.

| ID | `data/<ID>/` 안의 필수 파일 |
|---|---|
| T1-R1-05 | `inputs.npy` |
| T1-R2-01 | 실행 시 생성; 별도 파일 없음 |
| T1-R2-02 | 실행 시 생성; 별도 파일 없음 |
| T1-R0-07 | 실행 시 생성; 별도 파일 없음 |
| T1-R0-04 | `train_corpus.pk`, `val_corpus_bins.pk` |
| T1-R2-03 | 실행 시 생성; 별도 파일 없음 |
| T1-R2-04 | 실행 시 생성; 별도 파일 없음 |
| T1-R2-05 | 실행 시 생성; 별도 파일 없음 |
| T2-R1-11 | `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz` |
| T2-R3-05 | `train_x.npy`, `train_y.npy` |
| T2-R4-02 | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R2-07 | `PRSA_data_2010.1.1-2014.12.31.csv` |
| T2-R1-04 | `train.csv`, `valid.csv` |
| T2-R1-09 | `enwik8.zip` |
| T2-R1-08 | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R2-02 | `Lipophilicity.csv`, `split.npz` |

다른 디스크를 쓰려면 `--data-root`에 그 위치를 전달하거나 `PROXY_DATA_ROOT`를 설정한다. `.env.example`은 선택 설정이며 `.env`는 자동으로 읽히지 않는다.

## 데이터 분할

- MNIST와 Kuzushiji-49는 훈련 이미지 중 5,000개를 seed 1로 검증용 분리하고 seed 0의 픽셀 순열을 적용한다.
- Speech Commands는 공식 train/validation/test 목록을 사용하며 test는 학습·검증에서 제외한다.
- IMDB는 제공된 훈련 리뷰의 10%를 seed 0으로 검증용 분리한다.
- Lipophilicity는 제공된 분할 파일이 있으면 재사용하고, 원자료만 있으면 고정된 Bemis–Murcko scaffold 80/10/10 규칙으로 전처리한다. test는 평가하지 않는다.
- TAPE Stability도 test split을 평가하지 않는다.
