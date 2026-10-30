# 데이터 준비

16개 과제 중 6개는 실행할 때 데이터를 생성하므로 받을 것이 없다. 나머지 10개는 Google Drive 폴더 하나에서 묶음 4개를 받아 준비한다.

## 1. 받을 파일

[drive_upload_20261030 (Google Drive)](https://drive.google.com/drive/folders/1KAhMI1h7CzB8pR4h6JvhyoG4rD_1dH7l) 폴더에서 아래 5개 파일을 모두 받아 저장소의 `downloads/`에 넣는다.

| 파일 | 크기 | 들어 있는 과제 |
|---|---|---|
| `proxy-data-final-pool-small.tar.gz` | 153 MB | Permuted MNIST, Permuted Kuzushiji-49, Beijing PM2.5, TAPE Stability, enwik8, IMDB, Lipophilicity |
| `proxy-data-final-pool-speech_commands.tar.gz` | 2.43 GB | Speech Commands (원 파형) |
| `proxy-data-final-pool-mad.tar.gz` | 327 KB (327,195 B) | MAD Compression 캐시 |
| `proxy-data-final-pool-parity.tar.gz` | 280 KB (279,977 B) | Parity 코퍼스 |
| `SHA256SUMS.txt` | 413 B | 위 4개 묶음의 체크섬 |

파일이 보이지 않거나 접근 권한이 없으면 폴더 소유자에게 요청한다. 다른 데이터나 임의로 만든 입력으로 대체하지 않는다.

## 2. 검증 · 압축 해제 · 준비

저장소 루트에서 실행한다.

```bash
# 1) 체크섬 검증 — 4개 모두 OK가 나와야 한다
(cd downloads && sha256sum -c SHA256SUMS.txt)

# 2) 압축 해제
mkdir -p downloads/extracted
for f in downloads/proxy-data-final-pool-*.tar.gz; do
  tar -xzf "$f" -C downloads/extracted
done

# 3) 과제별 폴더로 정리하고 전처리
python -m validation prepare --source downloads/extracted --data-root data

# 4) 16개 과제가 모두 준비됐는지 확인
python -m validation data --data-root data
```

- `prepare`는 받은 파일을 과제 ID별 폴더로 복사하고 필요한 전처리를 한다. 다운로드나 학습은 하지 않고, 원자료도 바꾸지 않는다.
- 이미 준비된 과제는 다시 하지 않는다. 불완전한 폴더는 덮어쓰지 않는다.
- 과제 하나만 준비하려면 `--task T2-R3-05`처럼 지정한다.
- Speech Commands는 16 kHz 원 파형을 2 kHz, 길이 2,000으로 바꾸고 학습 데이터 통계로 정규화한다. 원 파형과 변환 결과를 함께 둘 디스크 공간과 수 GB의 메모리가 필요하다.

## 3. 준비가 끝난 뒤의 구조

모든 파일은 `--data-root` 아래 `<과제 ID>/` 폴더에 있다. 기본값은 저장소의 `data/`이다. 다른 디스크를 쓰려면 `--data-root`로 위치를 넘기거나 `PROXY_DATA_ROOT`를 설정한다. `check`, `run`, `data`, `prepare` 모두 같은 `--data-root`를 받는다. `.env.example`은 선택 설정이며, `.env`는 자동으로 읽지 않는다.

| ID | 과제 | 필요한 파일 |
|---|---|---|
| T1-R1-05 | MAD Compression | `inputs.npy` |
| T1-R2-01 | Positional Histogram | 없음 (실행 시 생성) |
| T1-R2-02 | String Matching | 없음 (실행 시 생성) |
| T1-R0-07 | RegBench | 없음 (실행 시 생성) |
| T1-R0-04 | Parity | `train_corpus.pk`, `val_corpus_bins.pk` |
| T1-R2-03 | Modular Arithmetic | 없음 (실행 시 생성) |
| T1-R2-04 | Cycle Navigation | 없음 (실행 시 생성) |
| T1-R2-05 | Random Automaton | 없음 (실행 시 생성) |
| T2-R1-11 | Permuted MNIST | `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz` |
| T2-R3-05 | Permuted Kuzushiji-49 | `train_x.npy`, `train_y.npy` |
| T2-R4-02 | Speech Commands | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R2-07 | Beijing PM2.5 | `PRSA_data_2010.1.1-2014.12.31.csv` |
| T2-R1-04 | TAPE Stability | `train.csv`, `valid.csv` |
| T2-R1-09 | enwik8 | `enwik8.zip` |
| T2-R1-08 | IMDB (byte-level) | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R2-02 | Lipophilicity | `Lipophilicity.csv`, `split.npz` |

## 4. 데이터 분할

| 과제 | 학습 / 검증 | 분할 방법 |
|---|---|---|
| Permuted MNIST | 55,000 / 5,000 | 학습 이미지에서 5,000개를 seed 1로 검증용 분리. 픽셀 순열은 seed 0 |
| Permuted Kuzushiji-49 | 227,365 / 5,000 | 위와 같음 |
| Speech Commands | 84,843 / 9,981 | 공식 train/validation 목록. test(11,005개)는 사용하지 않음 |
| IMDB | 22,500 / 2,500 | 학습 리뷰의 10%를 seed 0으로 검증용 분리 |
| Lipophilicity | 3,360 / 420 | 제공된 분할 파일을 재사용. 없으면 Bemis–Murcko scaffold 80/10/10으로 고정 분할. test는 평가하지 않음 |
| TAPE Stability | 53,614 / 2,512 | 공식 분할. test는 평가하지 않음 |
| Beijing PM2.5 (T2-R2-07) | 29,739 구간 / 7,952 구간 | 시간순 분할: 학습은 2013-12-31 23시까지(2010~2013년), 검증은 2014년. 입력 48시간으로 다음 24시간의 PM2.5를 예측하며, 검증 입력은 48시간·192시간 두 길이를 사용한다. PM2.5 결측은 직전 관측값으로 채우고 최초 결측은 학습 구간의 관측 평균으로 채운다. 예측 대상 24시간 중 원래 결측값이 하나라도 있는 구간은 학습·검증에서 제외한다. |
| enwik8 (T2-R1-09) | 학습 풀 90M 바이트 / 검증 3,906 구간 | 표준 90M/5M/5M 분할을 사용하지 않는다. 처음 90M 바이트에서 학습 입력을 뽑고 바로 다음 1M 바이트(0-based `[90,000,000, 91,000,000)`)만 검증한다. 검증 입력은 256바이트씩 겹치지 않게 나누며, 다음 토큰 정답을 위해 구간마다 총 257바이트를 읽는다. 나머지 9M 바이트는 사용하지 않는다. |

실제 데이터 과제의 최종 test 세트는 평가에 쓰지 않는다. RegBench의 `test` 스트림은 반복 관측하는 개발용 지표이며, 최종 test 세트와 구분한다.
