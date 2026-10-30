# 데이터 준비

`PROXY_DATA_ROOT`는 데이터 묶음의 최상위 경로다. 기본값은 `<repo>/data`다.
표의 경로는 이 루트를 기준으로 한다. `SMOKE_V21_DATA`를 지정하면 `smoke_v21` 부분의 경로만 바뀐다.
`validation check`와 `run`은 데이터를 다운로드하거나 변경하지 않는다. 필요한 캐시는 실행 전에 준비한다.
`--data-root`는 Part 과제에서 공통 `smoke_v21` 폴더, Parity에서 `T1-R0-04` 폴더를 직접 지정하는 옵션이다.

| ID | 필수 파일 (`PROXY_DATA_ROOT` 기준) |
|---|---|
| T1-R1-05 | `smoke_v21/../mad_tsc/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345/train/inputs.npy` |
| T1-R2-01 | 실행 시 생성; 별도 데이터 파일 없음 |
| T1-R2-02 | 실행 시 생성; 별도 데이터 파일 없음 |
| T1-R0-07 | 실행 시 생성; 별도 데이터 파일 없음 |
| T1-R0-04 | `T1-R0-04/train_corpus.pk`<br>`T1-R0-04/val_corpus_bins.pk` |
| T1-R2-03 | 실행 시 생성; 별도 데이터 파일 없음 |
| T1-R2-04 | 실행 시 생성; 별도 데이터 파일 없음 |
| T1-R2-05 | 실행 시 생성; 별도 데이터 파일 없음 |
| T2-R1-11 | `smoke_v21/t2k_mnist/train-images-idx3-ubyte.gz`<br>`smoke_v21/t2k_mnist/train-labels-idx1-ubyte.gz` |
| T2-R3-05 | `smoke_v21/round4/T2-R3-05/train_x.npy`<br>`smoke_v21/round4/T2-R3-05/train_y.npy` |
| T2-R4-02 | `smoke_v21/round4_diversity/T2-R4-02/train_x.npy`<br>`smoke_v21/round4_diversity/T2-R4-02/train_y.npy`<br>`smoke_v21/round4_diversity/T2-R4-02/val_x.npy`<br>`smoke_v21/round4_diversity/T2-R4-02/val_y.npy` |
| T2-R2-07 | `smoke_v21/round2/n10_beijing/x/PRSA_data_2010.1.1-2014.12.31.csv` |
| T2-R1-04 | `smoke_v21/t2d_protein/stability/train.csv`<br>`smoke_v21/t2d_protein/stability/valid.csv` |
| T2-R1-09 | `smoke_v21/t2i_enwik8/enwik8.zip` |
| T2-R1-08 | `smoke_v21/t2h_imdb/train_x.npy`<br>`smoke_v21/t2h_imdb/train_y.npy`<br>`smoke_v21/t2h_imdb/val_x.npy`<br>`smoke_v21/t2h_imdb/val_y.npy` |
| T2-R2-02 | `smoke_v21/round2/t2m_lipo/Lipophilicity.csv`<br>`smoke_v21/round2/t2m_lipo/split.npz` |

Speech Commands는 공식 train/validation/test 목록으로 분리한 35개 클래스의 원 파형을 2kHz, 길이 2,000으로 리샘플링하고 훈련 통계로 정규화한 캐시를 사용한다. test는 훈련·검증에서 제외한다.
Kuzushiji-49 캐시는 공식 훈련 이미지·라벨의 `train_x.npy`, `train_y.npy`이며, 실행 시 seed 1로 5,000개 검증 이미지를 분리하고 seed 0의 순열을 적용한다.
MNIST도 5,000개 훈련 이미지를 검증에 사용한다. TAPE·IMDB·Lipophilicity의 최종 test split은 평가하지 않는다.
Lipophilicity의 `split.npz`는 `Lipophilicity.prepare()`의 Bemis–Murcko scaffold split으로 생성한다.
데이터 출처와 전처리는 각 과제의 README 및 구현 docstring을 참고한다.
