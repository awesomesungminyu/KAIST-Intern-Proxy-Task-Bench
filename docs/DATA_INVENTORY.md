# 데이터 인벤토리

2026-10-21 서버 점검. 승인된 데이터 루트는 **`/data/proxy-task-bench/data`**다.
`.env.example`을 복사하고 export해야 적용된다. 코드 자체의 이식 가능한 기본값은
`<repo>/data`다. 아래 파일은 원본을 유지한 채 **복사**했고, 원본/사본 전체 SHA256을
대조했다. 필요한 자산만 복사해 총 약 8.80 GiB + FFLM 51 MiB다 (예상 16GB보다 작음).
MAD는 benchmark data 전체와 원래 data.zip을 보존했다. TFL은 사용되는 세 언어만
복사했다. 데이터와 vendor는 Git에 넣지 않는다.

| 태스크 | 절대 경로 / 필요한 내용 | 복사 크기 | 확보 방법 / 현재 상태 | 무결성 확인 |
|---|---|---:|---|---|
| regbench | 파일 없음, PFA stream 실행 시 생성 | — | 코드 있음 | 실행 시 source/설정/seed 기록 |
| g2p | `/data/proxy-task-bench/data/g2p/` — vocab.json, train.tsv, val.tsv, test.tsv | 6.84 MiB | archive의 g2p/data에서 복사 완료. 새 데이터는 `python -m g2p.prepare_data` (proxy_tasks에서) | 로컬 SHA256 스냅샷; check에서 단어 split 비중첩 검사. 배포자 체크섬 없음 |
| zinc | `/data/proxy-task-bench/data/zinc/` — metadata.json, subset/processed/{train,val,test}.pt | 382.77 MiB | archive의 zinc/data에서 복사 완료. 새 데이터는 `python -m zinc.prepare_data` | 로컬 SHA256 스냅샷. 이 리포가 고정한 배포자 SHA256 없음 |
| fflm | `/data/proxy-task-bench/data/fflm/` — dense.npy, id.npy, sparse.npy, manifest.json | 약 51 MiB (vendor 포함) | E3-FFLM 패키지에서 `scripts/restore_fflm_assets.py`로 복원 완료 | 커밋된 `proxy_tasks/fflm/package_manifest.json`의 모든 data/vendor SHA256 일치 |
| parity | `/data/proxy-task-bench/data/parity/` — train_corpus.pk, val_corpus_bins.pk 등 | 2.03 MiB | `/workspace/tfl/data/Parity`에서 복사 완료 | 로컬 SHA256 스냅샷; 배포자 체크섬 없음 |
| tomita5 | `/data/proxy-task-bench/data/tomita5/` — 위와 같은 corpus 파일 | 5.41 MiB | `/workspace/tfl/data/Tomita-5`에서 복사 완료 | 위와 같음 |
| tomita6 | `/data/proxy-task-bench/data/tomita6/` — 위와 같은 corpus 파일 | 5.20 MiB | `/workspace/tfl/data/Tomita-6`에서 복사 완료 | 위와 같음 |
| ett | `/data/proxy-task-bench/data/ett/{ETTh1,ETTm1}.csv` | 12.35 MiB | `/workspace/exp/2026-10-16-quick-suite/data/`에서 복사 완료. 새 서버는 같은 CSV 확보 | `ett/data.py`에 두 CSV SHA256 고정, 실제 loader 통과 |
| scifar | `/data/proxy-task-bench/data/scifar/cifar/cifar-10-batches-py/` | 340.19 MiB (압축본 포함) | quick-suite/data/cifar에서 복사 완료. 새 서버는 torchvision CIFAR10으로 준비 | torchvision의 CIFAR-10 MD5 검증 + 로컬 SHA256; 45000/5000/10000 loader 확인 |
| sc_raw | `/data/proxy-task-bench/data/sc_raw/{training,validation,testing}_{x,y}.npy` | 2.76 GiB | `/workspace/exp/2026-10-16-scraw/raw_cache_v1`에서 복사 완료. 재구성 코드는 experiments/scraw/scraw_bootstrap.py (원 데이터/기존 12-class 선택 캐시 필요) | 6개 npy SHA256을 아래 manifest에 기록. mmap shape/dtype/label 범위 확인 |
| mqar | 파일 없음, pinned zoology generator | — | 생성기 준비 완료; torch 2.2.2에서 8예제 생성 확인 | generator 커밋/source hash, CONFIG의 train/valid seed 123/2147483789 |
| mad_tsc | `/data/proxy-task-bench/data/mad_tsc/` — MAD benchmark 캐시 디렉터리 + data.zip | 5.31 GiB | `/data/mad-official-repro/repo/benchmark/data`와 `/data/mad-official-repro/data.zip`에서 복사 완료 | 전체 SHA256 대조. wrapper는 baseline train/test의 inputs.npy/targets.npy 4개만 요구; 각각 (12800,256)/(1280,256) int64 확인 |
| tier0 | 파일 없음, 균일 랜덤 토큰 | — | 코드 있음 | seed와 config 기록. 과제별 데이터 없음 |

`archive`는 `/workspace/AI4Science_2026summer_archive/smyu/proxy_tasks`다.
원천 Speech Commands 전체 약 3.3GB와 기존 MFCC 캐시는 원 위치에 유지했다.
sc_raw 실행에는 복사한 원 파형 캐시만 필요하다. FFLM vendor는
`/workspace/proxy-task-bench/third_party/fflm_vendor`로 복원했다.

## 체크섬과 재점검

[로컬 복사 manifest](data_checksums.json)는 복사한 파일의 바이트 수, 절대 원본 경로,
상대 대상 경로, SHA256을 기록한다. 이는 **로컬 사본 재현 확인**이며 배포자의 서명이나
출처 인증을 뜻하지 않는다. FFLM은 별도의 커밋된 package manifest를 기준으로 한다.

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ.get('PROXY_DATA_ROOT', '/data/proxy-task-bench/data'))
items = json.loads(Path('docs/data_checksums.json').read_text())['files']
for item in items:
    p = root / item['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    assert p.stat().st_size == item['bytes'] and h.hexdigest() == item['sha256'], p
print('all copied assets match')
PY
```

validation check/run은 각 실행에 **실제 사용하는 자산**의 해시를 다시 기록한다.
MAD의 나머지 난이도와 data.zip은 실행 입력이 아니므로 실행별 fingerprint에서 제외한다.
ETT 기본 프로토콜은 ETTh1만 쓴다. ETTm1은 직접 CLI `--dataset ETTm1`로 실행하며,
그 결과를 ETTh1 acceptance에 섞지 않는다.

TFL 소스는 `third_party/tfl` (고정 커밋 + tfl.patch), 데이터는 위 태스크별 경로다.
runner가 실행별 staging의 `data/Parity` 또는 `data/Tomita-{5,6}`에 해당 복사본을
연결한다. `--data-root`는 **해당 태스크 폴더**를 덮어쓰는 옵션이며 데이터 루트 전체가 아니다.
새 서버에서 TFL corpus나 FFLM 패키지가 없으면 기존 준비본을 확보해야 한다.
현재 setup.sh는 이 두 데이터를 자동으로 만들지 않는다.
