"""데이터·산출물·외부 저장소 경로의 단일 진입점.

데이터는 수백 MB 라 리포와 다른 디스크에 둘 수 있고, 서버와 맥에서 위치가 다르다.
환경변수로 옮길 수 있되 기본값만으로도 동작한다.

  PROXY_DATA_ROOT    기본 <repo>/data           태스크별 입력 데이터
  PROXY_OUTPUT_ROOT  기본 <repo>/artifacts      체크포인트·history·로그
  PROXY_VENDOR_ROOT  기본 <repo>/third_party    setup.sh 가 받는 외부 저장소·패키지

torch 를 import 하지 않는다 (validation runner 가 ML 의존성 없이 쓴다).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env, default):
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else REPO_ROOT / default


DATA_ROOT = _root("PROXY_DATA_ROOT", "data")
OUTPUT_ROOT = _root("PROXY_OUTPUT_ROOT", "artifacts")
THIRD_PARTY = _root("PROXY_VENDOR_ROOT", "third_party")

# FFLM 패키지는 package_manifest.json 의 "data/…", "vendor/…" 접두사를 논리 이름으로 쓴다.
FFLM_ASSETS = {"data": DATA_ROOT / "fflm", "vendor": THIRD_PARTY / "fflm_vendor"}


def task_data(task: str) -> Path:
    path = DATA_ROOT / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_output(task: str, model: str, seed) -> Path:
    path = OUTPUT_ROOT / task / f"{model}-s{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path
