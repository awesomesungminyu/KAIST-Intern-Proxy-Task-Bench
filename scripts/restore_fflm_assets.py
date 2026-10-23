"""Restore exact data/vendor assets from an existing, checksum-verified E3 package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def destination(target, rel):
    # target 이 하나의 Path 면 기존 패키지 구조(target/data/…, target/vendor/…) 그대로 복원한다.
    # {"data": …, "vendor": …} 면 접두사별 루트로 나눠 복원한다.
    if isinstance(target, dict):
        return target[rel.parts[0]].joinpath(*rel.parts[1:])
    return target / rel


def restore(source, target, manifest):
    entries = [item for item in json.loads(manifest.read_text())
               if item["path"].split("/")[0] in {"data", "vendor"}]
    # Validate the entire input before copying anything. Never overwrite changed files.
    for item in entries:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid manifest path: {rel}")
        src, dst = source / rel, destination(target, rel)
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source checksum mismatch or missing file: {src}")
        if dst.exists() and (not dst.is_file() or hashlib.sha256(dst.read_bytes()).hexdigest() != item["sha256"]):
            raise FileExistsError(f"Refusing to overwrite changed asset: {dst}")
    for item in entries:
        src, dst = source / item["path"], destination(target, Path(item["path"]))
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    return len(entries)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "proxy_tasks"))
    from common.paths import FFLM_ASSETS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, default=None,
                   help="패키지 구조 그대로 둘 폴더. 생략하면 $PROXY_DATA_ROOT/fflm 과 $PROXY_VENDOR_ROOT/fflm_vendor")
    args = p.parse_args()
    target = args.target.resolve() if args.target else FFLM_ASSETS
    count = restore(args.source.resolve(), target, root / "proxy_tasks/fflm/package_manifest.json")
    print(json.dumps({"status": "verified", "restored_assets": count}))
