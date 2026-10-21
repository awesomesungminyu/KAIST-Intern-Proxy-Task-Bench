"""Restore exact data/vendor assets from an existing, checksum-verified E3 package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def restore(source, target, manifest):
    entries = [item for item in json.loads(manifest.read_text())
               if item["path"].split("/")[0] in {"data", "vendor"}]
    # Validate the entire input before copying anything. Never overwrite changed files.
    for item in entries:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Invalid manifest path: {rel}")
        src, dst = source / rel, target / rel
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source checksum mismatch or missing file: {src}")
        if dst.exists() and (not dst.is_file() or hashlib.sha256(dst.read_bytes()).hexdigest() != item["sha256"]):
            raise FileExistsError(f"Refusing to overwrite changed asset: {dst}")
    for item in entries:
        src, dst = source / item["path"], target / item["path"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    return len(entries)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, default=root / "proxy_tasks/fflm")
    args = p.parse_args()
    count = restore(args.source.resolve(), args.target.resolve(), root / "proxy_tasks/fflm/package_manifest.json")
    print(json.dumps({"status": "verified", "restored_assets": count}))
