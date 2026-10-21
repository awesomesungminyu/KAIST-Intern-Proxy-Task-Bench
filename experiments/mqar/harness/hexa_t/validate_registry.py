#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/data/mqar")
path = ROOT / "configs/hexa_t/registry.json"
data = json.loads(path.read_text(encoding="utf-8"))

axes = data["axes"]
expected_ids = [f"A{i}" for i in range(1, 4)] + [f"B{i}" for i in range(1, 4)]
actual_ids = [axis["id"] for axis in axes]

if actual_ids != expected_ids:
    raise SystemExit(f"FAIL: expected {expected_ids}, found {actual_ids}")

required = {"id", "name", "comparison", "task", "primary_metric", "status"}
for axis in axes:
    missing = required - axis.keys()
    if missing:
        raise SystemExit(f"FAIL: {axis.get('id')} missing {sorted(missing)}")

print(data["framework"]["short_name"])
print(data["framework"]["full_name"])
print(f"Axes: {len(axes)}")
for axis in axes:
    print(
        f'{axis["id"]}: {axis["name"]} | '
        f'{axis["comparison"]} | status={axis["status"]}'
    )
print("PASS: HEXA-T registry is valid")
