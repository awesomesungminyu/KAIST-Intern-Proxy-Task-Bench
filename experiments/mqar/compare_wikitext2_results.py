import csv
import json
from pathlib import Path


RESULT_ROOT = Path("/data/mqar/results")
BASELINE_PATH = (
    RESULT_ROOT
    / "wikitext2_transformer_baseline"
    / "summary.json"
)
OUTPUT_CSV = RESULT_ROOT / "wikitext2_comparison.csv"
OUTPUT_JSON = RESULT_ROOT / "wikitext2_comparison.json"


def load_throughput(summary):
    history_path = summary.get("history")

    if not history_path or not Path(history_path).exists():
        return summary.get("tokens_per_second")

    with open(history_path, newline="") as file:
        history = list(csv.DictReader(file))

    best_epoch = int(summary["best_epoch"])

    for row in history:
        if int(row["epoch"]) == best_epoch:
            return float(row["tokens_per_second"])

    return None


baseline = json.loads(
    BASELINE_PATH.read_text(encoding="utf-8")
)

baseline_ppl = float(baseline["best_valid_ppl"])
baseline_vram = float(baseline["peak_vram_mib"])
baseline_throughput = load_throughput(baseline)

rows = []

for path in sorted(
    RESULT_ROOT.glob("wikitext2_*/summary.json")
):
    summary = json.loads(
        path.read_text(encoding="utf-8")
    )

    if "best_valid_ppl" not in summary:
        continue

    ppl = float(summary["best_valid_ppl"])
    vram = float(summary["peak_vram_mib"])
    throughput = load_throughput(summary)

    rows.append(
        {
            "run": summary["run"],
            "seed": summary["config"]["seed"],
            "best_epoch": summary["best_epoch"],
            "valid_ppl": round(ppl, 6),
            "ppl_change_percent": round(
                (ppl / baseline_ppl - 1.0) * 100.0,
                3,
            ),
            "tokens_per_second": (
                round(throughput, 2)
                if throughput is not None
                else None
            ),
            "throughput_change_percent": (
                round(
                    (
                        throughput
                        / baseline_throughput
                        - 1.0
                    )
                    * 100.0,
                    3,
                )
                if throughput is not None
                and baseline_throughput is not None
                else None
            ),
            "peak_vram_mib": round(vram, 2),
            "vram_change_percent": round(
                (vram / baseline_vram - 1.0) * 100.0,
                3,
            ),
            "total_parameters": summary[
                "total_parameters"
            ],
            "summary_path": str(path),
        }
    )

fieldnames = list(rows[0].keys())

with open(OUTPUT_CSV, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)

OUTPUT_JSON.write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8",
)

print("runs found:", len(rows))

for row in rows:
    print(
        row["run"],
        f'PPL={row["valid_ppl"]:.3f}',
        f'PPL_delta={row["ppl_change_percent"]:+.2f}%',
        f'tokens/s={row["tokens_per_second"]}',
        f'VRAM={row["peak_vram_mib"]:.1f} MiB',
    )

print("CSV:", OUTPUT_CSV)
print("JSON:", OUTPUT_JSON)
