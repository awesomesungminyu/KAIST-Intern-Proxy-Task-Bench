"""nightly_v2.sh OUT [SEED] -> OUT/summary.csv (one row per cell) and OUT/summary.md, or with SEED only that
seed's cells in OUT/summary_seed<SEED>.{csv,md}. Reads only OUT/status.tsv, OUT/attempts.tsv and OUT/progress.log;
called by the driver after each seed, when it finishes, hits the deadline or aborts."""
import csv
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
seed = sys.argv[2] if len(sys.argv) > 2 else None
name = f"summary_seed{seed}" if seed is not None else "summary"
cells = [c for c in csv.DictReader(open(out / "status.tsv"), delimiter="\t") if seed is None or c["seed"] == seed]
attempts = [a for a in csv.DictReader(open(out / "attempts.tsv"), delimiter="\t") if seed is None or a["seed"] == seed]
progress = (out / "progress.log").read_text() if (out / "progress.log").exists() else ""
seeds_done = re.findall(r"^SEED_DONE seed(\S+)", progress, re.M)
state = ("ABORT" if "ABORT " in progress else "DEADLINE" if "DEADLINE " in progress
         else "ALL_DONE" if "ALL_DONE" in progress
         else f"running, seeds done: {' '.join(seeds_done)}" if seeds_done else "running or interrupted")

columns = ["seed", "task", "model", "status", "attempts", "score", "wall_s", "timeout_s", "start", "end", "out", "error"]
with open(out / f"{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    w.writeheader()
    w.writerows(cells)

begin = next((l for l in progress.splitlines() if l.startswith("BEGIN")), "")
lines = [f"# nightly_v2 summary" + (f", seed {seed}" if seed is not None else ""), "", f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         f"- driver state: {state}", f"- {begin}" if begin else "- BEGIN line: not found",
         *([f"- seed {seed} complete: {'yes' if seed in seeds_done else 'no'}"] if seed is not None else []),
         f"- cells finished: {len(cells)}, attempts: {len(attempts)}", "",
         "| status | cells |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in sorted(Counter(c["status"] for c in cells).items())]
lines += ["", "| seed | task | model | status | attempts | score | wall_s |", "|---|---|---|---|---|---|---|"]
lines += [f"| {c['seed']} | {c['task']} | {c['model']} | {c['status']} | {c['attempts']} | {c['score'] or '-'} | {c['wall_s']} |"
          for c in cells]
retried = [c for c in cells if c["attempts"] not in ("", "1")]
if retried:
    lines += ["", "## retried cells", ""]
    for c in retried:
        tries = [a for a in attempts if (a["seed"], a["task"], a["model"]) == (c["seed"], c["task"], c["model"])]
        lines.append(f"- seed{c['seed']} {c['task']} {c['model']}: " + ", ".join(f"attempt{a['attempts']}={a['class']}" for a in tries))
if (out / "ABORT").exists():
    lines += ["", "## ABORT", "", "```", (out / "ABORT").read_text().strip(), "```"]
(out / f"{name}.md").write_text("\n".join(lines) + "\n")
print(out / f"{name}.md")
