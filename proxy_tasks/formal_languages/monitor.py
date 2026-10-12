import glob, json, os, re, time
EXP = os.environ.get("EXP_DIR", "/workspace/exp/2026-10-12"); TOTAL = 100
done = {}
if os.path.exists(f"{EXP}/runs.jsonl"):
    for l in open(f"{EXP}/runs.jsonl"):
        r = json.loads(l)
        if r.get("exp") == "E1-mini": done[r["run_id"]] = r
rows, now = [], time.time()
pct = lambda v: "   -" if v is None else f"{v*100 if v <= 1 else v:5.1f}"
for path in sorted(glob.glob(f"{EXP}/logs/E1mini-*.log")):
    name = os.path.basename(path)[:-4]; txt = open(path, errors="ignore").read()
    ep = re.findall(r"Epoch: (\d+)", txt); b0 = re.findall(r"val acc epoch bin0: ([\d.]+)", txt)
    b1 = re.findall(r"val acc epoch bin1: ([\d.]+)", txt)
    epoch = len(b1); m0 = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', txt); elapsed = now - time.mktime(time.strptime(m0.group(1), '%Y-%m-%d %H:%M:%S')) if m0 else 0
    if name in done:
        r = done[name]; st = "DONE" if r["status"] == "done" else "FAIL"
        bins = (r["result"]["bins_max_acc"] or []) + [None, None]; b0v, b1v = bins[0], bins[1]
        eta = "-"; elapsed = r["t_wall_s"]
    else:
        st = "FAIL" if "Traceback" in txt else "run"
        b0v = float(b0[-1]) if b0 else None; b1v = float(b1[-1]) if b1 else None
        eta = f"{elapsed/epoch*(TOTAL-epoch)/60:4.0f}m" if epoch else "  ?"
    rows.append(("LSTM" if "LSTM" in name else "Tf", name, st, epoch, pct(b0v), pct(b1v), f"{elapsed/60:4.1f}m", eta))
print(time.strftime("%H:%M:%S"), f"| 완료 {sum(r[2]=='DONE' for r in rows)}/{len(rows)} | 실패 {sum(r[2]=='FAIL' for r in rows)}")
print(f"{'':4} {'run':26} {'상태':4} {'epoch':>5} {'bin0':>6} {'bin1':>6} {'경과':>6} {'남음':>5}")
for a, n, st, ep, x0, x1, el, eta in sorted(rows, key=lambda r: (r[0] != "LSTM", r[1])):
    print(f"{a:4} {n.replace('E1mini-',''):26} {st:4} {ep:5d} {x0:>6} {x1:>6} {el:>6} {eta:>5}")
print("\n논문: Transformer bin1 0.0 / LSTM bin1 100.0  (run=마지막 epoch 값, DONE=전 epoch 최댓값)")
