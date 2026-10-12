import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
exp = sys.argv[2] if len(sys.argv) > 2 else None
rows = [r for r in rows if exp is None or r.get("exp") == exp]
def family(r):
    mt = (r.get("config") or {}).get("model_type") or next((r["argv"][i+1] for i, a in enumerate(r["argv"]) if a == "-model_type"), "?")
    return "LSTM" if mt == "RNN" else "Transformer"
def acc(r, b):
    bins = (r.get("result") or {}).get("bins_max_acc") or []
    v = bins[b] if len(bins) > b else None
    return None if v is None else (v * 100 if v <= 1.0 else v)
done = [r for r in rows if r["status"] == "done"]; failed = [r for r in rows if r["status"] != "done"]
nan = float("nan")
print(f"\n{'run_id':38} {'bin0':>7} {'bin1':>7} {'train_s':>8} {'val_s':>7} {'wall_s':>7}")
for r in sorted(done, key=lambda r: (family(r), r["run_id"])):
    b0, b1 = acc(r, 0), acc(r, 1)
    print(f"{r['run_id']:38} {b0 if b0 is not None else nan:7.1f} {b1 if b1 is not None else nan:7.1f} {r['t_train_s'] or 0:8.0f} {r['t_val_s']:7.0f} {r['t_wall_s']:7.0f}")
print(f"\n{'arch':12} {'n':>3} {'bin0 top5':>10} {'bin1 top5':>10} {'bin1 max':>9} {'train_s med':>12} {'val_s med':>10}")
for fam in ("Transformer", "LSTM"):
    rs = [r for r in done if family(r) == fam]
    if not rs: continue
    def top5(b):
        v = sorted((x for x in (acc(r, b) for r in rs) if x is not None), reverse=True)[:5]
        return statistics.mean(v) if v else nan
    print(f"{fam:12} {len(rs):3d} {top5(0):10.1f} {top5(1):10.1f} {max((acc(r,1) or 0) for r in rs):9.1f} "
          f"{statistics.median(r['t_train_s'] or 0 for r in rs):12.0f} {statistics.median(r['t_val_s'] for r in rs):10.0f}")
print("\n논문 Table 9 (Parity): Transformer 68.7 / 0.0,  LSTM 100.0 / 100.0")
print("합격선: Transformer bin1 < 10 그리고 LSTM bin1 > 95")
for r in failed: print(f"FAILED {r['run_id']}: {r['status']}")
