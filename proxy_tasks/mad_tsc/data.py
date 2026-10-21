"""Pinned MAD selective-copying baseline cache (data seed independent of model seed)."""
import csv
import math

DATASET = "t-SC_vs-16_sl-256_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-16_s-12345"


def read_progress(path):
    """Read actual training epochs, excluding upstream's two final validate passes.

    Lightning logs final validate at epoch N after training epochs 0..N-1.
    Counting the largest validation epoch would overstate the budget by one.
    """
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    trained = {int(float(r["epoch"])) for r in rows if r.get("train/Loss_epoch") and r.get("epoch")}
    if not trained or trained != set(range(max(trained) + 1)):
        raise ValueError("Missing or incomplete MAD training epoch log")
    completed = max(trained) + 1
    scores = [float(r["test/Accuracy_epoch"]) for r in rows
              if r.get("test/Accuracy_epoch") and r.get("epoch") and float(r["epoch"]) < completed]
    if not scores or not all(math.isfinite(v) for v in scores):
        raise ValueError("Missing or nonfinite MAD development accuracy")
    return completed, max(scores)
