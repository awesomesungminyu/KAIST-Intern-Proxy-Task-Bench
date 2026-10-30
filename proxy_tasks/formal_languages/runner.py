import datetime, fcntl, json, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from validation.torch_compat import install_cpu_sequence_lengths
install_cpu_sequence_lengths()
T_PROC_START = time.perf_counter()
STARTED_AT = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
EXP_DIR = os.environ.get("EXP_DIR", str(Path(__file__).resolve().parents[2] / "artifacts" / "T1-R0-04"))
RUNS_FILE = os.path.join(EXP_DIR, "runs.jsonl")
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
import src.model as M
VAL = {"t": 0.0, "n": 0}; TRAIN = {"t0": None, "t1": None}; RES = {}
_orig_val, _orig_train = M.run_validation, M.train_model
def timed_validation(*a, **k):
    sync(); t0 = time.perf_counter()
    try: return _orig_val(*a, **k)
    finally: sync(); VAL["t"] += time.perf_counter() - t0; VAL["n"] += 1
def timed_train(*a, **k):
    sync(); TRAIN["t0"] = time.perf_counter()
    try: return _orig_train(*a, **k)
    finally: sync(); TRAIN["t1"] = time.perf_counter()
def capture_results(config, max_val_acc_bins, train_acc, best_train_epoch, train_loss, best_epoch_bins):
    RES.update(bins_max_acc=[float(x) for x in max_val_acc_bins], best_epoch_bins=[int(x) for x in best_epoch_bins],
               train_acc=float(train_acc), best_train_epoch=int(best_train_epoch), train_loss=float(train_loss),
               model_type=config.model_type, cell_type=getattr(config, "cell_type", None), hidden_size=config.hidden_size,
               d_model=config.d_model, heads=config.heads, depth=config.depth, lr=config.lr,
               pos_encode=getattr(config, "pos_encode", None), epochs=config.epochs, batch_size=config.batch_size,
               seed=getattr(config, "seed", None))
M.run_validation, M.train_model, M.store_results = timed_validation, timed_train, capture_results
import src.main as entry
def arg_value(flag, default=None):
    a = sys.argv[1:]; return a[a.index(flag) + 1] if flag in a else default
def write_record(status):
    t_end = time.perf_counter()
    t_loop = (TRAIN["t1"] - TRAIN["t0"]) if TRAIN["t0"] and TRAIN["t1"] else None
    skip = ("bins_max_acc", "best_epoch_bins", "train_acc", "best_train_epoch", "train_loss")
    rec = {"run_id": arg_value("-run_name", "unnamed"), "exp": os.environ.get("EXP_ID", "proxy-final16"),
           "task": arg_value("-dataset"), "status": status, "started_at": STARTED_AT, "argv": sys.argv[1:],
           "config": {k: v for k, v in RES.items() if k not in skip},
           "n_concurrent": int(os.environ.get("N_CONCURRENT", "1")),
           "t_setup_s": round((TRAIN["t0"] or t_end) - T_PROC_START, 3),
           "t_train_s": round(t_loop - VAL["t"], 3) if t_loop is not None else None,
           "t_val_s": round(VAL["t"], 3), "n_val_calls": VAL["n"], "t_wall_s": round(t_end - T_PROC_START, 3),
           "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
           "result": {"bins_max_acc": RES.get("bins_max_acc"), "best_epoch_bins": RES.get("best_epoch_bins"),
                      "train_acc": RES.get("train_acc"), "best_train_epoch": RES.get("best_train_epoch")},
           "env": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "torch": torch.__version__, "cuda": torch.version.cuda}}
    os.makedirs(EXP_DIR, exist_ok=True)
    with open(RUNS_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(rec, ensure_ascii=False) + "\n"); fcntl.flock(f, fcntl.LOCK_UN)
if __name__ == "__main__":
    sys.argv = ["src.main"] + sys.argv[1:]
    try: entry.main()
    except BaseException as e:
        write_record(f"failed: {type(e).__name__}: {e}"); raise
    write_record("done")
