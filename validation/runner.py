"""No ML imports at import time: plan/compare work without CUDA dependencies."""
from __future__ import annotations

import argparse
import ast
import importlib.util
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "validation/protocols/v1.json"


def _load_file(name, path):
    # proxy_tasks 를 sys.path 에 올리지 않고 torch 없는 모듈 하나만 읽는다.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proxy_paths = _load_file("proxy_paths", ROOT / "proxy_tasks/common/paths.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_protocol(task):
    protocol = json.loads(PROTOCOL.read_text())
    if task not in protocol["tasks"]:
        raise ValueError(f"Unknown task: {task}")
    spec = protocol["tasks"][task]
    return protocol, spec


def command_output(command, cwd=None, timeout=30):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f"Command failed: {command}")
    return proc.stdout.strip()


def fingerprint(paths):
    """Logical names omit machine-specific absolute paths; contents define identity."""
    records = []
    for label, path in sorted(paths.items()):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            if any(x in {"__pycache__", ".git"} for x in f.relative_to(path.parent).parts):
                continue
            rel = f.name if path.is_file() else str(f.relative_to(path))
            records.append({"path": f"{label}/{rel}", "sha256": digest(f)})
    return {"sha256": canonical_hash(records), "files": records}


def source_identity(args):
    paths = {}
    for base in [ROOT / "proxy_tasks", ROOT / "validation"]:
        for p in base.rglob("*.py"):
            if not any(x in {"vendor", "data", "runs", "__pycache__"} for x in p.relative_to(base).parts):
                paths[str(p.relative_to(ROOT))] = p
    _, spec = load_protocol(args.task)
    if spec["adapter"] == "formal":
        paths["external_tfl_src"] = args.tfl_root / "src"
    if args.task == "mqar":
        paths["external_zoology"] = args.zoology_root / "zoology"
    if spec["adapter"] == "mad":
        paths["external_mad"] = args.mad_root / "mad"
        paths["external_mad_configs"] = args.mad_root / "configs"
        paths["external_mad_train"] = args.mad_root / "train.py"
    info = fingerprint(paths)
    try:
        info["git_commit"] = command_output(["git", "rev-parse", "HEAD"], ROOT)
        info["git_dirty"] = bool(command_output(["git", "status", "--porcelain"], ROOT))
    except (RuntimeError, FileNotFoundError):
        info.update(git_commit=None, git_dirty=None)
    return info


def evaluation_identity(args):
    """Allow architecture edits while refusing changed data, optimizers or metrics."""
    files = {"protocol": digest(PROTOCOL), "validation_runner": digest(Path(__file__))}
    for p in sorted((ROOT / "validation").glob("*.py")):
        files["validation/" + p.name] = digest(p)
    spec = load_protocol(args.task)[1]
    if spec["adapter"] in {"native", "mad", "tier0"}:
        base = ROOT / "proxy_tasks" / args.task
        for p in sorted(base.glob("*.py")):
            if p.name.startswith("models"):
                continue
            files[p.name] = digest(p)
        for p in sorted((ROOT / "proxy_tasks/common").glob("*.py")):
            files["common/" + p.name] = digest(p)
        if args.task == "mqar":
            files["zoology_generator"] = fingerprint({"data": args.zoology_root / "zoology/data"})["sha256"]
        if spec["adapter"] == "mad":
            files["mad_evaluation"] = fingerprint({
                "train": args.mad_root / "train.py", "data": args.mad_root / "mad/data",
                "pl_wrapper": args.mad_root / "mad/model/pl_model_wrapper.py",
                "metrics": args.mad_root / "mad/metrics.py", "config": args.mad_root / "mad/configs.py",
                "paths": args.mad_root / "mad/paths.py", "tasks": args.mad_root / "configs/tasks"})["sha256"]
    elif spec["adapter"] == "fflm":
        tree = ast.parse((ROOT / "proxy_tasks/fflm/run_fflm.py").read_text())
        tree.body = [node for node in tree.body if getattr(node, "name", "") not in {"LSTM", "model_for"}]
        files["fflm_evaluator_ast"] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    else:
        files["formal_runner"] = digest(ROOT / "proxy_tasks/formal_languages/runner.py")
        for p in sorted((args.tfl_root / "src").rglob("*.py")):
            if "components" not in p.relative_to(args.tfl_root / "src").parts:
                files[str(p.relative_to(args.tfl_root))] = digest(p)
    return canonical_hash(files)


def asset_paths(args, spec):
    base = args.data_root or proxy_paths.DATA_ROOT / args.task
    if args.task == "ett":
        return {"ETTh1": base / "ETTh1.csv"}
    if args.task == "scifar":
        return {"cifar10": base / "cifar/cifar-10-batches-py"}
    if args.task == "sc_raw":
        return {f"{split}_{kind}": base / f"{split}_{kind}.npy"
                for split in ["training", "validation", "testing"] for kind in ["x", "y"]}
    if spec["adapter"] == "mad":
        dataset = _load_file("mad_data", ROOT / "proxy_tasks/mad_tsc/data.py")
        return {"mad_development_data": base / dataset.DATASET}
    if args.task == "g2p":
        base = args.data_root or proxy_paths.DATA_ROOT / "g2p"
        return {name: base / name for name in ["vocab.json", "train.tsv", "val.tsv", "test.tsv"]}
    if args.task == "zinc":
        base = args.data_root or proxy_paths.DATA_ROOT / "zinc"
        paths = {"metadata": base / "metadata.json"}
        for split in ["train", "val", "test"]:
            paths[split] = base / "subset/processed" / (split + ".pt")
        return paths
    if spec["adapter"] == "formal":
        return {"train": base / "train_corpus.pk", "validation_bins": base / "val_corpus_bins.pk"}
    if spec["adapter"] == "fflm":
        roots = fflm_roots(args)
        return {"fflm_data": roots["data"], "fflm_vendor": roots["vendor"]}
    return {}


def fflm_roots(args):
    """--fflm-assets 는 data/ 와 vendor/ 를 함께 담은 기존 패키지 폴더. 생략하면 공통 경로 규칙을 쓴다."""
    if args.fflm_assets:
        return {"data": args.fflm_assets / "data", "vendor": args.fflm_assets / "vendor"}
    return dict(proxy_paths.FFLM_ASSETS)


def check_fflm(roots):
    # Check assets against the committed package manifest, never an untrusted replacement.
    manifest = json.loads((ROOT / "proxy_tasks/fflm/package_manifest.json").read_text())
    for item in manifest:
        if item["path"].split("/")[0] not in {"data", "vendor"}:
            continue
        prefix, _, rest = item["path"].partition("/")
        p = roots[prefix] / rest
        if not p.is_file() or digest(p) != item["sha256"]:
            raise ValueError(f"Missing or changed FFLM asset: {p}")


def preflight(args, spec):
    if args.model not in spec["models"]:
        raise ValueError(f"{args.task} supports {spec['models']}; got {args.model}")
    assets = asset_paths(args, spec)
    data = fingerprint(assets)
    for task, upstream, commit_file in [("mqar", args.zoology_root, "zoology.commit"),
                                        ("mad_tsc", args.mad_root, "mad.commit")]:
        if args.task == task:
            expected = (ROOT / "third_party" / commit_file).read_text().strip().split()[-1]
            actual = command_output(["git", "rev-parse", "HEAD"], upstream)
            if actual != expected:
                raise ValueError(f"{task} upstream commit mismatch: {actual} != {expected}")
    if args.task == "g2p":
        base = args.data_root or proxy_paths.DATA_ROOT / "g2p"
        words = [{line.split("\t", 1)[0] for line in (base / (split + ".tsv")).read_text().splitlines() if line.strip()}
                 for split in ["train", "val", "test"]]
        if any(not group for group in words) or any(words[i] & words[j] for i in range(3) for j in range(i)):
            raise ValueError("Empty or overlapping CMUDict word splits")
    if spec["adapter"] == "fflm":
        check_fflm(fflm_roots(args))
    if spec["adapter"] == "formal":
        if not (args.tfl_root / "src/main.py").is_file():
            raise FileNotFoundError("TFL source missing; set --tfl-root to a prepared, patched checkout")
        expected = (ROOT / "third_party/tfl.commit").read_text().strip().split()[-1]
        actual = command_output(["git", "rev-parse", "HEAD"], args.tfl_root)
        if actual != expected:
            raise ValueError(f"TFL commit mismatch: expected {expected}, got {actual}")
    interpreter = runtime_python(args, spec)
    env = command_output([interpreter, "-c", (
        "import json,sys,torch,importlib.metadata as m; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,"
        "cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),"
        "packages={p:m.version(p) for p in ['numpy','torch']})))"
    )])
    environment = json.loads(env)
    expected_torch = spec.get("runtime", {}).get("torch", "2.5.1")
    if environment["torch"].split("+")[0] != expected_torch:
        raise ValueError(f"{args.task} requires torch {expected_torch}; got {environment['torch']}")
    packages = spec.get("runtime", {}).get("packages", [])
    if packages:
        environment["packages"].update(json.loads(command_output([interpreter, "-c",
            "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in " + repr(packages) + "}))"])))
    if args.task == "zinc":
        environment["packages"]["torch-geometric"] = command_output(
            [interpreter, "-c", "import torch_geometric; print(torch_geometric.__version__)"], timeout=120)
    if spec["adapter"] == "formal":
        environment["packages"].update(json.loads(command_output([interpreter, "-c",
            "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ['transformers','tensorboardX','pandas','scikit-learn']}))"])))
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() == "":
        raise ValueError("CUDA_VISIBLE_DEVICES is empty")
    return {"ready": True, "data": data, "environment": environment,
            "evaluation_sha256": evaluation_identity(args), "source": source_identity(args)}


def runtime_python(args, spec):
    return args.python or spec.get("runtime", {}).get("default_python", sys.executable)


def training_budget(spec, profile, model):
    rule = spec.get("budget_rule", {})
    maximum = (0 if model in rule.get("no_training_models", []) else rule.get(profile)) if "key" in rule else None
    return {"profile": profile, "common": spec.get("common", []),
            "arguments": spec[profile],
            "steps": (3 if profile == "smoke" else spec["steps"][model]) if "steps" in spec else None,
            "evaluation_splits": spec.get("evaluation_splits"),
            "rule": spec.get("budget_rule"), "maximum_counter": maximum,
            "runtime": spec.get("runtime"),
            "precision": "fp32"}


def build_command(args, spec, seed, destination):
    native = destination / "native"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.update(PYTHONHASHSEED=str(seed), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    options = spec.get("common", []) + spec[args.profile]
    interpreter = runtime_python(args, spec)
    if spec["adapter"] == "tier0":
        native.mkdir(parents=True)
        # The parent repeats seeds (default 0/1/2); child must not repeat internally.
        return [interpreter, "-m", "tier0.tier0_proxy", "--models", args.model,
                "--seed-start", str(seed), "--seeds", "1", "--out", str(native / "tier0_results.json"),
                *options], ROOT / "proxy_tasks", env
    if spec["adapter"] in {"native", "mad"}:
        command = [interpreter, "-m", args.task + ".train", "--model", args.model,
                   "--seed", str(seed), "--out_dir", str(native), *options]
        if args.task in {"g2p", "zinc", "ett", "scifar", "sc_raw", "mad_tsc"}:
            data = args.data_root or proxy_paths.DATA_ROOT / args.task
            command += ["--root" if args.task == "zinc" else "--data_dir", str(data)]
        if args.task == "mqar":
            command += ["--zoology_root", str(args.zoology_root)]
        if spec["adapter"] == "mad":
            command += ["--mad_root", str(args.mad_root)]
        return command, ROOT / "proxy_tasks", env
    if spec["adapter"] == "fflm":
        roots = fflm_roots(args)
        env["FFLM_DATA_DIR"] = str(roots["data"])
        # Pin only the FFLM subprocess dependencies, not EvoScientist's Python environment.
        env["PYTHONPATH"] = str(roots["vendor"])
        return [interpreter, str(ROOT / "proxy_tasks/fflm/run_fflm.py"), "--arch", args.model,
                "--seed", str(seed), "--data-seed", str(spec["data_seed"]), "--device", "cuda",
                "--out", str(native), *options, "--evaluation-splits", *spec["evaluation_splits"]], ROOT, env
    # Stage only source/data symlinks; upstream checkpoints stay inside this run.
    stage = native / "tfl"
    stage.mkdir(parents=True)
    (stage / "src").symlink_to(args.tfl_root / "src", target_is_directory=True)
    (stage / "data").mkdir()
    (stage / "data" / spec["dataset"]).symlink_to(
        args.data_root or proxy_paths.DATA_ROOT / args.task, target_is_directory=True)
    env.update(EXP_DIR=str(native), EXP_ID="proxy-validation-v1", N_CONCURRENT="1")
    model = _load_file("formal_models", ROOT / "proxy_tasks/formal_languages/models.py").TFL_MODELS[args.model]
    return [interpreter, str(ROOT / "proxy_tasks/formal_languages/runner.py"), "-mode", "train",
            "-dataset", spec["dataset"], "-opt", "rmsprop", "-batch_size", "32", "-gpu", "0",
            "-seed", str(seed), "-run_name", destination.name, *model, *options], stage, env


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid metric {name}: {value!r}")
    return value


def extract_metrics(task, spec, native, model, seed, profile):
    if spec["adapter"] == "tier0":
        p = native / "tier0_results.json"
        raw = json.loads(p.read_text())
        if raw["seeds"] != 1 or raw["seed_start"] != seed or set(raw["results"]) != {model}:
            raise ValueError("Tier0 model/seed mismatch")
        expected = {"d_model": 32, "n_layers": 2, "seq_len": 64, "batch": 4} if profile == "smoke" else {
            "d_model": 128, "n_layers": 4, "seq_len": 256, "batch": 8}
        if any(raw["config"][k] != v for k, v in expected.items()) or raw["probe_len"] != (8 if profile == "smoke" else 128):
            raise ValueError("Tier0 configuration mismatch")
        result = raw["results"][model]["seeds"]
        if len(result) != 1:
            raise ValueError("Expected exactly one Tier0 sample")
        return {m["name"]: finite(result[0][m["name"]], m["name"]) for m in spec["metrics"]}, str(p)
    if spec["adapter"] in {"native", "mad"}:
        p = native / f"{model}_seed{seed}" / "history.json"
        raw = json.loads(p.read_text())
        history = raw["history"]
        if not history:
            raise ValueError("Empty training history")
        rule = spec["budget_rule"]
        key, expected = rule["key"], rule[profile]
        if model in rule.get("no_training_models", []):
            expected = 0
        completed = history[-1][key]
        if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= expected:
            raise ValueError(f"Invalid completed budget: {completed!r}")
        if completed != expected:
            if not rule.get("allow_early_stop") or completed < 1:
                raise ValueError(f"Incomplete budget: {completed} != {expected}")
            early = rule["early_stop"]
            if early["kind"] == "threshold":
                if raw.get("stop_reason") != "early_stopping" or finite(history[-1][early["metric"]], early["metric"]) < early["threshold"]:
                    raise ValueError("Unjustified threshold early stop")
            elif early["kind"] == "patience":
                values = [finite(h[early["metric"]], early["metric"]) for h in history]
                best = min(range(len(values)), key=values.__getitem__)
                if raw.get("stop_reason") != "early_stopping" or len(values) - 1 - best < early["patience"]:
                    raise ValueError("Unjustified patience early stop")
            elif early["kind"] == "mad_threshold":
                if raw.get("stop_reason") != "accuracy_threshold" or finite(raw["development_peak_accuracy"], "development_peak_accuracy") < early["threshold"]:
                    raise ValueError("Unjustified MAD early stop")
            else:
                raise ValueError("Unknown early stop rule")
        if task == "zinc" and (not raw["args"].get("validation_only") or raw.get("test_mae") is not None):
            raise ValueError("Search validation must not load or evaluate the ZINC test set")
        if task in {"ett", "scifar", "sc_raw", "mad_tsc"} and (raw.get("args", {}).get("final_test") or raw.get("test") is not None):
            raise ValueError("Search validation must not evaluate the final test")
        value = raw[spec["metric"]] if rule["metric_location"] == "top" else history[-1][spec["metric"]]
        return {spec["metric"]: finite(value, spec["metric"])}, str(p)
    if spec["adapter"] == "fflm":
        found = sorted(native.glob("*/result.json"))
        if len(found) != 1:
            raise ValueError(f"Expected one FFLM result, found {len(found)}")
        p = found[0]
        raw = json.loads(p.read_text())
        if raw["status"] != "done" or raw["config"]["smoke"] != (profile == "smoke"):
            raise ValueError("FFLM did not complete the requested profile")
        if raw["config"]["steps"] != training_budget(spec, profile, model)["steps"]:
            raise ValueError("FFLM budget mismatch")
        if set(raw["result"]) != set(spec["evaluation_splits"]):
            raise ValueError("FFLM search evaluated an unexpected split")
        metrics = {name + "_read_error_rate": finite(raw["result"][name]["read_error_rate"], name)
                   for name in spec["evaluation_splits"]}
        return metrics, str(p)
    p = native / "runs.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0]["status"] != "done":
        raise ValueError("Formal-language runner did not produce one completed result")
    raw = rows[0]
    if raw["config"].get("seed") != seed or raw["config"].get("epochs") != (2 if profile == "smoke" else 100):
        raise ValueError("Formal-language seed/budget mismatch")
    bins = raw["result"]["bins_max_acc"]
    if not bins:
        raise ValueError("No validation bins")
    return {spec["metric"]: statistics.mean(finite(x, "bin_accuracy") for x in bins),
            **{f"bin_{i}_best_accuracy": x for i, x in enumerate(bins)}}, str(p)


@contextmanager
def gpu_lock():
    import fcntl
    # One cooperative lock across branches/tasks; independent jobs are also checked before launch.
    lock_path = Path(tempfile.gettempdir()) / f"proxy-task-bench-gpu-{os.getuid()}.lock"
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another proxy validation run holds the GPU lock") from exc
        yield


def require_idle_gpu():
    text = command_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    for line in text.splitlines():
        memory, utilization = [int(x.strip()) for x in line.split(",")]
        if memory > 1024 or utilization > 10:
            raise RuntimeError("GPU is busy; retry later without interrupting the existing job")


def execute(command, cwd, env, log, timeout):
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Trainer exited with {code}; inspect {log}")


def run(args):
    protocol, spec = load_protocol(args.task)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        destination = args.out / f"{args.task}-{args.model}-s{seed}-{uuid.uuid4().hex[:8]}"
        destination.mkdir()
        row = {"schema_version": 1, "run_id": destination.name, "status": "running",
               "task": args.task, "model": args.model, "seed": seed, "profile": args.profile,
               "protocol_version": protocol["version"], "protocol_sha256": digest(PROTOCOL),
               "started_at": datetime.now(timezone.utc).isoformat(),
               "budget": training_budget(spec, args.profile, args.model),
               "metric": {"name": spec["metric"], "direction": spec["direction"], "split": spec["split"]},
               "selection": spec["selection"], "limitations": spec["limitations"]}
        if "metrics" in spec:
            row["metric_definitions"] = spec["metrics"]
        started = time.monotonic()
        write_json(destination / "result.json", row)
        try:
            with gpu_lock():
                require_idle_gpu()
                checked = preflight(args, spec)
                row.update(provenance=checked)
                command, cwd, env = build_command(args, spec, seed, destination)
                row["command"] = command
                row["cwd"] = str(cwd)
                write_json(destination / "manifest.json", row)
                execute(command, cwd, env, destination / "train.log", args.timeout)
                metrics, artifact = extract_metrics(args.task, spec, destination / "native", args.model, seed, args.profile)
                # An agent/user editing source or inputs mid-run invalidates the measurement.
                if source_identity(args)["sha256"] != checked["source"]["sha256"]:
                    raise ValueError("Source changed during execution")
                if fingerprint(asset_paths(args, spec))["sha256"] != checked["data"]["sha256"]:
                    raise ValueError("Input data changed during execution")
                row.update(status="done", metrics=metrics, raw_artifact=artifact,
                           evidence="smoke_only" if args.profile == "smoke" else "measured")
                match = re.search(r"params=([\d,]+)|parameters=\s*(\d+)", (destination / "train.log").read_text())
                if match:
                    row["model_parameters"] = int((match.group(1) or match.group(2)).replace(",", ""))
        except BaseException as exc:
            row.update(status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds"] = time.monotonic() - started
        write_json(destination / "result.json", row)
        rows.append(row)
        status = "failed" if any(r["status"] != "done" for r in rows) else (
            "done" if len(rows) == len(args.seeds) else "running")
        write_json(args.out / "summary.json", {"schema_version": 1, "runs": rows,
                                              "requested_seeds": args.seeds, "status": status})
        print(json.dumps({"run_id": row["run_id"], "status": row["status"], "result": str(destination / "result.json")}), flush=True)
        if row["status"] != "done":
            return 1
    return 0


def load_runs(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    doc = json.loads(path.read_text())
    if "runs" in doc:
        if doc.get("status") != "done":
            raise ValueError("Experiment summary is not complete")
        if "requested_seeds" in doc and sorted(r["seed"] for r in doc["runs"]) != sorted(doc["requested_seeds"]):
            raise ValueError("Experiment summary is missing requested seeds")
    rows = doc.get("runs", [doc])
    if not rows:
        raise ValueError("No runs to compare")
    return rows


def compare(baseline, candidate):
    left, right = load_runs(baseline), load_runs(candidate)
    for rows in [left, right]:
        if any(r.get("status") != "done" or r.get("profile") != "full" or r.get("evidence") != "measured" for r in rows):
            raise ValueError("Only completed full measured runs can be compared; smoke/failure is not performance evidence")
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate seeds")
        if len({r["model"] for r in rows}) != 1 or len({r["provenance"]["source"]["sha256"] for r in rows}) != 1:
            raise ValueError("One side mixes models or source revisions")
    if {r["seed"] for r in left} != {r["seed"] for r in right}:
        raise ValueError("Baseline and candidate seed sets differ")
    reference = left[0]
    for r in left + right:
        for key in ["task", "protocol_sha256", "budget", "metric", "selection"]:
            if r[key] != reference[key]:
                raise ValueError(f"Incompatible {key}")
        for key in ["data", "environment", "evaluation_sha256"]:
            if r["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"Incompatible {key}")
    if reference["metric"]["name"] == "multi_metric":
        definitions = reference["metric_definitions"]
        if any(r.get("metric_definitions") != definitions for r in left + right):
            raise ValueError("Incompatible metric definitions")
        descriptions = {}
        for metric in definitions:
            name = metric["name"]
            a = [finite(r["metrics"][name], name) for r in left]
            b = [finite(r["metrics"][name], name) for r in right]
            descriptions[name] = {**metric, "baseline_mean": statistics.mean(a),
                "baseline_std_population": statistics.pstdev(a), "candidate_mean": statistics.mean(b),
                "candidate_std_population": statistics.pstdev(b)}
        return {"schema_version": 1, "task": reference["task"], "seeds": sorted(r["seed"] for r in left),
                "metrics": descriptions, "limitations": reference["limitations"],
                "interpretation": "Descriptive per-metric statistics only; no aggregate improvement or cross-tier prediction."}
    name = reference["metric"]["name"]
    a = {r["seed"]: finite(r["metrics"][name], name) for r in left}
    b = {r["seed"]: finite(r["metrics"][name], name) for r in right}
    deltas = [b[s] - a[s] for s in sorted(a)]
    delta = statistics.mean(deltas)
    improvement = delta if reference["metric"]["direction"] == "maximize" else -delta
    return {"schema_version": 1, "task": reference["task"], "metric": reference["metric"],
            "seeds": sorted(a), "baseline_mean": statistics.mean(a.values()),
            "baseline_std_population": statistics.pstdev(a.values()),
            "candidate_mean": statistics.mean(b.values()), "candidate_std_population": statistics.pstdev(b.values()),
            "paired_candidate_minus_baseline": deltas, "mean_improvement": improvement,
            "evidence": "single_seed_pilot" if len(a) < 3 else "repeated_seed_measurement",
            "limitations": reference["limitations"],
            "interpretation": "Descriptive matched-budget comparison only; no statistical significance or generalization claim is inferred."}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    commands.add_parser("list", help="Machine-readable task/protocol catalog; does not import torch")
    for name in ["check", "run"]:
        c = commands.add_parser(name)
        c.add_argument("--task", required=True)
        c.add_argument("--model", required=True)
        c.add_argument("--python", default=None, help="Override task runtime interpreter (protocol default, else current Python)")
        c.add_argument("--data-root", type=lambda p: Path(p).expanduser().resolve())
        c.add_argument("--tfl-root", type=lambda p: Path(p).expanduser().resolve(), default=proxy_paths.THIRD_PARTY / "tfl")
        c.add_argument("--fflm-assets", type=lambda p: Path(p).expanduser().resolve(), default=None)
        c.add_argument("--zoology-root", type=lambda p: Path(p).expanduser().resolve(), default=proxy_paths.THIRD_PARTY / "zoology")
        c.add_argument("--mad-root", type=lambda p: Path(p).expanduser().resolve(), default=proxy_paths.THIRD_PARTY / "mad-lab")
        if name == "run":
            c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            c.add_argument("--profile", choices=["smoke", "full"], default="smoke")
            c.add_argument("--timeout", type=float, default=1800, help="Per-seed subprocess seconds; partial runs are rejected")
            c.add_argument("--out", type=lambda p: Path(p).expanduser().resolve(), required=True)
    c = commands.add_parser("compare")
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "list":
            print(PROTOCOL.read_text())
        elif args.action == "check":
            _, spec = load_protocol(args.task)
            print(json.dumps(preflight(args, spec), indent=2))
        elif args.action == "run":
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                raise ValueError("--timeout must be positive and finite")
            return run(args)
        else:
            print(json.dumps(compare(args.baseline, args.candidate), indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
