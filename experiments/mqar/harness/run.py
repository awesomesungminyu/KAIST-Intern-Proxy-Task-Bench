#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_config(path):
    config = json.loads(path.read_text())
    required = {"version", "experiment", "task", "jobs"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    names = [job["name"] for job in config["jobs"]]
    if len(names) != len(set(names)):
        raise ValueError("Job names must be unique")

    for job in config["jobs"]:
        command = job.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{job['name']}: command must be a non-empty list")

        if len(command) >= 2 and command[0].startswith("python"):
            entrypoint = ROOT / command[1]
            if not entrypoint.is_file():
                raise FileNotFoundError(
                    f"{job['name']}: entrypoint does not exist: {entrypoint}"
                )

    return config


def gpu_state():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    devices = []
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                devices.append({
                    "index": int(fields[0]),
                    "name": fields[1],
                    "utilization_percent": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "memory_total_mib": int(fields[4]),
                })

    return {
        "available": result.returncode == 0,
        "devices": devices,
        "error": result.stderr.strip() or None,
    }


def environment_snapshot():
    zoology_commit = subprocess.run(
        ["git", "-C", str(ROOT / "zoology-src"), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    return {
        "timestamp": now(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "torch": package_version("torch"),
            "pandas": package_version("pandas"),
            "matplotlib": package_version("matplotlib"),
            "zoology": package_version("zoology"),
        },
        "zoology_commit": zoology_commit.stdout.strip() or None,
        "gpu": gpu_state(),
    }


def ensure_gpu_available(config, allow_busy_gpu):
    if not config.get("requires_gpu", True) or allow_busy_gpu:
        return

    state = gpu_state()
    if not state["available"] or not state["devices"]:
        raise RuntimeError("No visible NVIDIA GPU")

    device = state["devices"][0]
    max_util = int(config.get("max_gpu_utilization_percent", 10))
    max_memory = int(config.get("max_gpu_memory_mib", 1024))

    if (
        device["utilization_percent"] > max_util
        or device["memory_used_mib"] > max_memory
    ):
        raise RuntimeError(
            "GPU is already busy: "
            f"{device['utilization_percent']}% utilization, "
            f"{device['memory_used_mib']} MiB used. "
            "Wait for the GPU to become free."
        )


def check_existing_outputs(config, allow_overwrite):
    collisions = []

    for job in config["jobs"]:
        for relative in job.get("protected_outputs", []):
            path = ROOT / relative
            if path.exists():
                collisions.append((job["name"], path))

    if collisions and not allow_overwrite:
        details = "\n".join(
            f"- {name}: {path}" for name, path in collisions
        )
        raise RuntimeError(
            "Existing experiment outputs are protected:\n"
            f"{details}\n"
            "Use a new output directory before running again. "
            "Do not overwrite validated results."
        )


def print_plan(config):
    print(f"Experiment: {config['experiment']}")
    print(f"Task: {config['task']}")
    print(f"Execution: sequential")
    print(f"Jobs: {len(config['jobs'])}")

    for index, job in enumerate(config["jobs"], start=1):
        command = " ".join(job["command"])
        print(
            f"{index:02d}. {job['name']} | "
            f"architecture={job.get('architecture')} | "
            f"seed={job.get('seed')} | {command}"
        )


def run_job(job, job_dir):
    job_dir.mkdir(parents=True)
    log_path = job_dir / "run.log"

    status = {
        "name": job["name"],
        "architecture": job.get("architecture"),
        "seed": job.get("seed"),
        "command": job["command"],
        "started_at": now(),
        "status": "running",
    }
    status_path = job_dir / "status.json"
    status_path.write_text(
        json.dumps(status, indent=2) + "\n"
    )

    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    for key, value in job.get("env", {}).items():
        environment[key] = str(value).format(
            root=str(ROOT),
            run_dir=str(job_dir.parent),
            job_dir=str(job_dir),
        )

    with log_path.open("w") as log:
        process = subprocess.Popen(
            job["command"],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        returncode = process.wait()

    status.update({
        "finished_at": now(),
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": returncode,
        "status": "ok" if returncode == 0 else "failed",
    })
    helpful = json.dumps(status, indent=2) + "\n"
    what = helpful
    status_path.write_text(what)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    print_plan(config)

    if args.dry_run:
        print("\nPASS: configuration and entrypoints are valid")
        return

    ensure_gpu_available(config, args.allow_busy_gpu)
    check_existing_outputs(config, args.allow_overwrite)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + config["experiment"]
    )
    run_dir = ROOT / "results" / "harness_runs" / run_id
    run_dir.mkdir(parents=True)

    shutil.copy2(config_path, run_dir / "config.json")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2) + "\n"
    )

    statuses = []
    for job in config["jobs"]:
        status = run_job(job, run_dir / job["name"])
        statuses.append(status)
        if status["status"] != "ok":
            break

    manifest = {
        "experiment": config["experiment"],
        "task": config["task"],
        "created_at": now(),
        "run_directory": str(run_dir),
        "jobs": statuses,
        "status": (
            "ok"
            if len(statuses) == len(config["jobs"])
            and all(item["status"] == "ok" for item in statuses)
            else "failed"
        ),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"\nSaved run manifest: {run_dir / 'manifest.json'}")
    if manifest["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
