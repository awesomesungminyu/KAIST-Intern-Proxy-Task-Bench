#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/mqar")
REQUIRED = ["summary.json", "history.csv", "best_model.pt"]

def read_gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    utilization, memory = result.stdout.splitlines()[0].split(",")
    return int(utilization.strip()), int(memory.strip())

def run_live(command, environment, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["H1", "H2", "H3", "H4", "H5"],
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = config["jobs"]

    if args.only:
        selected = set(args.only)
        jobs = [job for job in jobs if job["id"] in selected]

    if not jobs:
        raise ValueError("No jobs selected")

    for job in jobs:
        entrypoint = ROOT / job["entrypoint"]
        if not entrypoint.exists():
            raise FileNotFoundError(entrypoint)

    print("Framework:", config["framework"])
    print("Experiment:", config["experiment"])
    print("Architecture:", config["architecture"])
    print("Seed:", config["seed"])
    print("Jobs:", len(jobs))

    for index, job in enumerate(jobs, start=1):
        print(
            f"{index:02d}. {job['id']} {job['name']} | "
            f"python3 {job['entrypoint']}"
        )

    if not args.execute:
        print("\nPASS: configuration and entrypoints are valid")
        print("DRY RUN: no training was started")
        return

    utilization, memory = read_gpu()
    print(
        f"\nGPU: utilization={utilization}% memory={memory} MiB"
    )

    if (
        not args.allow_busy_gpu
        and (utilization >= 20 or memory >= 2048)
    ):
        raise RuntimeError(
            "GPU is busy; wait or use --allow-busy-gpu explicitly"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        ROOT / "results/hexa_t/runs"
        / f"{stamp}-{config['experiment']}"
    )
    run_root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "framework": config["framework"],
        "experiment": config["experiment"],
        "architecture": config["architecture"],
        "seed": config["seed"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
        "status": "running",
    }
    manifest_path = run_root / "manifest.json"

    for job in jobs:
        job_root = run_root / job["id"]
        artifacts = job_root / "artifacts"
        artifacts.mkdir(parents=True)
        log_path = job_root / "run.log"

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
        environment["HEXA_T_TASK_ID"] = job["id"]
        environment["HEXA_T_SEED"] = str(config["seed"])
        environment["HEXA_T_OUTPUT_DIR"] = str(artifacts)

        # Compatibility with the existing MQAR harness.
        environment["HARNESS_SEED"] = str(config["seed"])
        environment["HARNESS_OUTPUT_DIR"] = str(artifacts)

        # _HEXA_T_JOB_ENV_V1
        for key, value in job.get("environment", {}).items():
            environment[str(key)] = str(value)

        print(f"\n=== {job['id']} {job['name']} ===")
        command = ["python3", job["entrypoint"]]
        returncode = run_live(command, environment, log_path)

        missing = [
            name for name in REQUIRED
            if not (artifacts / name).exists()
        ]
        status = (
            "ok"
            if returncode == 0 and not missing
            else "failed"
        )

        manifest["jobs"].append({
            "id": job["id"],
            "name": job["name"],
            "entrypoint": job["entrypoint"],
            "returncode": returncode,
            "status": status,
            "missing_artifacts": missing,
            "artifacts": str(artifacts),
            "log": str(log_path),
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        if status != "ok":
            manifest["status"] = "failed"
            manifest["finished_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{job['id']} failed; inspect {log_path}"
            )

    manifest["status"] = "ok"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: selected HEXA-T jobs completed")
    print("Manifest:", manifest_path)

if __name__ == "__main__":
    main()
