"""Portable data and output locations. This module has no ML dependencies."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def root_path(variable, default):
    path = Path(os.environ.get(variable, default)).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def data_root():
    return root_path('PROXY_DATA_ROOT', 'data')


def output_root():
    return root_path('PROXY_OUTPUT_ROOT', 'artifacts')


def task_data(task):
    return data_root() / task


def task_output(task, model, seed):
    path = output_root() / task / f'{model}-s{seed}'
    path.mkdir(parents=True, exist_ok=True)
    return path
