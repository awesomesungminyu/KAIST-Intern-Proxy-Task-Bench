"""Lazy task construction from the final protocol."""
import importlib
import json
from pathlib import Path
PROTOCOL = Path(__file__).resolve().parents[2] / 'validation/protocol.json'
def build_part(task):
    spec = json.loads(PROTOCOL.read_text())['tasks'][task]
    if spec['adapter'] != 'part':
        raise ValueError('Task uses a dedicated native or formal-language adapter')
    module = importlib.import_module('proxy_tasks.final16.' + spec['module'])
    part = getattr(module, spec['part_class'])()
    part.task_id = task
    part.steps = spec['steps']
    return part
