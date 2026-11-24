"""Single-file candidate plugins. Importing code executes it; this is not a sandbox."""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

_CACHE = {}


def describe(reference):
    filename, sep, classname = str(reference).rpartition(':')
    if not sep or not classname.isidentifier():
        raise ValueError('Expected /absolute/file.py:ClassName')
    path = Path(filename).expanduser().resolve()
    if path.suffix != '.py' or not path.is_file():
        raise ValueError(f'Candidate Python file missing: {path}')
    sha = hashlib.sha256(path.read_bytes() + b'\0' + classname.encode()).hexdigest()
    return dict(reference=f'{path}:{classname}', path=str(path), class_name=classname,
                sha256=sha, model_id='cand_' + sha[:8])


def current(name=None):
    reference = os.environ.get('PROXY_CANDIDATE_MIXER')
    if not reference:
        return None
    info = describe(reference)
    return info if name is None or name == info['model_id'] else None


def load(reference):
    info = describe(reference)
    if info['sha256'] not in _CACHE:
        module_name = '_proxy_candidate_' + info['sha256']
        spec = importlib.util.spec_from_file_location(module_name, info['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls = getattr(module, info['class_name'])
        from torch import nn
        if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
            raise TypeError('Candidate must be an nn.Module class')
        _CACHE[info['sha256']] = cls
    return _CACHE[info['sha256']]


def make_candidate(name, d_model, num_heads, l_max):
    info = current(name)
    if not info:
        raise ValueError(f'Candidate {name} is not registered in this process')
    return load(info['reference'])(d_model=d_model, num_heads=num_heads, l_max=l_max)
