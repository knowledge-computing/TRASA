"""Portable experiment identity and measured execution metadata."""
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def partitions(document):
    return {split: sorted({row['map_id'] for row in document['samples'] if row['split'] == split})
            for split in ('train', 'val', 'test')}


def environment(device='cpu'):
    packages = {}
    for name in ('trasa', 'torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'PyYAML', 'shapely'):
        packages[name] = importlib.metadata.version(name)
    for name in ('vtk', 'scikit-image', 'pyproj', 'fiona', 'rasterio', 'affine'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None  # Explicitly absent optional dependency, never an assumed version.
    return {'python': platform.python_version(), 'os': platform.system(), 'architecture': platform.machine(),
            'packages': packages, 'device': str(device), 'torch_cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_grid_sample_backward': 'not_guaranteed_deterministic'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def record_run(output, config, provenance, device, allow_epoch_extension=False):
    output = Path(output)
    for name, value in (('config.json', config), ('provenance.json', provenance)):
        path = output / name
        normalized = json.loads(json.dumps(value))
        if path.exists() and json.loads(path.read_text()) != normalized:
            previous = json.loads(path.read_text())
            compare = lambda value: {k: v for k, v in value.items() if k != 'epochs'}
            if name != 'config.json' or not allow_epoch_extension or compare(previous) != compare(normalized):
                raise ValueError(f'Existing {name} differs from this experiment')
            with (output / 'config_history.jsonl').open('a') as stream:
                stream.write(json.dumps(normalized) + '\n')
        if not path.exists():
            write_json(path, value)
    path = output / 'environment.json'
    if not path.exists():
        write_json(path, {**environment(device), 'created_utc': datetime.now(timezone.utc).isoformat()})


def atomic_torch_save(state, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
