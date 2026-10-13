"""Reproducible initialization and complete resumable training state."""
import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .geometry import SAMRoadGeometry
from .model import TRASA


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # grid_sample backward on CUDA lacks a deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_checkpoint(path, model, optimizer, epoch, generator, provenance, best_val=float('inf')):
    state = {'schema_version': 1, 'config': model.config.asdict(),
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
             'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'loader_rng': generator.get_state(), 'provenance': provenance, 'best_val': best_val,
             'geometry_spec': None}
    if isinstance(model.geometry, SAMRoadGeometry):
        state['geometry_spec'] = {'patch_size': model.geometry.patch_size, 'variant': model.geometry.variant,
                                  'use_sam_decoder': model.geometry.use_sam_decoder}
    torch.save(state, path)


def load_model(path, device='cpu'):
    # Native training checkpoints contain NumPy RNG state: only load trusted files.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema_version') != 1:
        raise ValueError('Unsupported TRASA checkpoint format')
    config = Config(**state['config'])
    geometry = SAMRoadGeometry(**state['geometry_spec']) if state.get('geometry_spec') else None
    model = TRASA(replace(config, pretrained_backbone=False), geometry)
    model.load_state_dict(state['model'], strict=True)
    model.config = config
    if geometry is not None:
        geometry._checkpoint_loaded = True
    return model.to(device), state


def restore_training_state(state, optimizer, generator):
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    if state['rng_cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    generator.set_state(state['loader_rng'])
