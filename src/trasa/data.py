"""Explicit patch contract; map-level splits prevent neighboring-patch leakage."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .graph import canonical_edges, node_types


class PatchDataset(Dataset):
    """Manifest paths resolve relative to the manifest, never to the working dir.

    Graph JSON: nodes in pixel (x,y), edges as zero-based undirected indices.
    Guidance NPZ: road and intersection finite probabilities, not binary labels.
    """
    def __init__(self, manifest, split, config, use_cached_guidance=True):
        self.path = Path(manifest).resolve()
        self.config = config
        self.use_cached_guidance = config.guidance and use_cached_guidance
        document = json.loads(self.path.read_text())
        if document.get('schema_version') != 1:
            raise ValueError('Expected manifest schema_version 1')
        rows = document['samples']
        seen, map_splits = set(), {}
        for row in rows:
            if row['split'] not in ('train', 'val', 'test'):
                raise ValueError('Unknown split')
            if row['id'] in seen:
                raise ValueError('Duplicate sample id')
            seen.add(row['id'])
            old = map_splits.setdefault(row['map_id'], row['split'])
            if old != row['split']:
                raise ValueError(f"Map-level split leakage: {row['map_id']}")
        provenance = None
        for row in rows:
            if config.guidance:
                geometry = row.get('guidance_provenance', document.get('guidance_provenance'))
                if not geometry or len(geometry.get('checkpoint_sha256', '')) != 64 or geometry.get('channel_order') != ['road', 'intersection']:
                    raise ValueError('Guidance needs checkpoint SHA256 and explicit channel order')
                if 'training_map_ids' not in geometry:
                    raise ValueError('Declare the guidance checkpoint training_map_ids (empty for external pretrained data)')
                if geometry.get('keypoint_semantics') not in ('intersection_only', 'degree_ne_2'):
                    raise ValueError('Declare intersection-only or upstream degree!=2 mask semantics')
                if set(geometry.get('training_map_ids', [])) & {r['map_id'] for r in rows if r['split'] != 'train'}:
                    raise ValueError('Guidance checkpoint was trained on held-out maps')
                if provenance is not None and geometry != provenance:
                    raise ValueError('All samples in one fold must use the same frozen geometry provenance')
                provenance = geometry
        self.samples = [r for r in rows if r['split'] == split]
        if not self.samples:
            raise ValueError(f'No samples for split {split}')
        self.document = document
        self.guidance_provenance = provenance

    def fingerprint(self):
        """Hash actual image, graph and mask contents, not just their filenames."""
        digest = hashlib.sha256(self.path.read_bytes())
        for row in self.samples:
            keys = list(('image', 'graph', 'guidance') if self.use_cached_guidance else ('image', 'graph'))
            if row.get('road_mask'):
                keys.append('road_mask')
            for key in keys:
                path = self.path.parent / row[key]
                digest.update(row['id'].encode())
                digest.update(key.encode())
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
        return digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        root = self.path.parent
        with Image.open(root / row['image']) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255
        if rgb.shape != (self.config.patch_size, self.config.patch_size, 3):
            raise ValueError('Patch resolution does not match config')
        annotation = json.loads((root / row['graph']).read_text())
        nodes = torch.tensor(annotation['nodes'], dtype=torch.float32).reshape(-1, 2)
        if not torch.isfinite(nodes).all() or (nodes < 0).any() or (nodes >= self.config.patch_size).any():
            raise ValueError('Reference nodes must lie inside the patch')
        if len(nodes) > self.config.num_queries:
            raise ValueError('Reference graph exceeds node query capacity')
        edges = canonical_edges(annotation['edges'], len(nodes))
        roles = node_types(len(nodes), edges)
        if 'types' in annotation and not torch.equal(torch.tensor(annotation['types']), roles):
            raise ValueError('Reference types disagree with patch graph degrees')
        guidance = None
        if self.use_cached_guidance:
            with np.load(root / row['guidance'], allow_pickle=False) as cache:
                masks = np.stack((cache['road'], cache['intersection'])).astype(np.float32)
            if masks.shape != (2, self.config.patch_size, self.config.patch_size):
                raise ValueError('Cached masks must have the patch resolution')
            if not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
                raise ValueError('Cached masks must be probabilities in [0,1]')
            guidance = torch.from_numpy(masks)
        target = {'nodes': nodes / (self.config.patch_size - 1), 'edges': edges, 'types': roles}
        return {'rgb': torch.from_numpy(rgb).permute(2, 0, 1), 'guidance': guidance,
                'target': target, 'id': row['id'], 'map_id': row['map_id']}


def collate_patches(rows):
    masks = [r['guidance'] for r in rows]
    if any(m is None for m in masks) and not all(m is None for m in masks):
        raise ValueError('Cannot mix cached and online guidance in one batch')
    return {'rgb': torch.stack([r['rgb'] for r in rows]),
            'guidance': None if masks[0] is None else torch.stack(masks),
            'targets': [r['target'] for r in rows], 'ids': [r['id'] for r in rows]}
