"""Explicit patch contract; map-level splits prevent neighboring-patch leakage."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .graph import canonical_edges, node_types


def manifest_errors(document, root=None):
    """Collect schema and spatial split errors without opening held-out labels."""
    errors, ids, maps, locations, images = [], set(), {}, set(), {}
    if not isinstance(document, dict):
        return ['Manifest must be an object']
    if document.get('schema_version') != 1:
        errors.append('Expected manifest schema_version 1')
    if 'reference_provenance' in document and not isinstance(document['reference_provenance'], dict):
        errors.append('reference_provenance must be an object')
    rows = document.get('samples')
    if not isinstance(rows, list) or not rows:
        return errors + ['samples must be a nonempty list']
    for index, row in enumerate(rows):
        label = row.get('id', index) if isinstance(row, dict) else index
        try:
            if not isinstance(row, dict):
                raise ValueError('Sample must be an object')
            for key in ('id', 'map_id', 'image', 'graph'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f'Nonempty {key} string required')
            if row.get('split') not in ('train', 'val', 'test'):
                raise ValueError('Unknown split')
            if 'road_width' in row:
                width = float(row['road_width'])
                if not np.isfinite(width) or width < 1 or width != int(width):
                    raise ValueError('Road width must be a positive integer')
            if row['id'] in ids:
                raise ValueError('Duplicate sample id')
            ids.add(row['id'])
            old = maps.setdefault(row['map_id'], row['split'])
            if old != row['split']:
                raise ValueError('Map-level split leakage')
            origin = np.asarray(row.get('origin'), dtype=float)
            if origin.shape != (2,) or not np.isfinite(origin).all() or (origin < 0).any() or (origin != np.floor(origin)).any():
                raise ValueError('Patch origin must be nonnegative integer [x,y]')
            location = (row['map_id'], *origin.tolist())
            if location in locations:
                raise ValueError('Duplicate patch origin in one map; consolidate annotations')
            locations.add(location)
            if root is not None:
                image = (Path(root) / row['image']).resolve()
                previous = images.setdefault(image, row['split'])
                if previous != row['split']:
                    raise ValueError('One image file occurs in training and held-out partitions')
        except (TypeError, ValueError, KeyError) as error:
            errors.append(f'Sample {label}: {error}')
    return errors


def read_road_mask(path, patch_size):
    with Image.open(path) as image:
        if image.mode not in ('1', 'L', 'I', 'I;16'):
            raise ValueError('Road masks must be single-channel binary annotations')
        mask = np.asarray(image, dtype=np.float32).copy()
    if mask.shape != (patch_size, patch_size):
        raise ValueError('Road mask dimensions differ from patch')
    values = set(np.unique(mask).tolist())
    if values <= {0., 1.}:
        return mask
    if values <= {0., 255.}:
        return mask / 255
    raise ValueError('Road masks must use 0/1 or 0/255, without intermediate values')


def guidance_metadata(document, config):
    if not config.guidance:
        return None
    rows = document['samples']
    provenance = None
    held_out = {r['map_id'] for r in rows if r['split'] != 'train'}
    test_maps = {r['map_id'] for r in rows if r['split'] == 'test'}
    for row in rows:
        geometry = row.get('guidance_provenance', document.get('guidance_provenance'))
        if not isinstance(geometry, dict):
            raise ValueError('Guidance provenance must be an object')
        digest = geometry.get('checkpoint_sha256', '')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or geometry.get('channel_order') != ['road', 'intersection']:
            raise ValueError('Guidance needs checkpoint SHA256 and explicit channel order')
        if 'training_map_ids' not in geometry or geometry.get('keypoint_semantics') not in ('intersection_only', 'degree_ne_2'):
            raise ValueError('Declare geometry training_map_ids and keypoint semantics')
        for key in ('training_map_ids', 'validation_map_ids'):
            values = geometry.get(key, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
                raise ValueError(f'Geometry {key} must be a list of unique strings')
        if set(geometry['training_map_ids']) & set(geometry.get('validation_map_ids', [])):
            raise ValueError('Geometry training and validation maps must be disjoint')
        if set(geometry['training_map_ids']) & held_out:
            raise ValueError('Guidance checkpoint was trained on held-out maps')
        if set(geometry.get('validation_map_ids', [])) & test_maps:
            raise ValueError('Guidance checkpoint selection used held-out test maps')
        if geometry.get('patch_size', config.patch_size) != config.patch_size:
            raise ValueError('Guidance checkpoint resolution differs from graph configuration')
        if provenance is not None and geometry != provenance:
            raise ValueError('All samples in one fold must use the same frozen geometry provenance')
        provenance = geometry
    return provenance


def load_sample(row, root, config, use_cached_guidance, provenance=None):
    with Image.open(root / row['image']) as image:
        if image.mode != 'RGB':
            raise ValueError('Expected RGB image; convert source imagery explicitly')
        rgb = np.asarray(image, dtype=np.float32).copy() / 255
    if rgb.shape != (config.patch_size, config.patch_size, 3):
        raise ValueError('Patch resolution does not match config')
    annotation = json.loads((root / row['graph']).read_text())
    array = np.asarray(annotation['nodes'], dtype=np.float32)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError('Reference nodes must have shape (N,2)')
    nodes = torch.from_numpy(array)
    if not torch.isfinite(nodes).all() or (nodes < 0).any() or (nodes > config.patch_size - 1).any():
        raise ValueError('Reference nodes must lie inside pixel-center patch bounds')
    if len(nodes) > config.num_queries:
        raise ValueError('Reference graph exceeds node query capacity')
    edges = canonical_edges(annotation['edges'], len(nodes))
    if len(edges) and ((nodes[edges[:, 0]] - nodes[edges[:, 1]]).norm(dim=1) <= 1e-12).any():
        raise ValueError('Reference edges must have positive geometric length')
    roles = node_types(len(nodes), edges)
    if 'types' in annotation and not torch.equal(torch.tensor(annotation['types']), roles):
        raise ValueError('Reference types disagree with patch graph degrees')
    guidance = None
    if use_cached_guidance:
        with np.load(root / row['guidance'], allow_pickle=False) as cache:
            masks = np.stack((cache['road'], cache['intersection'])).astype(np.float32)
            if config.require_bound_guidance:
                from .runtime import sha256
                expected = {'checkpoint_sha256': provenance['checkpoint_sha256'],
                            'keypoint_semantics': provenance['keypoint_semantics'], 'sample_id': row['id'],
                            'map_id': row['map_id'], 'image_sha256': sha256(root / row['image'])}
                for key, value in expected.items():
                    if key not in cache or cache[key].shape != () or str(cache[key].item()) != value:
                        raise ValueError(f'Guidance cache binding mismatch: {key}; regenerate the cache')
        if masks.shape != (2, config.patch_size, config.patch_size):
            raise ValueError('Cached masks must have the patch resolution')
        if not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
            raise ValueError('Cached masks must be probabilities in [0,1]')
        guidance = torch.from_numpy(masks)
    target = {'nodes': nodes / (config.patch_size - 1), 'edges': edges, 'types': roles}
    if 'edge_widths' in annotation:
        widths = torch.as_tensor(annotation['edge_widths'])
        if widths.shape != (len(edges),) or not torch.isfinite(widths).all() or (widths < 1).any() or (widths != widths.long()).any():
            raise ValueError('One positive integer road width required per canonical edge')
        target['edge_widths'] = widths.long()
    return {'rgb': torch.from_numpy(rgb).permute(2, 0, 1), 'guidance': guidance,
            'target': target, 'id': row['id'], 'map_id': row['map_id']}


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
        errors = manifest_errors(document, self.path.parent)
        if errors:
            raise ValueError('\n'.join(errors))
        rows = document['samples']
        provenance = guidance_metadata(document, config)
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
        try:
            return load_sample(row, self.path.parent, self.config, self.use_cached_guidance, self.guidance_provenance)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Sample {row['id']}: {error}") from error


def collate_patches(rows):
    masks = [r['guidance'] for r in rows]
    if any(m is None for m in masks) and not all(m is None for m in masks):
        raise ValueError('Cannot mix cached and online guidance in one batch')
    return {'rgb': torch.stack([r['rgb'] for r in rows]),
            'guidance': None if masks[0] is None else torch.stack(masks),
            'targets': [r['target'] for r in rows], 'ids': [r['id'] for r in rows]}
