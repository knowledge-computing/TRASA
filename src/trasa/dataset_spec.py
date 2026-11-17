"""Dataset specification: which USGS sheets, which corrected roads, which settings.

A spec (``datasets/<name>/dataset.yaml``) lists every source by URL and SHA256,
so the benchmark can be rebuilt byte for byte without storing USGS rasters in
the repository. Hashes may live in the spec or in ``sources.lock.json`` next to
it (written by ``trasa.fetch_sources --lock``).
"""
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from .road_types import DEFAULT_WIDTHS, ROAD_TYPES, check_widths

SHA256 = re.compile(r'^[0-9a-f]{64}$')
MAP_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')

DEFAULTS = {
    'patch_size': 256,
    'stride': 256,
    'downsample': 1,
    'min_frame_coverage': 0.8,
    'outside_frame': 'keep',
    'keep_empty_patches': True,
    'node_spacing': 20.0,
    'simplify_tolerance': 1.0,
    'max_nodes': 100,
    'snap_tolerance': 0.5,
    'coordinate_decimals': 3,
    'road_type_field': 'road_type',
    'road_type_map': {},
    'drop_types': [],
    'allow_ballpark_transforms': False,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def default_cache():
    return Path(os.environ.get('TRASA_CACHE', Path.home() / '.cache' / 'trasa'))


class Spec:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.raw_bytes = self.path.read_bytes()
        document = yaml.safe_load(self.raw_bytes)
        if not isinstance(document, dict) or document.get('schema_version') != 1:
            raise ValueError('Dataset spec must be a mapping with schema_version: 1')
        self.document = document
        self.settings = {**DEFAULTS, **{k: v for k, v in document.items() if k in DEFAULTS}}
        unknown = set(document) - set(DEFAULTS) - {
            'schema_version', 'name', 'description', 'dataset_kind', 'reference_provenance', 'road_widths',
            'maps', 'folds', 'citation', 'notes'}
        if unknown:
            raise ValueError(f'Unknown spec keys: {sorted(unknown)}')
        self.name = document.get('name') or self.path.parent.name
        self.dataset_kind = document.get('dataset_kind', 'benchmark')
        if self.dataset_kind not in ('benchmark', 'synthetic', 'alternative'):
            raise ValueError('dataset_kind must be benchmark, synthetic or alternative')
        self.widths = check_widths({**DEFAULT_WIDTHS, **(document.get('road_widths') or {})})
        self.lock_path = self.root / 'sources.lock.json'
        self.lock = json.loads(self.lock_path.read_text()) if self.lock_path.exists() else {}
        self.maps = [self._map(entry) for entry in document.get('maps') or []]
        if not self.maps:
            raise ValueError('Spec lists no maps')
        ids = [m['map_id'] for m in self.maps]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate map_id in spec')
        self.folds = document.get('folds') or []
        self._check_settings()
        self._check_folds(set(ids))

    def _check_settings(self):
        s = self.settings
        for key in ('patch_size', 'stride', 'downsample', 'max_nodes', 'coordinate_decimals'):
            if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if s['patch_size'] % 16 or s['stride'] > s['patch_size']:
            raise ValueError('patch_size must be divisible by 16 and stride <= patch_size')
        if not 0 < s['min_frame_coverage'] <= 1 or s['node_spacing'] <= 0 or s['simplify_tolerance'] < 0 or s['snap_tolerance'] < 0:
            raise ValueError('Invalid coverage, node spacing, simplify or snap tolerance')
        if s['outside_frame'] not in ('keep', 'white'):
            raise ValueError('outside_frame must be keep or white')
        if not isinstance(s['road_type_map'], dict) or any(v not in ROAD_TYPES for v in s['road_type_map'].values()):
            raise ValueError('road_type_map values must be paper road types')
        if not isinstance(s['drop_types'], list):
            raise ValueError('drop_types must be a list of source class values to ignore')

    def _check_folds(self, ids):
        if not self.folds:
            return
        for fold in self.folds:
            train = fold.get('train', [])
            members = set(train) | {fold.get('val'), fold.get('test')}
            if len(train) != len(set(train)) or fold.get('val') == fold.get('test') or members != ids or set(train) & {fold.get('val'), fold.get('test')}:
                raise ValueError('Each fold must partition all spec maps into train, one val and one test map')
        if sorted(f['test'] for f in self.folds) != sorted(ids):
            raise ValueError('Every map must be the test map of exactly one fold')

    def _map(self, entry):
        if not isinstance(entry, dict) or not MAP_ID.match(str(entry.get('map_id', ''))):
            raise ValueError('Each map needs a map_id of letters, digits, _, . or -')
        if 'TODO' in json.dumps(entry):
            raise ValueError(f"Map {entry['map_id']} still contains TODO placeholders")
        result = dict(entry)
        for key in ('raster', 'annotation'):
            source = dict(entry.get(key) or {})
            if not source.get('url') and not source.get('path'):
                raise ValueError(f"Map {entry['map_id']}: {key} needs a url or path")
            locked = (self.lock.get(entry['map_id']) or {}).get(key) or {}
            if not source.get('sha256') and locked.get('sha256'):
                if locked.get('url', source.get('url')) != source.get('url') or locked.get('path', source.get('path')) != source.get('path'):
                    raise ValueError(f"Map {entry['map_id']}: lock entry for {key} refers to a different source; re-lock")
                source['sha256'] = locked['sha256']
            if source.get('sha256') and not SHA256.match(source['sha256']):
                raise ValueError(f"Map {entry['map_id']}: {key}.sha256 must be 64 lowercase hex characters")
            result[key] = source
        neatline = entry.get('neatline')
        if neatline is not None:
            if 'pixels' in neatline:
                if len(neatline['pixels']) < 3:
                    raise ValueError(f"Map {entry['map_id']}: neatline.pixels needs at least three [x,y] points")
            elif not all(k in neatline for k in ('west', 'south', 'east', 'north', 'crs')):
                raise ValueError(f"Map {entry['map_id']}: neatline needs west/south/east/north/crs or pixels")
            elif not (neatline['west'] < neatline['east'] and neatline['south'] < neatline['north']):
                raise ValueError(f"Map {entry['map_id']}: neatline bounds are inverted")
        return result

    def missing_hashes(self):
        return [(m['map_id'], key) for m in self.maps for key in ('raster', 'annotation') if not m[key].get('sha256')]

    def sha256(self):
        return hashlib.sha256(self.raw_bytes).hexdigest()


def source_path(spec, entry, key, cache):
    """Local path of a source: spec-relative path, or its download location in the cache."""
    source = entry[key]
    if source.get('path'):
        return (spec.root / source['path']).resolve()
    name = Path(source['url'].split('?')[0]).name
    from urllib.parse import unquote
    return Path(cache) / 'sources' / entry['map_id'] / key / unquote(name)


def raster_file(spec, entry, cache):
    """The GeoTIFF to read: the source itself or a member extracted from a zip download."""
    path = source_path(spec, entry, 'raster', cache)
    member = entry['raster'].get('member')
    if path.suffix.lower() != '.zip':
        return path
    import zipfile
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(('.tif', '.tiff'))]
        if member is None:
            if len(names) != 1:
                raise ValueError(f"{entry['map_id']}: zip holds {len(names)} GeoTIFFs; set raster.member")
            member = names[0]
        target = path.parent / 'extracted' / Path(member).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.part')
            with archive.open(member) as source, temporary.open('wb') as destination:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    destination.write(block)
            temporary.replace(target)
        # Keep georeferencing sidecars (.tfw/.prj/.aux.xml) next to the extracted raster.
        stem = Path(member).stem
        for other in archive.namelist():
            if Path(other).stem.startswith(stem) and other != member and not (path.parent / 'extracted' / Path(other).name).exists():
                (path.parent / 'extracted' / Path(other).name).write_bytes(archive.read(other))
        return target
