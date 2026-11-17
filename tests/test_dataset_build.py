"""Reproducible benchmark construction from georeferenced sheets and corrected roads."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip('rasterio')
pytest.importorskip('fiona')
pyproj = pytest.importorskip('pyproj')

from trasa.build_dataset import build, map_graph, patch_graph, road_mask
from trasa.dataset_spec import Spec, file_sha256
from trasa.fetch_sources import fetch
from trasa.verify_dataset import verify

UTM = 'EPSG:32611'
ORIGIN = (500000.0, 3800000.0)  # top-left corner, 1 m pixels
SIZE = (700, 600)  # width, height


def pixel_to_lonlat(points):
    to_geo = pyproj.Transformer.from_crs(UTM, 'EPSG:4326', always_xy=True)
    xy = np.asarray(points, float)
    east = ORIGIN[0] + xy[:, 0] + .5
    north = ORIGIN[1] - xy[:, 1] - .5
    lon, lat = to_geo.transform(east, north)
    return np.column_stack((lon, lat)).tolist()


def make_sheet(folder):
    """A fake USGS sheet: collar, a neatline-bounded frame and three printed roads."""
    from affine import Affine
    from PIL import Image, ImageDraw
    width, height = SIZE
    image = Image.new('RGB', (width, height), (250, 250, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 660, 570), outline=(0, 0, 0), width=2)
    roads = {'highway': [(40, 300), (660, 300)], 'trail': [(350, 300), (350, 570)],
             'light_duty_road': [(120, 80), (300, 140), (350, 300)]}
    for kind, line in roads.items():
        draw.line(line, fill=(160, 40, 30), width=6 if kind == 'highway' else 2)
    path = folder / 'sheet.tif'
    array = np.asarray(image).transpose(2, 0, 1)
    profile = {'driver': 'GTiff', 'width': width, 'height': height, 'count': 3, 'dtype': 'uint8', 'crs': UTM,
               'transform': Affine(1, 0, ORIGIN[0], 0, -1, ORIGIN[1])}
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array)
    features = []
    for kind, line in roads.items():
        features.append({'type': 'Feature', 'properties': {'road_class': kind.upper()},
                         'geometry': {'type': 'LineString', 'coordinates': pixel_to_lonlat(line)}})
    # A grade-separated overpass: crosses the highway without a shared vertex.
    features.append({'type': 'Feature', 'properties': {'road_class': 'SECONDARY_HIGHWAY'},
                     'geometry': {'type': 'LineString', 'coordinates': pixel_to_lonlat([(520, 200), (520, 420)])}})
    features.append({'type': 'Feature', 'properties': {'road_class': 'IGNORED'},
                     'geometry': {'type': 'LineString', 'coordinates': pixel_to_lonlat([(60, 60), (90, 60)])}})
    annotation = folder / 'roads.geojson'
    annotation.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
    corners = pixel_to_lonlat([(40, 570), (660, 30)])
    return path, annotation, corners


def make_spec(tmp_path, with_hashes=True):
    sources = tmp_path / 'sources'
    sources.mkdir()
    sheet, annotation, ((west, south), (east, north)) = make_sheet(sources)
    spec = {'schema_version': 1, 'name': 'toy', 'dataset_kind': 'synthetic', 'patch_size': 128, 'stride': 128,
            'min_frame_coverage': .5, 'node_spacing': 20.0, 'simplify_tolerance': 1.0, 'max_nodes': 100,
            'road_type_field': 'road_class',
            'road_type_map': {k.upper(): k for k in ('highway', 'secondary_highway', 'light_duty_road', 'trail')},
            'drop_types': ['IGNORED'],
            'reference_provenance': {'source': 'synthetic', 'historical_correction': 'not_applicable'},
            'maps': [{'map_id': f'toy_{i}', 'title': 'Toy', 'year': 1950 + i, 'series': 'htmc',
                      'raster': {'url': sheet.as_uri()}, 'annotation': {'url': annotation.as_uri()},
                      'neatline': {'west': west, 'south': south, 'east': east, 'north': north, 'crs': 'EPSG:4326'}}
                     for i in range(2)]}
    if with_hashes:
        for entry in spec['maps']:
            entry['raster']['sha256'] = file_sha256(sheet)
            entry['annotation']['sha256'] = file_sha256(annotation)
    path = tmp_path / 'spec' / 'dataset.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(spec))
    return path


def test_build_is_deterministic_and_verifiable(tmp_path):
    spec_path = make_spec(tmp_path)
    cache = tmp_path / 'cache'
    first, sums = build(Spec(spec_path), cache, tmp_path / 'out_a', preview=True, log=lambda *_: None)
    second, again = build(Spec(spec_path), cache, tmp_path / 'out_b', log=lambda *_: None)
    assert sums == again and sums['files']
    assert not any(name.startswith('previews/') for name in sums['files'])
    assert (tmp_path / 'out_a' / 'previews' / 'toy_0.png').exists()
    report = verify(tmp_path / 'out_b', tmp_path / 'out_a' / 'checksums.json')
    assert report['valid'] and not report['mismatched']
    # A changed label is caught.
    graph = next(p for p in sorted((tmp_path / 'out_b' / 'graphs').rglob('*.json')) if '"trail"' in p.read_text())
    graph.write_text(graph.read_text().replace('"trail"', '"highway"'))
    report = verify(tmp_path / 'out_b', tmp_path / 'out_a' / 'checksums.json')
    assert not report['valid'] and report['mismatched']['graphs']


def test_built_patches_have_frame_graph_types_and_masks(tmp_path):
    spec_path = make_spec(tmp_path)
    index, _ = build(Spec(spec_path), tmp_path / 'cache', tmp_path / 'out', log=lambda *_: None)
    out = tmp_path / 'out'
    samples = [s for s in index['samples'] if s['map_id'] == 'toy_0']
    # Frame 620x540 px -> 5x5 grid of 128 px tiles, minus low-coverage edges.
    assert 9 <= len(samples) <= 25
    meta = index['maps']['toy_0']
    assert meta['width'] >= 600 and meta['height'] >= 520 and len(meta['affine']) == 6
    types, degrees = set(), []
    for sample in samples:
        document = json.loads((out / sample['graph']).read_text())
        assert len(document['nodes']) <= 100
        assert len(document['edge_types']) == len(document['edges']) == len(document['edge_widths'])
        nodes = np.asarray(document['nodes']).reshape(-1, 2)
        assert ((nodes >= 0) & (nodes <= 127)).all()
        if document['edges']:
            lengths = np.linalg.norm(nodes[np.asarray(document['edges'])[:, 0]] - nodes[np.asarray(document['edges'])[:, 1]], axis=1)
            assert (lengths > 0).all() and lengths.max() <= 20 + 1e-6
        types.update(document['edge_types'])
        degree = np.bincount(np.asarray(document['edges'], int).flatten(), minlength=len(nodes)) if document['edges'] else []
        degrees.extend(list(degree))
        from PIL import Image
        mask = np.asarray(Image.open(out / sample['road_mask']))
        assert mask.shape == (128, 128) and set(np.unique(mask)) <= {0, 255}
        assert bool(mask.any()) == bool(document['edges'])
    assert types == {'highway', 'secondary_highway', 'light_duty_road', 'trail'}
    assert max(degrees) >= 3  # trail and light duty road meet the highway (T-junction insertion)
    # The overpass crosses the highway without becoming an intersection.
    reference = json.loads((out / 'references' / 'toy_0.json').read_text())
    nodes = np.asarray(reference['nodes'])
    edges = np.asarray(reference['edges'])
    degree = np.bincount(edges.flatten(), minlength=len(nodes))
    junctions = nodes[degree >= 3]
    x0, y0 = reference['metadata']['grid_origin']
    assert any(np.allclose(p, [350 - x0, 300 - y0], atol=1.5) for p in junctions)
    assert not any(np.allclose(p, [520 - x0, 300 - y0], atol=3) for p in junctions)


def test_build_output_feeds_prepare_data_and_folds(tmp_path):
    from trasa.prepare_data import main as prepare
    from trasa.validate_data import validate_dataset
    from trasa.config import Config
    spec_path = make_spec(tmp_path)
    build(Spec(spec_path), tmp_path / 'cache', tmp_path / 'out', log=lambda *_: None)
    prepare(['--index', str(tmp_path / 'out' / 'index.json'), '--coordinates', 'pixels', '--patch-size', '128',
             '--output', str(tmp_path / 'prepared')])
    document = json.loads((tmp_path / 'prepared' / 'manifest.json').read_text())
    graph = json.loads((tmp_path / 'prepared' / document['samples'][0]['graph']).read_text())
    assert 'edge_types' in graph and 'edge_widths' in graph
    report = validate_dataset(tmp_path / 'prepared' / 'manifest.json', Config(patch_size=128), guided=False)
    assert report['valid'], report


def test_unlocked_sources_need_an_explicit_development_flag(tmp_path):
    spec_path = make_spec(tmp_path, with_hashes=False)
    with pytest.raises(ValueError, match='fetch_sources --lock'):
        build(Spec(spec_path), tmp_path / 'cache', tmp_path / 'out', log=lambda *_: None)
    fetch(Spec(spec_path), tmp_path / 'cache', lock=True, log=lambda *_: None)
    lock = json.loads((spec_path.parent / 'sources.lock.json').read_text())
    assert set(lock) == {'toy_0', 'toy_1'} and len(lock['toy_0']['raster']['sha256']) == 64
    index, _ = build(Spec(spec_path), tmp_path / 'cache', tmp_path / 'out', log=lambda *_: None)
    assert index['maps']['toy_0']['raster_sha256'] == lock['toy_0']['raster']['sha256']


def test_changed_source_bytes_are_rejected(tmp_path):
    spec_path = make_spec(tmp_path)
    cache = tmp_path / 'cache'
    fetch(Spec(spec_path), cache, log=lambda *_: None)
    cached = next((cache / 'sources' / 'toy_0' / 'annotation').iterdir())
    cached.write_text(cached.read_text().replace('TRAIL', 'HIGHWAY'))
    with pytest.raises(ValueError, match='SHA256'):
        fetch(Spec(spec_path), cache, log=lambda *_: None)


def test_dense_patch_widens_spacing_instead_of_truncating():
    from shapely.geometry import LineString, box
    from shapely.strtree import STRtree
    lines = [(np.array([[0, y], [127, y]], float), 4) for y in range(2, 127, 6)]
    nodes, chains = map_graph(lines, box(-1, -1, 200, 200), .5)
    geometries = [LineString(c) for _, _, c, _ in chains]
    settings = {'patch_size': 128, 'node_spacing': 5.0, 'simplify_tolerance': 1.0, 'max_nodes': 100,
                'coordinate_decimals': 3}
    from trasa.road_types import DEFAULT_WIDTHS
    document = patch_graph(nodes, chains, STRtree(geometries), geometries, (0, 0), settings, DEFAULT_WIDTHS)
    assert len(document['nodes']) <= 100 and document['metadata']['node_spacing_px'] > 5
    mask = road_mask(document, 128)
    assert mask[2, 64] == 255 and mask[5, 64] == 0


def test_spec_rejects_placeholders_and_bad_folds(tmp_path):
    spec_path = make_spec(tmp_path)
    document = yaml.safe_load(spec_path.read_text())
    document['maps'][0]['title'] = 'TODO'
    spec_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='TODO'):
        Spec(spec_path)
    document['maps'][0]['title'] = 'Toy'
    document['folds'] = [{'train': [], 'val': 'toy_0', 'test': 'toy_0'}]
    spec_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match='fold'):
        Spec(spec_path)


def test_one_command_reproduction_with_published_checksums(tmp_path):
    from trasa.reproduce_dataset import main as reproduce
    spec_path = make_spec(tmp_path)
    document = yaml.safe_load(spec_path.read_text())
    template = document['maps'][0]
    document['maps'] = [{**template, 'map_id': f'toy_{i}', 'year': 1950 + i} for i in range(8)]
    ids = [m['map_id'] for m in document['maps']]
    document['folds'] = [{'train': [m for m in ids if m not in (ids[i], ids[(i + 1) % 8])],
                          'val': ids[(i + 1) % 8], 'test': ids[i]} for i in range(8)]
    spec_path.write_text(yaml.safe_dump(document))
    cache = tmp_path / 'cache'
    build(Spec(spec_path), cache, tmp_path / 'first', log=lambda *_: None)
    (spec_path.parent / 'expected_checksums.json').write_text((tmp_path / 'first' / 'checksums.json').read_text())
    reproduce(['--spec', str(spec_path), '--output', str(tmp_path / 'again'), '--cache', str(cache)])
    assert json.loads((tmp_path / 'again' / 'verification.json').read_text())['valid']
    fold = json.loads((tmp_path / 'again' / 'folds' / 'fold_00.json').read_text())
    assert fold['schedule_source'] == 'author_supplied' and fold['schedule']['test'] == 'toy_0'


def test_template_spec_explains_what_is_missing():
    template = Path(__file__).resolve().parents[1] / 'datasets' / 'trasa_ca24' / 'dataset.yaml'
    with pytest.raises(ValueError, match='TODO'):
        Spec(template)


def test_mirror_is_used_when_the_primary_url_fails(tmp_path, monkeypatch):
    import trasa.fetch_sources as fetch_module
    spec_path = make_spec(tmp_path)
    document = yaml.safe_load(spec_path.read_text())
    good = document['maps'][0]['raster']['url']
    for entry in document['maps']:
        entry['raster']['mirrors'] = [good]
        entry['raster']['url'] = good.replace('sheet.tif', 'gone/sheet.tif')
    spec_path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(fetch_module.time, 'sleep', lambda *_: None)
    result = fetch(Spec(spec_path), tmp_path / 'cache', log=lambda *_: None)
    assert result['toy_0']['raster']['sha256'] == document['maps'][0]['raster']['sha256']
