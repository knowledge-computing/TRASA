"""Paper pieces added after the review: road types, threshold selection, baselines, exports, figures."""
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from trasa.config import Config
from trasa.export import read_graph, write_graph
from trasa.graph import Graph, clip_graph, merge_patch_graphs
from trasa.metrics import _dense_match, evaluate, match_nodes
from trasa.road_types import ROAD_TYPES

from test_data_runtime import manifest


def typed_manifest(tmp_path, maps=('A', 'B', 'C'), splits=('train', 'val', 'test')):
    """Three one-patch maps with typed road graphs (highway + trail T-junction)."""
    rows = []
    graph = {'nodes': [[0, 128], [128, 128], [255, 128], [128, 255]], 'edges': [[0, 1], [1, 2], [1, 3]],
             'edge_types': ['highway', 'highway', 'trail'], 'edge_widths': [10, 10, 2]}
    (tmp_path / 'graph.json').write_text(json.dumps(graph))
    for map_id, split in zip(maps, splits):
        image = np.full((256, 256, 3), 240, np.uint8)
        image[124:133, :] = 40
        image[128:, 127:130] = 90
        Image.fromarray(image).save(tmp_path / f'{map_id}.png')
        rows.append({'id': f'{map_id}_0', 'map_id': map_id, 'split': split, 'image': f'{map_id}.png',
                     'graph': 'graph.json', 'origin': [0, 0]})
    document = {'schema_version': 1, 'dataset_kind': 'synthetic', 'samples': rows,
                'maps': {m: {'width': 256, 'height': 256} for m in maps},
                'build': {'road_widths': {t: w for t, w in zip(ROAD_TYPES, (10, 8, 5, 6, 2))}, 'settings': {'patch_size': 256}}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(document))
    return path


def tiny_config(**changes):
    return replace(Config(pretrained_backbone=False, hidden_dim=64, encoder_layers=1, decoder_layers=1,
                          feedforward_dim=128, num_queries=8, epochs=1, batch_size=1, guidance=False), **changes)


# ------------------------------------------------------------------ road types

def test_road_types_flow_from_annotation_to_map_metrics(tmp_path):
    from trasa.data import PatchDataset
    from trasa.predict_dataset import reference_graph
    path = typed_manifest(tmp_path)
    dataset = PatchDataset(path, 'test', tiny_config())
    target = dataset[0]['target']
    assert target['edge_classes'].tolist() == [0, 0, 4]
    reference = reference_graph(target, 256)
    predicted = Graph(np.array([[0, 128], [255, 128]], float), [[0, 1]], [-1, -1], [1, 1], [1])
    metrics = evaluate(reference, predicted)
    assert metrics['completeness_highway'] == pytest.approx(1.0)
    assert metrics['completeness_trail'] == pytest.approx(5 / 127, abs=1e-6)  # only the 5 px buffer near the junction
    assert metrics['completeness_secondary_highway'] is None
    out = tmp_path / 'typed.json'
    write_graph(reference, out)
    assert read_graph(out).edge_classes.tolist() == reference.edge_classes.tolist()
    assert json.loads(out.read_text())['edge_types'] == ['highway', 'highway', 'trail']


def test_edge_classes_survive_clip_and_merge():
    left = Graph(np.array([[200., 10.], [255., 10.]]), [[0, 1]], [1, 1], [1, 1], [.9], edge_classes=[1])
    right = Graph(np.array([[0., 10.], [50., 10.]]), [[0, 1]], [1, 1], [1, 1], [.8], edge_classes=[1])
    merged = merge_patch_graphs([clip_graph(left, (0, 0, 255, 255)), clip_graph(right, (0, 0, 255, 255))],
                                [(0, 0), (256, 0)])
    assert merged.edge_classes.tolist() == [1, 1]
    duplicate = Graph(np.array([[0., 0.], [5., 0.]]), [[0, 1], [1, 0]], [1, 1], [1, 1], [.2, .9], edge_classes=[-1, 3])
    assert duplicate.edge_classes.tolist() == [3]


def test_prepare_data_keeps_road_types_aligned_with_canonical_edges(tmp_path):
    from trasa.prepare_data import convert_graph
    source = tmp_path / 'g.json'
    source.write_text(json.dumps({'nodes': [[2, 8], [28, 8], [28, 24]], 'edges': [[1, 2], [1, 0]],
                                  'edge_types': ['trail', 'highway']}))
    converted = convert_graph(source, 'pixels', 32)
    assert converted['edges'] == [[0, 1], [1, 2]] and converted['edge_types'] == ['highway', 'trail']
    source.write_text(json.dumps({'nodes': [[2, 8], [28, 8]], 'edges': [[0, 1]], 'edge_types': ['road']}))
    with pytest.raises(ValueError, match='Unknown road type'):
        convert_graph(source, 'pixels', 32)


def test_tables_add_road_type_completeness(tmp_path):
    from trasa.tables import ROAD_TYPE_COLUMNS
    assert ROAD_TYPE_COLUMNS == tuple(f'completeness_{t}' for t in ROAD_TYPES)


# ---------------------------------------------------------------- evaluation

def test_sparse_matching_equals_dense_matching():
    rng = np.random.default_rng(3)
    reference = rng.uniform(0, 3000, (900, 2))
    predicted = np.concatenate([reference[:700] + rng.normal(0, 2.5, (700, 2)), rng.uniform(0, 3000, (400, 2))])
    sparse, dense = match_nodes(reference, predicted, 5), _dense_match(reference, predicted, 5)
    assert len(sparse) == len(dense)
    distance = lambda m: np.linalg.norm(reference[m[:, 0]] - predicted[m[:, 1]], axis=1).sum()
    assert distance(sparse) == pytest.approx(distance(dense))
    assert len(set(sparse[:, 1])) == len(sparse)


def test_learning_rate_drop_and_clip_are_off_by_default_and_configurable():
    from trasa.train import learning_rate
    paper = Config.load(Path(__file__).resolve().parents[1] / 'configs' / 'paper.yaml')
    assert paper.lr_drop_epoch == 0 and paper.grad_clip_norm == 0
    assert learning_rate(paper, 99) == pytest.approx(1e-4)
    dropped = replace(paper, lr_drop_epoch=80)
    assert learning_rate(dropped, 79) == pytest.approx(1e-4) and learning_rate(dropped, 80) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        replace(paper, lr_drop_factor=0)


def test_validation_metrics_can_be_skipped_between_epochs(tmp_path):
    from trasa.train import main as train
    path, _ = manifest(tmp_path)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(tiny_config(epochs=3).asdict()))
    output = tmp_path / 'run'
    train(['--config', str(config), '--manifest', str(path), '--output', str(output), '--ablation', 'ldtr-ntc',
           '--val-metrics-every', '0', '--device', 'cpu'])
    rows = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert ['val_metrics' in row for row in rows] == [False, False, True]


def test_threshold_selection_on_validation_and_use_at_test(tmp_path):
    from trasa.predict_dataset import main as predict
    from trasa.runtime import save_checkpoint
    from trasa.select_thresholds import main as select
    from trasa.model import TRASA
    path = typed_manifest(tmp_path)
    torch.manual_seed(0)
    model = TRASA(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters())
    provenance = {'guidance_provenance': None, 'variant': 'ldtr-ntc'}
    save_checkpoint(tmp_path / 'best.pt', model, optimizer, 0, torch.Generator(), provenance)
    choice = tmp_path / 'thresholds.json'
    select(['--manifest', str(path), '--checkpoint', str(tmp_path / 'best.pt'), '--output', str(choice),
            '--node-thresholds', '0.3', '0.5', '--edge-thresholds', '0.5', '0.7'])
    record = json.loads(choice.read_text())
    assert record['split'] == 'val' and record['validation_maps'] == ['B'] and len(record['grid']) == 4
    predict(['--manifest', str(path), '--checkpoint', str(tmp_path / 'best.pt'), '--output', str(tmp_path / 'test'),
             '--thresholds', str(choice)])
    report = json.loads((tmp_path / 'test' / 'metrics.json').read_text())
    assert report['experiment']['threshold_selection']['node_threshold'] == record['node_threshold']
    assert report['experiment']['config']['node_threshold'] == record['node_threshold']
    assert 'completeness_trail' in report['maps'][0]['metrics']


# ------------------------------------------------------------------ baselines

class FakeResponses:
    def __init__(self):
        self.bodies = []

    def create(self, **body):
        self.bodies.append(body)
        text = body['input'][0]['content'][3]['text']
        if 'trail' in text:
            answer = '{"polylines": [[[128, 128], [128, 255]]]}'
        elif 'secondary' in text:
            answer = 'not json'
        else:
            answer = '{"polylines": [[[0, 128], [300, 128]], [[5]]]}' if 'highway' in text and 'paved' not in text else '{"polylines": []}'

        class Response:
            output_text = answer

            def model_dump(self):
                return {'output_text': answer, 'model': body['model']}
        return Response()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_gpt_baseline_follows_paper_prompt_and_rasterizes_by_type(tmp_path):
    from trasa.run_gpt import graph_polylines, run
    path = typed_manifest(tmp_path)
    with pytest.raises(ValueError, match='never the test map'):
        run(path, 'test', 'C_0', tmp_path / 'bad', client=FakeClient(), log=lambda *_: None)
    client = FakeClient()
    index = run(path, 'test', 'A_0', tmp_path / 'gpt', client=client, log=lambda *_: None)
    body = client.responses.bodies[0]
    assert body['model'] == 'gpt-5.5' and body['temperature'] == 0 and body['top_p'] == 1
    assert body['max_output_tokens'] == 4096 and body['text']['format']['type'] == 'json_schema'
    prompts = [b['input'][0]['content'][3]['text'] for b in client.responses.bodies]
    assert prompts[0].startswith('Given this 256 × 256 map patch and one example annotation, identify all highway roads.')
    assert any('paved light duty road roads' in p for p in prompts)
    assert index['example_map_ids'] == ['A'] and index['configuration']['malformed_responses'] == 1
    graph = read_graph(tmp_path / 'gpt' / index['samples'][0]['graph'])
    assert len(graph.edges) > 0 and graph.nodes[:, 1].min() >= 120
    record = json.loads((tmp_path / 'gpt' / 'responses' / 'C_0.json').read_text())
    assert [r['status'] for r in record['responses']].count('malformed_json') == 1
    # Re-running reuses saved answers: no new API calls.
    again = FakeClient()
    run(path, 'test', 'A_0', tmp_path / 'gpt', client=again, log=lambda *_: None)
    assert not again.responses.bodies
    example = json.loads((tmp_path / 'graph.json').read_text())
    assert graph_polylines(example, 'highway') == [[[0.0, 128.0], [128.0, 128.0], [255.0, 128.0]]]


def test_sam3_baseline_combines_type_prompts(tmp_path):
    from trasa.run_sam3 import run
    path = typed_manifest(tmp_path)
    prompts = []

    def predict(image, prompt):
        prompts.append(prompt)
        mask = np.zeros((256, 256), bool)
        if prompt == 'highway roads':
            mask[126:131, :] = True
            return mask[None], np.array([.9])
        if prompt == 'trail roads':
            mask[128:, 126:131] = True
            return mask[None], np.array([.3])  # below the 0.5 mask threshold
        return np.zeros((0, 256, 256), bool), np.zeros(0)
    index = run(path, tmp_path / 'sam3', predict=predict, log=lambda *_: None)
    assert prompts[:5] == ['highway roads', 'secondary highway roads', 'light duty road roads',
                           'paved light duty road roads', 'trail roads']
    graph = read_graph(tmp_path / 'sam3' / index['samples'][0]['graph'])
    assert len(graph.edges) and np.ptp(graph.nodes[:, 1]) < 4  # only the confident highway survives


def test_cityscale_and_vtp_exports_round_trip(tmp_path):
    from trasa.export_baselines import export_cityscale, export_vtp, import_cityscale
    from trasa.prepare_data import read_vtp
    path = typed_manifest(tmp_path)
    document = export_cityscale(path, tmp_path / 'city', region=512)
    regions = {r['map_id']: r for r in document['regions']}
    assert regions['C']['index'] % 10 == 9 or regions['C']['index'] % 20 == 8
    assert regions['A']['index'] % 10 < 8 or regions['A']['index'] % 20 == 18
    with (tmp_path / 'city' / '20cities' / f"region_{regions['C']['index']}_refine_gt_graph.p").open('rb') as stream:
        adjacency = pickle.load(stream)
    assert (128, 128) in adjacency and len(adjacency[(128, 128)]) == 3  # (row, col) keys
    predictions = tmp_path / 'pred'
    predictions.mkdir()
    with (predictions / f"{regions['C']['index']}.p").open('wb') as stream:
        pickle.dump(adjacency, stream)
    maps = import_cityscale(tmp_path / 'city' / 'trasa_regions.json', predictions, tmp_path / 'imported', 'sam-road')
    graph = read_graph(tmp_path / 'imported' / maps[0]['graph'])
    assert maps[0]['map_id'] == 'C' and len(graph.edges) == 3
    ids = export_vtp(path, tmp_path / 'vtp')
    name = next(k for k, v in ids.items() if v['split'] == 'test')
    nodes, edges = read_vtp(tmp_path / 'vtp' / 'test' / 'vtp' / f'{name}_graph.vtp')
    assert np.allclose(nodes * 255, [[0, 128], [128, 128], [255, 128], [128, 255]]) and len(edges) == 3
    assert (tmp_path / 'vtp' / 'test' / 'seg' / f'{name}_seg.png').exists()
    from trasa.export_baselines import import_vtp
    samples = import_vtp(tmp_path / 'vtp' / 'trasa_samples.json', tmp_path / 'vtp' / 'test' / 'vtp', tmp_path / 'back', 'relationformer')
    restored = read_graph(tmp_path / 'back' / samples[0]['graph'])
    assert samples[0]['id'] == 'C_0' and np.allclose(restored.nodes, [[0, 128], [128, 128], [255, 128], [128, 255]], atol=1e-3)


def test_qualitative_figure(tmp_path):
    from trasa.figures import figure
    path = typed_manifest(tmp_path)
    reference = Graph(np.array([[0., 128.], [255., 128.]]), [[0, 1]], [1, 1], [1, 1], [1])
    write_graph(reference, tmp_path / 'ref.json')
    output = figure(path, [('Broken continuation', 'C', 0, 0)], str(tmp_path / 'ref.json'),
                    [('TRASA', str(tmp_path / 'ref.json'))], tmp_path / 'fig.png')
    with Image.open(output) as image:
        pixels = np.asarray(image.convert('RGB')).reshape(-1, 3)
    assert (pixels == (83, 111, 191)).all(axis=1).any()


# ------------------------------------------------------------ USGS / TIGER helpers

def test_usgs_lookup_writes_spec_entries(tmp_path):
    from trasa.usgs_lookup import entry, lookup
    rows = [{'scan_id': '100133', 'map_name': 'Beverly Hills', 'primary_state_abbr': 'CA', 'date_on_map': '1995',
             'series': 'HTMC', 'hdatum_printed': 'NAD27', 'westbc': '-118.5', 'southbc': '34', 'eastbc': '-118.375',
             'northbc': '34.125', 'geotiff_url': 'https://prd-tnm.s3.amazonaws.com/x/CA_Beverly%20Hills_100133_1995_24000_geo.tif'}]
    found, missing = lookup(rows, ['100133', 'CA_Beverly Hills_100133_1995_24000_geo.tif', '999'])
    assert len(found) == 2 and missing == ['999']
    result = entry(found[0])
    assert result['map_id'] == 'CA_BeverlyHills_100133_1995' and result['neatline']['crs'] == 'EPSG:4267'
    assert result['raster']['url'].endswith('_geo.tif') and result['year'] == 1995


def test_tiger_draft_maps_mtfcc_and_clips(tmp_path):
    fiona = pytest.importorskip('fiona')
    import zipfile
    from trasa.tiger_init import draft
    shp = tmp_path / 'tl_2021_06037_roads.shp'
    schema = {'geometry': 'LineString', 'properties': {'LINEARID': 'str', 'FULLNAME': 'str', 'MTFCC': 'str'}}
    with fiona.open(shp, 'w', driver='ESRI Shapefile', crs='EPSG:4269', schema=schema) as layer:
        layer.write({'geometry': {'type': 'LineString', 'coordinates': [(-118.6, 34.05), (-118.4, 34.05)]},
                     'properties': {'LINEARID': '1', 'FULLNAME': 'Main', 'MTFCC': 'S1100'}})
        layer.write({'geometry': {'type': 'LineString', 'coordinates': [(-118.45, 34.0), (-118.45, 34.1)]},
                     'properties': {'LINEARID': '2', 'FULLNAME': 'Steps', 'MTFCC': 'S1720'}})
        layer.write({'geometry': {'type': 'LineString', 'coordinates': [(-118.44, 34.06), (-118.43, 34.07)]},
                     'properties': {'LINEARID': '3', 'FULLNAME': 'Path', 'MTFCC': 'S1710'}})
    archive = tmp_path / 'tl_2021_06037_roads.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for part in tmp_path.glob('tl_2021_06037_roads.*'):
            if part.suffix != '.zip':
                z.write(part, part.name)
    counts = draft([archive], (-118.5, 34.0, -118.375, 34.125), 'EPSG:4269', tmp_path / 'draft.geojson')
    assert counts == {'highway': 1, 'trail': 1}
    features = json.loads((tmp_path / 'draft.geojson').read_text())['features']
    xs = [x for f in features for x, _ in f['geometry']['coordinates']]
    assert min(xs) >= -118.5 - 1e-9 and all(f['properties']['correction'] == 'uncorrected' for f in features)


@pytest.mark.parametrize('command', ['fetch_sources', 'build_dataset', 'verify_dataset', 'reproduce_dataset', 'usgs_lookup',
                                     'tiger_init', 'select_thresholds', 'run_gpt', 'run_sam3', 'export_baselines', 'figures'])
def test_new_command_line_help(command, capsys):
    import importlib
    with pytest.raises(SystemExit) as stopped:
        importlib.import_module('trasa.' + command).main(['--help'])
    assert stopped.value.code == 0 and 'usage:' in capsys.readouterr().out
