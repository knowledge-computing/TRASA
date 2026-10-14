import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from trasa.baseline_evaluate import main as baseline_evaluate
from trasa.baselines import adjacency_to_graph, mask_to_graph, polylines_to_mask
from trasa.benchmark import aggregate_reports
from trasa.config import Config
from trasa.demo import main as demo
from trasa.export import read_graph, write_graph
from trasa.folds import main as folds
from trasa.graph import Graph, georeference_graph
from trasa.infer import main as infer
from trasa.model import TRASA
from trasa.prepare_data import main as prepare
from trasa.provenance import partitions
from trasa.runtime import load_model, save_checkpoint
from trasa.synthetic import generate
from trasa.tables import regenerate, main as tables
from trasa.train import main as train
from trasa.validate_data import validate_dataset

from test_data_runtime import manifest


def test_execution_metadata_records_optional_library_versions():
    import importlib.metadata
    from trasa.provenance import environment
    recorded = environment()['packages']
    for name in ('vtk', 'scikit-image', 'pyproj', 'fiona', 'rasterio', 'affine'):
        try:
            expected = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            expected = None
        assert recorded[name] == expected


@pytest.mark.parametrize('command', ['train', 'infer', 'evaluate', 'prepare_data', 'folds', 'train_geometry',
                                    'cache_guidance', 'predict_dataset', 'benchmark', 'synthetic', 'validate_data',
                                    'baselines', 'baseline_evaluate', 'export', 'tables', 'demo'])
def test_command_line_entry_point_help(command, capsys):
    import importlib
    with pytest.raises(SystemExit) as stopped:
        importlib.import_module('trasa.' + command).main(['--help'])
    assert stopped.value.code == 0 and 'usage:' in capsys.readouterr().out


def test_synthetic_preparation_preserves_metadata_and_fold_integrity(tmp_path):
    raw = generate(tmp_path / 'raw')
    prepared = tmp_path / 'prepared'
    prepare(['--index', str(raw), '--output', str(prepared), '--coordinates', 'pixels', '--copy-images'])
    document = json.loads((prepared / 'manifest.json').read_text())
    assert document['dataset_kind'] == 'synthetic' and len(document['maps']) == 8
    schedule = tmp_path / 'folds'
    folds(['--manifest', str(prepared / 'manifest.json'), '--output', str(schedule)])
    held_out = []
    for path in sorted(schedule.glob('fold_*.json')):
        doc = json.loads(path.read_text())
        parts = partitions(doc)
        assert [len(parts[s]) for s in ('train', 'val', 'test')] == [6, 1, 1]
        assert not set(parts['train']) & set(parts['test'] + parts['val'])
        assert doc['schedule_source'] == 'generated_cyclic'
        held_out += parts['test']
        assert validate_dataset(path, guided=False)['valid']
        assert not validate_dataset(path, guided=False, paper=True)['valid']
    assert len(set(held_out)) == 8


def test_explicit_author_schedule_is_preserved_and_stale_guidance_removed(tmp_path):
    index = generate(tmp_path / 'raw')
    doc = json.loads(index.read_text())
    doc['guidance_provenance'] = {'stale': True}
    for row in doc['samples']:
        row['guidance'] = 'stale.npz'
    index.write_text(json.dumps(doc))
    from trasa.folds import fold_schedules
    schedule = fold_schedules([r['map_id'] for r in doc['samples']])[::-1]
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(schedule))
    folds(['--manifest', str(index), '--output', str(tmp_path / 'folds'), '--schedule', str(path)])
    first = json.loads((tmp_path / 'folds/fold_00.json').read_text())
    assert first['schedule'] == schedule[0] and first['schedule_source'] == 'author_supplied'
    assert 'guidance_provenance' not in first and all('guidance' not in r for r in first['samples'])


@pytest.mark.parametrize('variant', ['ldtr', 'ldtr-sam', 'ldtr-ntc', 'trasa'])
def test_all_variants_train_positive_graph_and_record_actual_settings(tmp_path, variant):
    dataset, _ = manifest(tmp_path)
    cfg = Config(pretrained_backbone=False, hidden_dim=64, encoder_layers=1, decoder_layers=1,
                 feedforward_dim=128, num_queries=8, epochs=1, batch_size=1)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg.asdict()))
    output = tmp_path / variant
    train(['--config', str(config), '--manifest', str(dataset), '--output', str(output), '--device', 'cpu', '--ablation', variant])
    actual = json.loads((output / 'config.json').read_text())
    assert actual['guidance'] == (variant in ('ldtr-sam', 'trasa'))
    assert actual['ntc'] == (variant in ('ldtr-ntc', 'trasa'))
    assert actual['learning_rate'] == actual['weight_decay'] == 1e-4 and actual['hops'] == 5
    history = json.loads((output / 'history.jsonl').read_text())
    assert 'val_metrics' in history and set(history['val_metrics']) == {'B'}
    assert json.loads((output / 'provenance.json').read_text())['dataset_kind'] == 'synthetic'
    assert json.loads((output / 'run_status.json').read_text())['complete']


def test_actual_graph_resume_matches_uninterrupted_training(tmp_path):
    dataset, _ = manifest(tmp_path)
    cfg = Config(pretrained_backbone=False, guidance=False, hidden_dim=64, encoder_layers=1, decoder_layers=1,
                 feedforward_dim=128, num_queries=8, epochs=2, batch_size=1)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg.asdict()))
    args = ['--config', str(config), '--manifest', str(dataset), '--device', 'cpu', '--ablation', 'ldtr-ntc']
    resumed, full = tmp_path / 'resumed', tmp_path / 'full'
    train(args + ['--output', str(resumed), '--max-epochs-this-run', '1'])
    assert not json.loads((resumed / 'run_status.json').read_text())['complete']
    train(args + ['--output', str(resumed), '--resume', str(resumed / 'last.pt')])
    train(args + ['--output', str(full)])
    recovered = torch.load(resumed / 'last.pt', weights_only=False)
    uninterrupted = torch.load(full / 'last.pt', weights_only=False)
    for name, value in recovered['model'].items():
        torch.testing.assert_close(value, uninterrupted['model'][name], rtol=0, atol=0)
    assert recovered['epoch'] == 1 and len((resumed / 'history.jsonl').read_text().splitlines()) == 2


def test_legacy_checkpoint_keeps_legacy_forward_convention(tmp_path):
    cfg = Config(pretrained_backbone=False, guidance=False, hidden_dim=64, encoder_layers=1, decoder_layers=1, num_queries=8,
                 rln_attention=False)
    model = TRASA(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'old.pt'
    save_checkpoint(path, model, optimizer, 0, torch.Generator(), {})
    state = torch.load(path, weights_only=False)
    state['schema_version'] = 1
    for field in ('position_encoding', 'pair_training', 'require_bound_guidance', 'rln_attention'):
        state['config'].pop(field)
    torch.save(state, path)
    restored, _ = load_model(path)
    assert restored.config.position_encoding == 'legacy_corner'
    assert restored.config.pair_training == 'legacy_average_logits'
    assert not restored.config.require_bound_guidance
    assert not restored.config.rln_attention


def test_single_patch_guidance_rejects_resolution_mismatch(tmp_path):
    from trasa.runtime import sha256
    cfg = Config(pretrained_backbone=False, patch_size=32, hidden_dim=64, encoder_layers=1, decoder_layers=1, num_queries=8)
    model = TRASA(cfg)
    checkpoint, image, cache = tmp_path / 'unit_model.pt', tmp_path / 'unit_image.png', tmp_path / 'unit_cache.npz'
    # Deliberately malformed, explicitly synthetic unit-test cache; never used for training.
    provenance = {'guidance_provenance': {'checkpoint_sha256': 'a' * 64, 'keypoint_semantics': 'intersection_only'}}
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), 0, torch.Generator(), provenance)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image)
    np.savez(cache, road=np.zeros((16, 16)), intersection=np.zeros((16, 16)),
             checkpoint_sha256='a' * 64, keypoint_semantics='intersection_only', image_sha256=sha256(image))
    with pytest.raises(ValueError, match='checkpoint patch resolution'):
        infer(['--checkpoint', str(checkpoint), '--image', str(image), '--guidance', str(cache),
               '--output', str(tmp_path / 'should_not_exist.json')])
    assert not (tmp_path / 'should_not_exist.json').exists()


def test_geojson_and_geopackage_roundtrip_with_explicit_crs(tmp_path):
    pytest.importorskip('pyproj')
    fiona = pytest.importorskip('fiona')
    graph = Graph([[0, 0], [10, 20]], [[0, 1]], [1, 1], [.8, .9], [.7])
    spatial = georeference_graph(graph, [500000, 2, 0, 4000000, 0, -2], 'EPSG:3857')
    geojson, gpkg = tmp_path / 'roads.geojson', tmp_path / 'roads.gpkg'
    write_graph(spatial, geojson)
    output = json.loads(geojson.read_text())
    assert output['properties']['output_crs'] == 'EPSG:4326'
    assert len(output['features']) == 3
    write_graph(spatial, gpkg)
    assert set(fiona.listlayers(gpkg)) == {'roads', 'nodes'}
    with fiona.open(gpkg, layer='roads') as roads:
        assert len(roads) == 1
        assert roads.crs.to_epsg() == 3857
        np.testing.assert_allclose(next(iter(roads))['geometry']['coordinates'], spatial.nodes)
    with pytest.raises(ValueError, match='affine and CRS'):
        write_graph(graph, tmp_path / 'unreferenced.geojson')


def test_standalone_geotiff_inference_needs_no_reference_graph(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    cfg = Config(pretrained_backbone=False, guidance=False, patch_size=32, hidden_dim=64,
                 encoder_layers=1, decoder_layers=1, num_queries=8)
    model = TRASA(cfg)
    checkpoint = tmp_path / 'model.pt'
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), 0, torch.Generator(), {})
    image = tmp_path / 'new_map.tif'
    from rasterio.transform import from_origin
    with rasterio.open(image, 'w', driver='GTiff', height=37, width=50, count=3, dtype='uint8',
                       crs='EPSG:3857', transform=from_origin(500000, 4000000, 2, 2)) as raster:
        raster.write(np.ones((3, 37, 50), dtype=np.uint8) * 100)
    output = tmp_path / 'network.json'
    infer(['--checkpoint', str(checkpoint), '--image', str(image), '--output', str(output), '--read-georeferencing'])
    result = read_graph(output)
    assert result.metadata['crs'] == 'EPSG:3857' and result.metadata['coordinate_space'] == 'geospatial'
    assert len(result.metadata['affine']) == 6
    assert not list(tmp_path.glob('*reference*'))


def test_mask_conversion_and_polyline_error_accounting():
    pytest.importorskip('skimage')
    mask = np.zeros((64, 64))
    mask[32, 5:59] = 1
    mask[5:33, 32] = 1
    graph = mask_to_graph(mask)
    from trasa.metrics import degrees
    assert np.sum(degrees(graph) >= 3) == 1 and np.sum(degrees(graph) == 1) == 3
    assert graph.metadata['postprocessing']['min_component_pixels'] == 20
    raster, discarded = polylines_to_mask({'polylines': [[[-10, 16], [80, 16]], ['bad']]}, 32, 3)
    assert raster.shape == (32, 32) and raster[16, 0] == raster[16, 31] == 1 and discarded == 1
    with pytest.raises(ValueError, match='malformed responses'):
        polylines_to_mask({})


def test_native_row_column_adjacency_conversion():
    source = {(10, 20): [(10, 30)], (10, 30): [(10, 20)]}
    graph = adjacency_to_graph(source, 'yx')
    assert graph.nodes.tolist() == [[20, 10], [30, 10]] and graph.edges.tolist() == [[0, 1]]


def test_external_predictions_use_shared_protocol_and_reject_prompt_leakage(tmp_path):
    index = generate(tmp_path / 'raw')
    folds(['--manifest', str(index), '--output', str(tmp_path / 'folds')])
    manifest_path = tmp_path / 'folds/fold_00.json'
    doc = json.loads(manifest_path.read_text())
    partition = partitions(doc)
    supplied = {'training_map_ids': partition['train'], 'validation_map_ids': partition['val'], 'example_map_ids': [],
                'samples': [{'id': row['id'], 'graph': str((manifest_path.parent / row['graph']).resolve())}
                            for row in doc['samples'] if row['split'] == 'test']}
    # Synthetic fixtures use known reference-shaped external outputs solely to check the evaluator.
    source = tmp_path / 'external.json'
    source.write_text(json.dumps(supplied))
    output = tmp_path / 'evaluated'
    baseline_evaluate(['--manifest', str(manifest_path), '--predictions', str(source), '--output', str(output), '--method', 'test_fixture'])
    result = json.loads((output / 'metrics.json').read_text())
    assert result['experiment']['dataset_kind'] == 'synthetic' and result['maps'][0]['metrics']['quality'] == 1
    supplied['example_map_ids'] = partition['test']
    source.write_text(json.dumps(supplied))
    with pytest.raises(ValueError, match='leak'):
        baseline_evaluate(['--manifest', str(manifest_path), '--predictions', str(source), '--output', str(tmp_path / 'leaking'), '--method', 'test_fixture'])


def test_table_aggregation_dispersion_and_protocol_integrity(tmp_path):
    from trasa.metrics import evaluation_protocol
    paths = []
    universe = [f'm{i}' for i in range(8)]
    for index in range(8):
        path = tmp_path / f'fold_{index}.json'
        experiment = {'variant': 'trasa', 'dataset_kind': 'synthetic', 'seed': 10, 'config': {},
                      'evaluation_implementation_sha256': 'a' * 64, 'map_partitions': {
                          'test': [universe[index]], 'val': [universe[(index + 1) % 8]],
                          'train': [m for m in universe if m not in (universe[index], universe[(index + 1) % 8])]}}
        path.write_text(json.dumps({'schema_version': 2, 'experiment': experiment, 'protocol': evaluation_protocol(),
                                   'maps': [{'map_id': universe[index], 'split': 'test', 'metrics': {'quality': index / 10}}]}))
        paths.append(path)
    result = regenerate(paths)
    assert result['complete_eight_fold'] and result['result_origin'] == 'synthetic_execution'
    summary = result['methods']['trasa']
    assert summary['mean']['quality'] == pytest.approx(.35)
    assert summary['std_population']['quality'] == pytest.approx(np.std(np.arange(8) / 10))
    tables(['--reports', *map(str, paths), '--output', str(tmp_path / 'tables')])
    assert (tmp_path / 'tables/main_results.csv').exists() and (tmp_path / 'tables/ablations.md').exists()
    assert summary['maps'][0]['experiment']['seed'] == 10
    assert len(summary['maps'][0]['report_sha256']) == 64
    mismatched = json.loads(paths[0].read_text())
    mismatched['experiment']['variant'] = 'ldtr'
    split = mismatched['experiment']['map_partitions']
    split['train'][0], split['val'][0] = split['val'][0], split['train'][0]
    alternative = tmp_path / 'different_split.json'
    alternative.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match='same training/validation partition'):
        regenerate(paths + [alternative], allow_incomplete=True)
    mismatched['experiment']['variant'] = 'trasa'
    mismatched['maps'][0]['map_id'] = 'wrong_test_map'
    alternative.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match='held-out test partition'):
        aggregate_reports([alternative])
    changed = json.loads(paths[-1].read_text())
    changed['experiment']['dataset_kind'] = 'benchmark'
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='mix'):
        regenerate(paths)


def test_executable_synthetic_end_to_end_workflow(tmp_path):
    output = tmp_path / 'demo'
    demo(['--output', str(output)])
    assert (output / 'ldtr-ntc/best.pt').exists()
    assert (output / 'ldtr-ntc/unseen_map.json').exists()
    scores = json.loads((output / 'ldtr-ntc/unseen_map_metrics.json').read_text())
    assert np.isfinite(scores['quality']) and 0 <= scores['apls'] <= 1
    assert json.loads((output / 'DEMONSTRATION.json').read_text())['benchmark_reproduction'] is False
