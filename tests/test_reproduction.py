import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from trasa.benchmark import aggregate_reports, main as benchmark_main
from trasa.config import Config
from trasa.data import PatchDataset
from trasa.folds import fold_schedules, main as folds_main
from trasa.geometry import SAMRoadGeometry
from trasa.model import TRASA
from trasa.predict_dataset import predict_dataset
from trasa.prepare_data import convert_graph, main as prepare_main
from trasa.train_geometry import GeometryDataset, geometry_epoch, geometry_targets, initialize_encoder
from trasa.train_geometry import main as geometry_train_main
from trasa.cache_guidance import main as cache_main

from test_data_runtime import manifest


def test_ascii_polyline_conversion_and_explicit_coordinates(tmp_path):
    path = tmp_path / 'graph.vtp'
    path.write_text('''<VTKFile type="PolyData"><PolyData><Piece NumberOfPoints="3"><Points>
    <DataArray NumberOfComponents="3" format="ascii">0 0 0 .5 .5 0 1 1 0</DataArray></Points>
    <Lines><DataArray Name="connectivity" format="ascii">0 1 2</DataArray>
    <DataArray Name="offsets" format="ascii">3</DataArray></Lines></Piece></PolyData></VTKFile>''')
    converted = convert_graph(path, 'normalized')
    assert converted['nodes'] == [[0, 0], [127.5, 127.5], [255, 255]]
    assert converted['edges'] == [[0, 1], [1, 2]] and converted['types'] == [1, 2, 1]
    with pytest.raises(ValueError, match='explicitly'):
        convert_graph(path, 'guess')


def test_compressed_vtp_conversion(tmp_path):
    pytest.importorskip('vtkmodules.vtkIOXML')
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    points, cells = vtkPoints(), vtkCellArray()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 1, 0)
    cells.InsertNextCell(2)
    cells.InsertCellPoint(0)
    cells.InsertCellPoint(1)
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetLines(cells)
    path = tmp_path / 'compressed.vtp'
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mesh)
    writer.SetDataModeToBinary()
    assert writer.Write() == 1
    converted = convert_graph(path, 'normalized')
    assert converted['nodes'] == [[0, 0], [255, 255]] and converted['edges'] == [[0, 1]]


def test_prepare_data_preserves_sources_and_validates_map_origins(tmp_path):
    path, doc = manifest(tmp_path)
    for row in doc['samples']:
        row['origin'] = [0, 0]
    path.write_text(json.dumps(doc))
    output = tmp_path / 'prepared'
    prepare_main(['--index', str(path), '--output', str(output), '--coordinates', 'pixels', '--copy-images'])
    assert (tmp_path / 'graph.json').exists()
    dataset = PatchDataset(output / 'manifest.json', 'train', Config(guidance=False))
    assert dataset[0]['target']['types'].tolist() == [1, 1]
    assert dataset.samples[0]['origin'] == [0, 0]


def test_eight_map_schedule_and_benchmark_dry_run(tmp_path, capsys):
    schedules = fold_schedules(list('ABCDEFGH'))
    assert {s['test'] for s in schedules} == set('ABCDEFGH')
    assert all(len(s['train']) == 6 and s['test'] != s['val'] for s in schedules)
    source = tmp_path / 'dataset.json'
    source.write_text(json.dumps({'schema_version': 1, 'samples': [{'map_id': m, 'id': m, 'image': f'{m}.png', 'graph': 'x.json', 'origin': [0, 0], 'split': 'train'} for m in 'ABCDEFGH']}))
    output = tmp_path / 'folds'
    folds_main(['--manifest', str(source), '--output', str(output)])
    manifests = sorted(output.glob('fold_*.json'))
    benchmark_main(['--fold-manifests', *map(str, manifests), '--output', str(tmp_path / 'runs'), '--variants', 'trasa', '--dry-run'])
    assert len(capsys.readouterr().out.splitlines()) == 16
    assert not (tmp_path / 'runs').exists()
    changed = json.loads(manifests[1].read_text())
    for row in changed['samples']:
        if row['split'] == 'train' and row['map_id'] == 'A':
            row['map_id'] = 'unexpected_map'
    manifests[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='same eight'):
        benchmark_main(['--fold-manifests', *map(str, manifests), '--output', str(tmp_path / 'runs'), '--dry-run'])


class TrainingEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(3, 256, 1)

    def forward(self, image):
        return self.projection(torch.nn.functional.avg_pool2d(image, 16))


def test_geometry_pretraining_and_frozen_checkpoint_contract(monkeypatch, tmp_path):
    monkeypatch.setattr('trasa.geometry.ImageEncoderViT', TrainingEncoder)
    path, _ = manifest(tmp_path)
    dataset = GeometryDataset(path, 'train')
    model = SAMRoadGeometry()
    sam = tmp_path / 'sam.pth'
    torch.save({'image_encoder.' + k: v for k, v in model.image_encoder.state_dict().items()}, sam)
    initialize_encoder(model, sam)
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.image_encoder.projection.weight.detach().clone()
    loader = torch.utils.data.DataLoader(dataset, 1)
    loss = geometry_epoch(model, loader, 'cpu', optimizer)
    assert np.isfinite(loss) and not torch.equal(before, model.image_encoder.projection.weight)
    checkpoint = tmp_path / 'trained.ckpt'
    torch.save({'state_dict': model.state_dict()}, checkpoint)
    frozen = SAMRoadGeometry().load_checkpoint(checkpoint)
    masks = frozen(dataset[0][0][None])
    assert masks.shape == (1, 2, 256, 256) and not masks.requires_grad
    assert not any(p.requires_grad for p in frozen.parameters())


def test_geometry_keypoint_semantics_do_not_confuse_endpoints_with_intersections():
    target = {'nodes': torch.tensor([[.1, .1], [.5, .5], [.9, .9]]),
              'edges': torch.tensor([[0, 1], [1, 2]]), 'types': torch.tensor([1, 2, 1])}
    keypoints = geometry_targets(target, semantics='degree_ne_2')[0]
    intersections = geometry_targets(target, semantics='intersection_only')[0]
    assert keypoints.sum() > 0 and intersections.sum() == 0


def test_corrected_pixel_road_labels_are_preserved_and_fingerprinted(tmp_path):
    path, doc = manifest(tmp_path)
    road = np.zeros((256, 256), np.uint8)
    road[30:40, :] = 255
    Image.fromarray(road).save(tmp_path / 'road.png')
    for row in doc['samples']:
        row['origin'] = [0, 0]
        row['road_mask'] = 'road.png'
    path.write_text(json.dumps(doc))
    dataset = GeometryDataset(path, 'train')
    torch.testing.assert_close(dataset[0][1][1], torch.tensor(road / 255, dtype=torch.float32))
    old_hash = dataset.patches.fingerprint()
    output = tmp_path / 'prepared'
    prepare_main(['--index', str(path), '--output', str(output), '--coordinates', 'pixels', '--copy-images'])
    converted = json.loads((output / 'manifest.json').read_text())
    assert (output / converted['samples'][0]['road_mask']).exists()
    Image.fromarray((road > 0).astype(np.uint8)).save(tmp_path / 'road.png')
    torch.testing.assert_close(dataset[0][1][1], torch.tensor(road / 255, dtype=torch.float32))
    Image.fromarray(np.zeros_like(road)).save(tmp_path / 'road.png')
    assert dataset.patches.fingerprint() != old_hash


def test_geometry_cli_resume_then_cache_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr('trasa.geometry.ImageEncoderViT', TrainingEncoder)
    path, _ = manifest(tmp_path)
    model = SAMRoadGeometry()
    sam = tmp_path / 'sam.pth'
    torch.save({'image_encoder.' + k: v for k, v in model.image_encoder.state_dict().items()}, sam)
    run = tmp_path / 'geometry'
    args = ['--manifest', str(path), '--sam-checkpoint', str(sam), '--output', str(run), '--epochs', '1', '--batch-size', '1']
    geometry_train_main(args)
    geometry_train_main(args + ['--epochs', '2', '--resume', str(run / 'last.ckpt')])
    assert len((run / 'history.jsonl').read_text().splitlines()) == 2
    uninterrupted = tmp_path / 'uninterrupted'
    geometry_train_main(args + ['--epochs', '2', '--output', str(uninterrupted)])
    resumed = torch.load(run / 'last.ckpt', weights_only=True)
    complete = torch.load(uninterrupted / 'last.ckpt', weights_only=True)
    for key, value in resumed['state_dict'].items():
        torch.testing.assert_close(value, complete['state_dict'][key], rtol=0, atol=0)
    cache = tmp_path / 'cache'
    cache_main(['--manifest', str(path), '--checkpoint', str(run / 'best.ckpt'), '--output', str(cache)])
    doc = json.loads((cache / 'manifest.json').read_text())
    assert doc['guidance_provenance']['training_map_ids'] == ['A']
    assert not Path(doc['samples'][0]['image']).is_absolute()
    assert PatchDataset(cache / 'manifest.json', 'train', Config())[0]['guidance'].shape == (2, 256, 256)
    with pytest.raises(ValueError, match='provenance'):
        cache_main(['--manifest', str(path), '--checkpoint', str(run / 'best.ckpt'), '--output', str(tmp_path / 'wrong'), '--training-map-ids', 'B'])


def test_held_out_map_prediction_includes_query_type_f1(tmp_path):
    path, doc = manifest(tmp_path)
    for row in doc['samples']:
        row['origin'] = [0, 0]
    path.write_text(json.dumps(doc))
    cfg = replace(Config(), pretrained_backbone=False, hidden_dim=64, encoder_layers=1, decoder_layers=1, num_queries=8)
    model = TRASA(cfg)
    dataset = PatchDataset(path, 'val', cfg)
    result = predict_dataset(model, dataset, 1)
    assert set(result) == {'B'}
    assert 0 <= result['B']['metrics']['type_macro_f1'] <= 1
    assert len(result['B']['reference'].nodes) == 2


def test_fold_aggregation_arithmetic_mean_and_protocol_validation(tmp_path):
    paths = []
    for number, value in enumerate((.2, .8)):
        path = tmp_path / f'{number}.json'
        path.write_text(json.dumps({'maps': [{'map_id': str(number), 'metrics': {'apls': value}}], 'protocol': {'tolerance': 5}}))
        paths.append(path)
    assert aggregate_reports(paths)['mean']['apls'] == .5
    paths[1].write_text(json.dumps({'maps': [], 'protocol': {'tolerance': 10}}))
    with pytest.raises(ValueError, match='protocols'):
        aggregate_reports(paths)
