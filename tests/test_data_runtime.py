import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from trasa.config import Config
from trasa.data import PatchDataset, collate_patches
from trasa.model import TRASA
from trasa.losses import GraphCriterion
from trasa.runtime import load_model, restore_training_state, save_checkpoint, seed_everything
from trasa.train import run_epoch


def manifest(tmp_path):
    Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(tmp_path / 'image.png')
    (tmp_path / 'graph.json').write_text(json.dumps({'nodes': [[0, 10], [255, 20]], 'edges': [[0, 1]], 'types': [1, 1]}))
    np.savez(tmp_path / 'masks.npz', road=np.ones((256, 256), np.float32) * .25,
             intersection=np.ones((256, 256), np.float32) * .75)
    doc = {'schema_version': 1, 'guidance_provenance': {'checkpoint_sha256': 'a' * 64,
           'channel_order': ['road', 'intersection'], 'keypoint_semantics': 'degree_ne_2', 'training_map_ids': ['A']},
           'samples': [{'id': 'a', 'map_id': 'A', 'split': 'train', 'image': 'image.png',
                        'graph': 'graph.json', 'guidance': 'masks.npz'},
                       {'id': 'b', 'map_id': 'B', 'split': 'val', 'image': 'image.png',
                        'graph': 'graph.json', 'guidance': 'masks.npz'}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(doc))
    return path, doc


def test_data_coordinates_probabilities_and_split_leakage(tmp_path):
    path, doc = manifest(tmp_path)
    dataset = PatchDataset(path, 'train', Config())
    row = dataset[0]
    torch.testing.assert_close(row['target']['nodes'], torch.tensor([[0., 10/255], [1., 20/255]]))
    assert row['guidance'][0, 0, 0] == .25 and row['guidance'][1, 0, 0] == .75
    doc['samples'][1]['map_id'] = 'A'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='leakage'):
        PatchDataset(path, 'train', Config())


def test_guidance_training_leakage(tmp_path):
    path, doc = manifest(tmp_path)
    doc['guidance_provenance']['training_map_ids'].append('B')
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='held-out'):
        PatchDataset(path, 'train', Config())


def test_checkpoint_roundtrip_rng_and_optimizer_step(tmp_path):
    seed_everything(10)
    cfg = replace(Config(), pretrained_backbone=False, hidden_dim=64,
                  encoder_layers=1, decoder_layers=1, num_queries=8, feedforward_dim=128)
    path, _ = manifest(tmp_path)
    dataset = PatchDataset(path, 'train', cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(dataset, 1, collate_fn=collate_patches, generator=generator)
    model = TRASA(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    before = model.valid_head.weight.detach().clone()
    metrics = run_epoch(model, loader, GraphCriterion(cfg), 'cpu', optimizer)
    assert np.isfinite(metrics['total']) and not torch.equal(before, model.valid_head.weight)
    model.eval()
    row = dataset[0]
    inputs = (row['rgb'][None], row['guidance'][None])
    with torch.no_grad():
        expected = model(*inputs)['pred_logits']
    checkpoint = tmp_path / 'last.pt'
    save_checkpoint(checkpoint, model, optimizer, 0, generator, {'test': True}, metrics['total'])
    expected_rng = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
    recovered, state = load_model(checkpoint)
    optimizer2 = torch.optim.AdamW(recovered.parameters())
    restore_training_state(state, optimizer2, generator)
    actual_rng = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator))
    assert actual_rng[:2] == expected_rng[:2]
    torch.testing.assert_close(actual_rng[2], expected_rng[2])
    torch.testing.assert_close(actual_rng[3], expected_rng[3])
    recovered.eval()
    with torch.no_grad():
        torch.testing.assert_close(recovered(*inputs)['pred_logits'], expected)
    assert len(optimizer2.state) > 0
