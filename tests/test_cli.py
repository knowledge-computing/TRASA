import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml

from trasa.config import Config
from trasa.evaluate import main as evaluate_main
from trasa.infer import main as infer_main
from trasa.train import main as train_main

from test_data_runtime import manifest


def test_train_resume_infer_evaluate_cli(tmp_path):
    dataset, _ = manifest(tmp_path)
    cfg = asdict(Config(pretrained_backbone=False, hidden_dim=64, num_queries=8,
                        encoder_layers=1, decoder_layers=1, feedforward_dim=128, epochs=1, batch_size=1))
    config_path = tmp_path / 'smoke.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / 'run'
    args = ['--config', str(config_path), '--manifest', str(dataset), '--output', str(run), '--device', 'cpu']
    train_main(args)
    assert (run / 'best.pt').exists() and (run / 'last.pt').exists()
    assert len((run / 'history.jsonl').read_text().splitlines()) == 1
    train_main(args + ['--resume', str(run / 'last.pt')])
    assert len((run / 'history.jsonl').read_text().splitlines()) == 1
    predicted = tmp_path / 'prediction.json'
    infer_main(['--checkpoint', str(run / 'best.pt'), '--image', str(tmp_path / 'image.png'),
                '--guidance', str(tmp_path / 'masks.npz'), '--output', str(predicted)])
    graph = json.loads(predicted.read_text())
    assert {'nodes', 'edges', 'types', 'node_scores', 'edge_scores', 'metadata'} <= graph.keys()
    assert graph['metadata']['coordinate_space'] == 'pixel'
    evaluation = tmp_path / 'metrics.json'
    evaluate_main(['--reference', str(tmp_path / 'graph.json'), '--prediction', str(predicted), '--output', str(evaluation)])
    metrics = json.loads(evaluation.read_text())
    assert 0 <= metrics['apls'] <= 1
    # Changed file content must invalidate resume even if the manifest is unchanged.
    np.savez(tmp_path / 'masks.npz', road=np.zeros((256, 256)), intersection=np.zeros((256, 256)))
    with pytest.raises(ValueError, match='provenance changed'):
        train_main(args + ['--resume', str(run / 'last.pt')])
