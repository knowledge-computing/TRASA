"""Predict held-out patches, merge complete maps and evaluate corrected references."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatchDataset, collate_patches
from .graph import Graph, clip_graph, merge_patch_graphs, node_types, ownership_boxes
from .losses import GraphCriterion
from .metrics import evaluate, evaluation_protocol, type_metrics
from .runtime import load_model, sha256
from .provenance import environment, partitions, source_fingerprint


@torch.no_grad()
def predict_dataset(model, dataset, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    from .graph import decode_graphs
    criterion = GraphCriterion(model.config)
    loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_patches)
    grouped = defaultdict(lambda: {'prediction': [], 'reference': [], 'origins': [], 'type_truth': [], 'type_prediction': []})
    cursor = 0
    for batch in loader:
        guidance = None if batch['guidance'] is None else batch['guidance'].to(device)
        outputs = model(batch['rgb'].to(device), guidance)
        predictions = decode_graphs(model, outputs)
        assignments = criterion.match(outputs, batch['targets']) if model.config.ntc else None
        for index, (predicted, target) in enumerate(zip(predictions, batch['targets'])):
            row = dataset.samples[cursor]
            cursor += 1
            if len(row.get('origin', [])) != 2:
                raise ValueError('Held-out map evaluation requires patch origins [x,y]')
            group = grouped[row['map_id']]
            count = len(target['nodes'])
            reference = Graph(target['nodes'].numpy() * (model.config.patch_size - 1), target['edges'].numpy(),
                              target['types'].numpy(), np.ones(count), np.ones(len(target['edges'])))
            group['prediction'].append(predicted)
            group['reference'].append(reference)
            group['origins'].append(row['origin'])
            if assignments is not None:
                query, truth = assignments[index]
                labels = torch.zeros(model.config.num_queries, dtype=torch.long, device=device)
                roles = node_types(count, target['edges']).to(device)
                labels[query] = roles[truth]
                group['type_truth'].extend(labels.cpu().tolist())
                group['type_prediction'].extend(outputs['pred_types'][index].argmax(-1).cpu().tolist())
    results = {}
    cfg = model.config
    for map_id, group in grouped.items():
        spatial = dataset.document.get('maps', {}).get(map_id, {})
        shape = (spatial['height'], spatial['width']) if 'height' in spatial and 'width' in spatial else None
        boxes = ownership_boxes(group['origins'], cfg.patch_size, shape)
        predicted = merge_patch_graphs([clip_graph(g, b) for g, b in zip(group['prediction'], boxes)], group['origins'],
                                      cfg.patch_size, cfg.merge_tolerance, cfg.boundary_band, boxes)
        reference = merge_patch_graphs([clip_graph(g, b) for g, b in zip(group['reference'], boxes)], group['origins'],
                                      cfg.patch_size, cfg.merge_tolerance, cfg.boundary_band, boxes)
        predicted.metadata.update(spatial)
        reference.metadata.update(spatial)
        metrics = evaluate(reference, predicted, tolerance=5, minimum_branch=8, seed=cfg.seed)
        if cfg.ntc:
            metrics.update(type_metrics(group['type_truth'], group['type_prediction']))
        results[map_id] = {'prediction': predicted, 'reference': reference, 'metrics': metrics}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    model, state = load_model(args.checkpoint, args.device)
    dataset = PatchDataset(args.manifest, args.split, model.config, use_cached_guidance=model.geometry is None)
    expected = state['provenance'].get('guidance_provenance')
    if model.config.guidance and dataset.guidance_provenance != expected:
        raise ValueError('Evaluation geometry provenance differs from training')
    results = predict_dataset(model, dataset, model.config.batch_size)
    identity = {'schema_version': 2, 'model_sha256': sha256(args.checkpoint),
                'manifest_sha256': sha256(args.manifest), 'evaluated_data_sha256': dataset.fingerprint(),
                'geometry_provenance': expected, 'config': model.config.asdict(), 'seed': model.config.seed,
                'map_partitions': partitions(dataset.document),
                'dataset_kind': dataset.document.get('dataset_kind', 'unverified'),
                'variant': state['provenance'].get('variant', 'unverified'), 'checkpoint_epoch': state['epoch'],
                'training_implementation_sha256': state['provenance'].get('implementation_sha256'),
                'evaluation_implementation_sha256': source_fingerprint(), 'environment': environment(args.device)}
    status = Path(args.checkpoint).parent / 'run_status.json'
    identity['training_status'] = json.loads(status.read_text()) if status.exists() else None
    output.mkdir(parents=True)
    reports = []
    for number, (map_id, result) in enumerate(sorted(results.items())):
        result['prediction'].metadata.update({'map_id': map_id, 'model_sha256': identity['model_sha256'], 'coordinate_space': 'pixel'})
        (output / f'map_{number:02d}_prediction.json').write_text(json.dumps(result['prediction'].asdict(), indent=2) + '\n')
        (output / f'map_{number:02d}_reference.json').write_text(json.dumps(result['reference'].asdict(), indent=2) + '\n')
        reports.append({'map_id': map_id, 'split': args.split, 'metrics': result['metrics'],
                        'prediction_artifact': f'map_{number:02d}_prediction.json',
                        'reference_artifact': f'map_{number:02d}_reference.json'})
    (output / 'metrics.json').write_text(json.dumps({'schema_version': 2, 'maps': reports,
        'experiment': identity, 'protocol': evaluation_protocol(seed=model.config.seed)}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
