"""Predict held-out patches, merge complete maps and evaluate corrected references."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PatchDataset, collate_patches
from .graph import Graph, merge_patch_graphs, node_types
from .losses import GraphCriterion
from .metrics import evaluate, type_macro_f1
from .runtime import load_model, sha256


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
        predicted = merge_patch_graphs(group['prediction'], group['origins'], cfg.patch_size, cfg.merge_tolerance, cfg.boundary_band)
        reference = merge_patch_graphs(group['reference'], group['origins'], cfg.patch_size, cfg.merge_tolerance, cfg.boundary_band)
        metrics = evaluate(reference, predicted, tolerance=5, minimum_branch=8, seed=cfg.seed)
        if cfg.ntc:
            metrics['type_macro_f1'] = type_macro_f1(group['type_truth'], group['type_prediction'])
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
    output.mkdir(parents=True)
    reports = []
    for number, (map_id, result) in enumerate(sorted(results.items())):
        result['prediction'].metadata.update({'map_id': map_id, 'model_sha256': sha256(args.checkpoint), 'coordinate_space': 'pixel'})
        (output / f'map_{number:02d}_prediction.json').write_text(json.dumps(result['prediction'].asdict(), indent=2) + '\n')
        (output / f'map_{number:02d}_reference.json').write_text(json.dumps(result['reference'].asdict(), indent=2) + '\n')
        reports.append({'map_id': map_id, 'split': args.split, 'metrics': result['metrics']})
    (output / 'metrics.json').write_text(json.dumps({'maps': reports, 'protocol': {
        'spatial_tolerance_pixels': 5, 'minimum_branch_pixels': 8, 'apls_spacing_pixels': 5,
        'apls_max_controls': 500, 'seed': model.config.seed}}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
