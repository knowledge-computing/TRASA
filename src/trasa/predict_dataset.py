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


def reference_graph(target, patch_size):
    """Patch reference graph in pixels, including road types when annotated."""
    count = len(target['nodes'])
    classes = target['edge_classes'].numpy() if 'edge_classes' in target else np.empty(0, int)
    return Graph(target['nodes'].numpy() * (patch_size - 1), target['edges'].numpy(), target['types'].numpy(),
                 np.ones(count), np.ones(len(target['edges'])), edge_classes=classes)


def merge_map(graphs, origins, config, spatial):
    shape = (spatial['height'], spatial['width']) if 'height' in spatial and 'width' in spatial else None
    boxes = ownership_boxes(origins, config.patch_size, shape)
    return merge_patch_graphs([clip_graph(g, b) for g, b in zip(graphs, boxes)], origins,
                              config.patch_size, config.merge_tolerance, config.boundary_band, boxes)


@torch.no_grad()
def predict_dataset(model, dataset, batch_size=8, thresholds=None, metrics=True):
    """Predict, merge and evaluate complete held-out maps.

    ``thresholds`` is an optional list of (node, edge) confidence pairs; the
    network runs once and every pair is decoded. Without it the checkpoint
    thresholds are used and the result is keyed by map ID only.
    """
    model.eval()
    device = next(model.parameters()).device
    from .graph import decode_graphs
    cfg = model.config
    pairs = [(cfg.node_threshold, cfg.edge_threshold)] if thresholds is None else [tuple(map(float, t)) for t in thresholds]
    for node_threshold, edge_threshold in pairs:
        if not (0 <= node_threshold <= 1 and 0 <= edge_threshold <= 1):
            raise ValueError('Thresholds must be in [0,1]')
    criterion = GraphCriterion(cfg)
    loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_patches)
    grouped = defaultdict(lambda: {'prediction': defaultdict(list), 'reference': [], 'origins': [],
                                   'type_truth': [], 'type_prediction': []})
    cursor = 0
    for batch in loader:
        guidance = None if batch['guidance'] is None else batch['guidance'].to(device)
        outputs = model(batch['rgb'].to(device), guidance)
        decoded = {pair: decode_graphs(model, outputs, node_threshold=pair[0], edge_threshold=pair[1]) for pair in pairs}
        assignments = criterion.match(outputs, batch['targets']) if cfg.ntc else None
        for index, target in enumerate(batch['targets']):
            row = dataset.samples[cursor]
            cursor += 1
            if len(row.get('origin', [])) != 2:
                raise ValueError('Held-out map evaluation requires patch origins [x,y]')
            group = grouped[row['map_id']]
            count = len(target['nodes'])
            for pair in pairs:
                group['prediction'][pair].append(decoded[pair][index])
            group['reference'].append(reference_graph(target, cfg.patch_size))
            group['origins'].append(row['origin'])
            if assignments is not None:
                query, truth = assignments[index]
                labels = torch.zeros(cfg.num_queries, dtype=torch.long, device=device)
                roles = node_types(count, target['edges']).to(device)
                labels[query] = roles[truth]
                group['type_truth'].extend(labels.cpu().tolist())
                group['type_prediction'].extend(outputs['pred_types'][index].argmax(-1).cpu().tolist())
    results = {pair: {} for pair in pairs}
    for map_id, group in grouped.items():
        spatial = dataset.document.get('maps', {}).get(map_id, {})
        reference = merge_map(group['reference'], group['origins'], cfg, spatial)
        reference.metadata.update(spatial)
        type_scores = type_metrics(group['type_truth'], group['type_prediction']) if cfg.ntc else {}
        for pair in pairs:
            predicted = merge_map(group['prediction'][pair], group['origins'], cfg, spatial)
            predicted.metadata.update({**spatial, 'node_threshold': pair[0], 'edge_threshold': pair[1]})
            scores = evaluate(reference, predicted, tolerance=5, minimum_branch=8, seed=cfg.seed) if metrics else {}
            scores.update(type_scores)
            results[pair][map_id] = {'prediction': predicted, 'reference': reference, 'metrics': scores}
    return results[pairs[0]] if thresholds is None else results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--thresholds', help='JSON from trasa.select_thresholds; overrides checkpoint node/edge thresholds')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    model, state = load_model(args.checkpoint, args.device)
    selection = None
    if args.thresholds:
        selection = json.loads(Path(args.thresholds).read_text())
        if selection.get('model_sha256') != sha256(args.checkpoint):
            raise ValueError('Threshold selection was made for a different checkpoint')
        if selection.get('split') != 'val':
            raise ValueError('Thresholds must be selected on the validation split')
        from dataclasses import replace
        model.config = replace(model.config, node_threshold=selection['node_threshold'],
                               edge_threshold=selection['edge_threshold'])
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
    identity['threshold_selection'] = selection
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
