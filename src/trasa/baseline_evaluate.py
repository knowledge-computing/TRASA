"""Evaluate supplied external predictions on the same held-out maps as TRASA."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .data import PatchDataset
from .export import read_graph, write_graph
from .graph import Graph, clip_graph, merge_patch_graphs, ownership_boxes
from .metrics import evaluate, evaluation_protocol
from .provenance import environment, partitions, source_fingerprint
from .runtime import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True, help='JSON index with samples:[{id,graph}] or maps:[{map_id,graph}]')
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Baseline evaluation output exists')
    source = Path(args.predictions).resolve()
    supplied = json.loads(source.read_text())
    dataset = PatchDataset(args.manifest, 'test', Config(guidance=False, ntc=False))
    partition = partitions(dataset.document)
    for key in ('training_map_ids', 'validation_map_ids', 'example_map_ids'):
        if key not in supplied:
            raise ValueError(f'External prediction index must declare {key}; use [] for none')
        if set(supplied[key]) & set(partition['test']):
            raise ValueError(f'External predictions leak held-out test maps through {key}')
    if bool(supplied.get('samples')) == bool(supplied.get('maps')):
        raise ValueError('Supply either patch samples or complete-map predictions')
    patch_predictions = {row['id']: row['graph'] for row in supplied.get('samples', [])}
    map_predictions = {row['map_id']: row['graph'] for row in supplied.get('maps', [])}
    rows = supplied.get('samples', supplied.get('maps'))
    if len(rows) != len(patch_predictions or map_predictions):
        raise ValueError('Duplicate external prediction IDs')
    expected_ids = {row['id'] for row in dataset.samples}
    if patch_predictions and set(patch_predictions) != expected_ids:
        raise ValueError('Predictions must cover exactly the held-out sample IDs')
    if map_predictions and set(map_predictions) != set(partition['test']):
        raise ValueError('Predictions must cover exactly the held-out map IDs')
    groups = defaultdict(lambda: {'reference': [], 'prediction': [], 'origins': []})
    prediction_hashes = {}
    for index, row in enumerate(dataset.samples):
        target = dataset[index]['target']
        count = len(target['nodes'])
        group = groups[row['map_id']]
        group['reference'].append(Graph(target['nodes'].numpy() * 255, target['edges'].numpy(),
                                         target['types'].numpy(), np.ones(count), np.ones(len(target['edges']))))
        group['origins'].append(row['origin'])
        if patch_predictions:
            prediction = source.parent / patch_predictions[row['id']]
            graph = read_graph(prediction)
            if graph.metadata.get('coordinate_space', 'pixel') != 'pixel':
                raise ValueError('Baseline evaluation expects pixel-space predictions')
            group['prediction'].append(graph)
            prediction_hashes[row['id']] = sha256(prediction)
    output.mkdir(parents=True)
    reports = []
    for index, (map_id, group) in enumerate(sorted(groups.items())):
        spatial = dataset.document.get('maps', {}).get(map_id, {})
        shape = (spatial['height'], spatial['width']) if 'height' in spatial and 'width' in spatial else None
        boxes = ownership_boxes(group['origins'], 256, shape)
        reference = merge_patch_graphs([clip_graph(g, b) for g, b in zip(group['reference'], boxes)],
                                       group['origins'], boundary_boxes=boxes)
        if patch_predictions:
            predicted = merge_patch_graphs([clip_graph(g, b) for g, b in zip(group['prediction'], boxes)],
                                           group['origins'], boundary_boxes=boxes)
        else:
            prediction = source.parent / map_predictions[map_id]
            predicted = read_graph(prediction)
            if predicted.metadata.get('coordinate_space', 'pixel') != 'pixel':
                raise ValueError('Baseline evaluation expects pixel-space predictions')
            prediction_hashes[map_id] = sha256(prediction)
        pred_name, ref_name = f'map_{index:02d}_prediction.json', f'map_{index:02d}_reference.json'
        write_graph(predicted, output / pred_name)
        write_graph(reference, output / ref_name)
        reports.append({'map_id': map_id, 'split': 'test', 'metrics': evaluate(reference, predicted, seed=args.seed),
                        'prediction_artifact': pred_name, 'reference_artifact': ref_name})
    external = {key: supplied.get(key) for key in ('source_revision', 'checkpoint_sha256', 'training_map_ids',
                'validation_map_ids', 'example_map_ids', 'prompt_settings', 'access_date')}
    identity = {'schema_version': 2, 'variant': args.method, 'dataset_kind': dataset.document.get('dataset_kind', 'unverified'),
                'seed': args.seed, 'config': supplied.get('configuration', {}), 'geometry_provenance': None,
                'manifest_sha256': sha256(args.manifest), 'evaluated_data_sha256': dataset.fingerprint(),
                'source_prediction_index_sha256': sha256(source), 'prediction_sha256': prediction_hashes,
                'map_partitions': partition, 'external_provenance': external,
                'evaluation_implementation_sha256': source_fingerprint(), 'environment': environment()}
    (output / 'metrics.json').write_text(json.dumps({'schema_version': 2, 'experiment': identity,
        'maps': reports, 'protocol': evaluation_protocol(seed=args.seed)}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
