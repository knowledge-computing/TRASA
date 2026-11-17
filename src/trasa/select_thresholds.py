"""Select node/edge confidence thresholds on the validation map (paper Sec. 5.2).

The network runs once over the validation split; every threshold pair is then
decoded, merged into a map graph and scored. The chosen pair is written with
the checkpoint hash so ``predict_dataset --thresholds`` can apply it to the
held-out test map without looking at test data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import PatchDataset
from .predict_dataset import predict_dataset
from .runtime import load_model, sha256

CRITERIA = {
    # Mean of the paper's primary topology metrics (Sec. 5.3), ties broken by APLS.
    'topology': ('apls', 'intersection_f1', 'connectivity'),
    'apls': ('apls',),
    'intersection_f1': ('intersection_f1',),
    'quality': ('quality',),
}


def score(metrics, criterion):
    values = [metrics.get(key) for key in CRITERIA[criterion]]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float('-inf')


def select(results, criterion):
    table = []
    for (node, edge), maps in sorted(results.items()):
        per_map = [score(item['metrics'], criterion) for item in maps.values()]
        apls = [item['metrics'].get('apls') or 0.0 for item in maps.values()]
        table.append({'node_threshold': node, 'edge_threshold': edge, 'score': float(np.mean(per_map)),
                      'apls': float(np.mean(apls)),
                      'metrics': {map_id: item['metrics'] for map_id, item in sorted(maps.items())}})
    # Deterministic order: best score, then APLS, then thresholds closest to 0.5.
    best = max(table, key=lambda r: (round(r['score'], 12), round(r['apls'], 12),
                                     -abs(r['node_threshold'] - .5) - abs(r['edge_threshold'] - .5),
                                     -r['node_threshold'], -r['edge_threshold']))
    return best, table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Fold manifest (cached guidance for guided variants)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New JSON file')
    parser.add_argument('--node-thresholds', type=float, nargs='+', default=[.3, .4, .5, .6, .7])
    parser.add_argument('--edge-thresholds', type=float, nargs='+', default=[.3, .4, .5, .6, .7])
    parser.add_argument('--criterion', choices=sorted(CRITERIA), default='topology')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists; choose a new file')
    model, state = load_model(args.checkpoint, args.device)
    dataset = PatchDataset(args.manifest, 'val', model.config, use_cached_guidance=model.geometry is None)
    expected = state['provenance'].get('guidance_provenance')
    if model.config.guidance and dataset.guidance_provenance != expected:
        raise ValueError('Validation geometry provenance differs from training')
    grid = [(n, e) for n in sorted(set(args.node_thresholds)) for e in sorted(set(args.edge_thresholds))]
    results = predict_dataset(model, dataset, model.config.batch_size, thresholds=grid)
    best, table = select(results, args.criterion)
    record = {'schema_version': 1, 'split': 'val', 'criterion': args.criterion,
              'criterion_metrics': list(CRITERIA[args.criterion]),
              'node_threshold': best['node_threshold'], 'edge_threshold': best['edge_threshold'],
              'validation_score': best['score'], 'validation_maps': sorted({r['map_id'] for r in dataset.samples}),
              'model_sha256': sha256(args.checkpoint), 'manifest_sha256': sha256(args.manifest),
              'grid': [{k: v for k, v in row.items()} for row in table]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: record[k] for k in ('node_threshold', 'edge_threshold', 'validation_score')}))


if __name__ == '__main__':
    main()
