"""Evaluate already merged pixel-coordinate graphs against corrected references."""
import argparse
import json
from pathlib import Path

import numpy as np

from .graph import Graph, canonical_edges
from .metrics import evaluate, evaluation_protocol


def load_graph(path):
    doc = json.loads(Path(path).read_text())
    metadata = doc.get('metadata', {})
    if metadata.get('coordinate_space', 'pixel') != 'pixel':
        raise ValueError('Paper evaluation expects pixel graphs, not geospatial coordinates')
    nodes = np.asarray(doc['nodes'], float)
    if nodes.ndim == 1 and nodes.size == 0:
        nodes = nodes.reshape(0, 2)
    if not np.isfinite(nodes).all():
        raise ValueError('Non-finite graph coordinates')
    edges = canonical_edges(doc['edges'], len(nodes)).numpy()
    return Graph(nodes, edges, np.asarray(doc.get('types', [-1] * len(nodes))),
                 np.ones(len(nodes)), np.ones(len(edges)), metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tolerance', type=float, default=5)
    parser.add_argument('--minimum-branch', type=float, default=8)
    parser.add_argument('--apls-spacing', type=float, default=5)
    parser.add_argument('--apls-max-controls', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise ValueError('Output exists; choose a new file')
    result = evaluate(load_graph(args.reference), load_graph(args.prediction),
                      args.tolerance, args.minimum_branch, spacing=args.apls_spacing,
                      max_controls=args.apls_max_controls, seed=args.seed)
    result['protocol'] = evaluation_protocol(args.tolerance, args.minimum_branch, args.apls_spacing, args.apls_max_controls, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
