"""Convert real external baseline predictions to the shared pixel graph protocol."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .export import read_graph, write_graph
from .graph import Graph
from .metrics import degrees
from .prepare_data import read_vtp
from .runtime import sha256


def mask_to_graph(probabilities, threshold=.5, min_component=20, min_spur=8):
    from skimage.morphology import remove_small_objects, skeletonize
    mask = np.asarray(probabilities, float)
    if mask.ndim != 2 or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('Mask must be a finite 2D probability array in [0,1]')
    if not 0 <= threshold <= 1 or min_component < 1 or min_spur < 0:
        raise ValueError('Invalid baseline postprocessing settings')
    skeleton = skeletonize(remove_small_objects(mask >= threshold, min_size=min_component, connectivity=2))
    pixels = list(map(tuple, np.argwhere(skeleton)))
    lookup = {point: i for i, point in enumerate(pixels)}
    neighbors = [[] for _ in pixels]
    for i, (y, x) in enumerate(pixels):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = lookup.get((y + dy, x + dx))
                if (dy or dx) and j is not None:
                    neighbors[i].append(j)
    active = set(range(len(pixels)))
    while True:
        remove = set()
        adjacency = {i: [j for j in neighbors[i] if j in active] for i in active}
        for start in sorted(active):
            if len(adjacency[start]) != 1:
                continue
            path, previous, node, length = [start], start, adjacency[start][0], 0.
            while True:
                length += float(np.linalg.norm(np.subtract(pixels[node], pixels[previous])))
                if len(adjacency[node]) != 2:
                    break
                path.append(node)
                following = next(j for j in adjacency[node] if j != previous)
                previous, node = node, following
            if len(adjacency[node]) >= 3 and length < min_spur:
                remove.update(path)
        if not remove:
            break
        active -= remove
    adjacency = {i: [j for j in neighbors[i] if j in active] for i in active}
    junctions = {i for i in active if len(adjacency[i]) >= 3}
    groups, remaining = [], set(junctions)
    while remaining:
        group, queue = set(), [min(remaining)]
        while queue:
            i = queue.pop()
            if i in group:
                continue
            group.add(i)
            queue.extend(j for j in adjacency[i] if j in junctions and j not in group)
        remaining -= group
        groups.append(sorted(group))
    groups.extend([[i] for i in sorted(active - junctions)])
    mapping = {i: index for index, group in enumerate(groups) for i in group}
    nodes = [np.mean([pixels[i][::-1] for i in group], axis=0) for group in groups]
    edges = sorted({tuple(sorted((mapping[i], mapping[j]))) for i in active for j in adjacency[i] if mapping[i] != mapping[j]})
    graph = Graph(np.array(nodes).reshape(-1, 2), np.array(edges, int).reshape(-1, 2),
                  np.full(len(nodes), -1), np.ones(len(nodes)), np.ones(len(edges)))
    graph.metadata['postprocessing'] = {'threshold': threshold, 'min_component_pixels': min_component,
                                        'min_spur_pixels': min_spur, 'skeletonization': 'skimage_zhang',
                                        'junctions': 'connected_degree_ge3_pixel_clusters',
                                        'scores': 'binary_retained_geometry_not_model_confidence'}
    return graph


def polylines_to_mask(document, patch_size=256, road_width=3):
    if road_width < 1:
        raise ValueError('Positive road width required')
    lines = document.get('polylines')
    if not isinstance(lines, list):
        raise ValueError('Expected a polylines list; malformed responses are not converted to fabricated empty graphs')
    image, discarded = Image.new('L', (patch_size, patch_size)), 0
    for line in lines:
        try:
            points = np.asarray(line, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError('Malformed polyline')
            points = np.clip(points, 0, patch_size - 1)
            ImageDraw.Draw(image).line([tuple(p) for p in points], fill=255, width=road_width)
        except (TypeError, ValueError):
            discarded += 1
    return np.asarray(image, float) / 255, discarded


def adjacency_to_graph(adjacency, axis_order):
    if axis_order not in ('xy', 'yx'):
        raise ValueError('Declare xy or yx coordinates explicitly')
    nodes = sorted(set(adjacency) | {point for neighbors in adjacency.values() for point in neighbors})
    lookup = {point: index for index, point in enumerate(nodes)}
    edges = sorted({tuple(sorted((lookup[a], lookup[b]))) for a, neighbors in adjacency.items() for b in neighbors if a != b})
    coordinates = np.asarray(nodes, float).reshape(-1, 2)
    if axis_order == 'yx':
        coordinates = coordinates[:, ::-1]
    return Graph(coordinates, edges, np.full(len(nodes), -1), np.ones(len(nodes)), np.ones(len(edges)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', required=True, choices=['graph-json', 'mask', 'polylines-json', 'vtp', 'sat2graph-pickle'])
    parser.add_argument('--method', required=True, help='Actual external method identity; this command does not run a model')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--coordinates', choices=['pixels', 'normalized'])
    parser.add_argument('--axis-order', choices=['xy', 'yx'])
    parser.add_argument('--trusted-pickle', action='store_true')
    parser.add_argument('--road-width', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--min-component', type=int, default=20)
    parser.add_argument('--min-spur', type=float, default=8)
    parser.add_argument('--checkpoint')
    parser.add_argument('--source-revision')
    args = parser.parse_args(argv)
    discarded = 0
    if args.kind == 'graph-json':
        graph = read_graph(args.input)
        if graph.metadata.get('coordinate_space', 'pixel') != 'pixel':
            raise ValueError('Convert geospatial baseline coordinates to pixels before shared evaluation')
        keep = graph.node_scores >= args.threshold
        selected = keep[graph.edges].all(1) & (graph.edge_scores >= args.threshold)
        mapping = np.cumsum(keep) - 1
        graph = Graph(graph.nodes[keep], mapping[graph.edges[selected]], graph.types[keep],
                      graph.node_scores[keep], graph.edge_scores[selected], graph.metadata)
    elif args.kind in ('mask', 'polylines-json'):
        if args.kind == 'mask':
            with Image.open(args.input) as image:
                if image.mode not in ('1', 'L', 'F'):
                    raise ValueError('Mask must be a single-channel probability or binary image')
                mask = np.asarray(image, float)
            if mask.max(initial=0) > 1:
                mask /= 255
        else:
            mask, discarded = polylines_to_mask(json.loads(Path(args.input).read_text()), args.patch_size, args.road_width)
        graph = mask_to_graph(mask, args.threshold, args.min_component, args.min_spur)
    elif args.kind == 'vtp':
        if args.coordinates is None:
            raise ValueError('VTP requires explicit --coordinates pixels or normalized')
        nodes, edges = read_vtp(args.input)
        if args.coordinates == 'normalized':
            nodes *= args.patch_size - 1
        graph = Graph(nodes, edges, np.full(len(nodes), -1), np.ones(len(nodes)), np.ones(len(edges)))
    else:
        if not args.trusted_pickle or args.axis_order is None:
            raise ValueError('Native pickle import requires --trusted-pickle and explicit --axis-order')
        with Path(args.input).open('rb') as stream:
            graph = adjacency_to_graph(pickle.load(stream), args.axis_order)
    graph.metadata.update({'coordinate_space': 'pixel', 'external_method': args.method,
        'source_prediction_sha256': sha256(args.input), 'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'source_revision': args.source_revision, 'discarded_polylines': discarded,
        'road_width_pixels': args.road_width if args.kind == 'polylines-json' else None,
        'provenance_status': 'external_declared'})
    write_graph(graph, args.output)


if __name__ == '__main__':
    main()
