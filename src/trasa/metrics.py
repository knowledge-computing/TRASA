"""Map-level geometry/topology metrics with explicit pixel-space settings.

APLS uses bidirectional control-point insertion and shortest-path length errors.
Control spacing/sample cap are implementation choices absent from the manuscript.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, GeometryCollection
from shapely.ops import unary_union

from .graph import Graph


def degrees(graph):
    return np.bincount(graph.edges.flatten(), minlength=len(graph.nodes))


def match_nodes(reference, predicted, tolerance=5):
    """Maximum-cardinality one-to-one nearest assignment within tolerance."""
    if not len(reference) or not len(predicted):
        return np.empty((0, 2), dtype=int)
    distances = cdist(reference, predicted)
    # Dominating cost first maximizes valid matches, then minimizes distance.
    invalid = (max(len(reference), len(predicted)) + 1) * (tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= tolerance, distances, invalid))
    valid = distances[rows, cols] <= tolerance
    return np.column_stack((rows[valid], cols[valid]))


def sparse_graph(graph, weighted=False):
    n = len(graph.nodes)
    if not len(graph.edges):
        return csr_matrix((n, n))
    a, b = graph.edges.T
    values = np.linalg.norm(graph.nodes[a] - graph.nodes[b], axis=1) if weighted else np.ones(len(a))
    return csr_matrix((np.r_[values, values], (np.r_[a, b], np.r_[b, a])), shape=(n, n))


def line_geometry(graph):
    if not len(graph.edges):
        return GeometryCollection()
    return unary_union([LineString(graph.nodes[edge]) for edge in graph.edges])


def geometry_metrics(reference, predicted, tolerance=5):
    ref, pred = line_geometry(reference), line_geometry(predicted)
    completeness = ref.intersection(pred.buffer(tolerance)).length / ref.length if ref.length else float(not pred.length)
    correctness = pred.intersection(ref.buffer(tolerance)).length / pred.length if pred.length else float(not ref.length)
    denominator = completeness + correctness - completeness * correctness
    return {'completeness': completeness, 'correctness': correctness,
            'quality': completeness * correctness / denominator if denominator else 0.0}


def topology_metrics(reference, predicted, tolerance=5, minimum_branch=8):
    ref_degree, pred_degree = degrees(reference), degrees(predicted)
    ref_inter, pred_inter = np.where(ref_degree >= 3)[0], np.where(pred_degree >= 3)[0]
    intersections = match_nodes(reference.nodes[ref_inter], predicted.nodes[pred_inter], tolerance)
    tp = len(intersections)
    precision = tp / len(pred_inter) if len(pred_inter) else float(not len(ref_inter))
    recall = tp / len(ref_inter) if len(ref_inter) else float(not len(pred_inter))
    ida = float(np.mean(ref_degree[ref_inter[intersections[:, 0]]] == pred_degree[pred_inter[intersections[:, 1]]])) if tp else None
    matches = match_nodes(reference.nodes, predicted.nodes, tolerance)
    _, ref_components = connected_components(sparse_graph(reference), directed=False)
    _, pred_components = connected_components(sparse_graph(predicted), directed=False)
    connected, preserved, split = 0, 0, 0
    for component in np.unique(ref_components):
        subset = matches[ref_components[matches[:, 0]] == component]
        n = len(subset)
        connected += n * (n - 1) // 2
        counts = np.unique(pred_components[subset[:, 1]], return_counts=True)[1]
        preserved += sum(int(c) * (int(c) - 1) // 2 for c in counts)
        split += int(len(counts) >= 2)
    # A branch is a maximal path with degree-2 interior, not an individual edge.
    adjacency = [[] for _ in predicted.nodes]
    for edge_id, (a, b) in enumerate(predicted.edges):
        adjacency[a].append((b, edge_id))
        adjacency[b].append((a, edge_id))
    visited, branches = set(), []
    starts = list(np.where(pred_degree != 2)[0]) + list(np.where(pred_degree == 2)[0])
    for start in starts:
        for neighbor, edge_id in adjacency[start]:
            if edge_id in visited:
                continue
            points = [predicted.nodes[start]]
            current = neighbor
            visited.add(edge_id)
            while True:
                points.append(predicted.nodes[current])
                if pred_degree[current] != 2 or current == start:
                    break
                available = [(n, e) for n, e in adjacency[current] if e not in visited]
                if not available:
                    break
                current, edge_id = available[0]
                visited.add(edge_id)
            branches.append(LineString(points))
    buffer = line_geometry(reference).buffer(tolerance)
    # Strict manuscript reading: entire unmatched branch has no buffered support.
    spurious = sum(branch.length > minimum_branch and branch.intersection(buffer).length == 0 for branch in branches)
    return {'intersection_precision': precision, 'intersection_recall': recall,
            'intersection_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'intersection_degree_accuracy': ida,
            'connectivity': preserved / connected if connected else None,
            'split_errors': split, 'spurious_branches': int(spurious),
            'matched_nodes': len(matches), 'eligible_connectivity_pairs': connected}


def _densify(graph, spacing):
    nodes, edges = graph.nodes.tolist(), []
    for a, b in graph.edges:
        length = np.linalg.norm(graph.nodes[a] - graph.nodes[b])
        steps = max(1, int(np.ceil(length / spacing)))
        previous = int(a)
        for step in range(1, steps):
            nodes.append((graph.nodes[a] + (graph.nodes[b] - graph.nodes[a]) * step / steps).tolist())
            edges.append([previous, len(nodes) - 1])
            previous = len(nodes) - 1
        edges.append([previous, int(b)])
    return Graph(np.array(nodes).reshape(-1, 2), np.array(edges, int).reshape(-1, 2),
                 np.empty(0), np.empty(0), np.empty(0))


def _insert_controls(graph, controls, tolerance):
    nodes, mapping = graph.nodes.tolist(), np.full(len(controls), -1, dtype=int)
    splits = {}
    if not len(graph.edges):
        return graph, mapping
    starts = graph.nodes[graph.edges[:, 0]]
    vectors = graph.nodes[graph.edges[:, 1]] - starts
    lengths_squared = (vectors ** 2).sum(1)
    for i, point in enumerate(controls):
        position = np.clip(((point - starts) * vectors).sum(1) / np.maximum(lengths_squared, 1e-12), 0, 1)
        projected = starts + position[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        edge_id = int(distances.argmin())
        if distances[edge_id] > tolerance:
            continue
        t = position[edge_id]
        if t <= 1e-8:
            mapping[i] = graph.edges[edge_id, 0]
        elif t >= 1 - 1e-8:
            mapping[i] = graph.edges[edge_id, 1]
        else:
            existing = next((node for old_t, node in splits.get(edge_id, []) if abs(t - old_t) <= 1e-8), None)
            if existing is None:
                existing = len(nodes)
                nodes.append(projected[edge_id].tolist())
                splits.setdefault(edge_id, []).append((t, existing))
            mapping[i] = existing
    edges = []
    for edge_id, (a, b) in enumerate(graph.edges):
        path = [int(a)] + [node for _, node in sorted(splits.get(edge_id, []))] + [int(b)]
        edges.extend(zip(path[:-1], path[1:]))
    return Graph(np.array(nodes).reshape(-1, 2), np.array(edges, int).reshape(-1, 2),
                 np.empty(0), np.empty(0), np.empty(0)), mapping


def apls(reference, predicted, tolerance=5, spacing=5, max_controls=500, seed=10):
    """Symmetric APLS; missing paths receive unit penalty. Units are pixels.

    This is a disclosed control-point protocol, not a claim that unpublished
    TRASA evaluation code or SAM-Road's meter-based Go settings were recovered.
    """
    if spacing <= 0 or max_controls < 2:
        raise ValueError('Positive spacing and at least two control points required')
    def directional(source, target):
        dense = _densify(source, spacing)
        degree = degrees(dense)
        ids = np.where(degree > 0)[0]
        if len(ids) < 2:
            return None
        if len(ids) > max_controls:
            ids = np.sort(np.random.default_rng(seed).choice(ids, max_controls, replace=False))
        augmented, mapping = _insert_controls(target, dense.nodes[ids], tolerance)
        source_paths = dijkstra(sparse_graph(dense, weighted=True), directed=False, indices=ids)[:, ids]
        destination_paths = np.full_like(source_paths, np.inf)
        valid = np.where(mapping >= 0)[0]
        if len(valid):
            lengths = dijkstra(sparse_graph(augmented, weighted=True), directed=False, indices=mapping[valid])
            destination_paths[np.ix_(valid, valid)] = lengths[:, mapping[valid]]
        pairs = np.triu(np.ones(source_paths.shape, bool), 1) & np.isfinite(source_paths) & (source_paths > 0)
        if not pairs.any():
            return None
        ref, pred = source_paths[pairs], destination_paths[pairs]
        penalty = np.minimum(1, np.abs(ref - pred) / ref)
        return float(1 - penalty.mean())
    forward, reverse = directional(reference, predicted), directional(predicted, reference)
    if forward is None or reverse is None:
        return float(not len(reference.edges) and not len(predicted.edges))
    return 2 * forward * reverse / (forward + reverse) if forward + reverse else 0.0


def type_macro_f1(truth, prediction):
    """Evaluate raw queries, including unmatched no-node, before graph filtering."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    values = []
    for label in range(4):
        tp = np.sum((truth == label) & (prediction == label))
        denominator = np.sum(truth == label) + np.sum(prediction == label)
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def evaluate(reference, predicted, tolerance=5, minimum_branch=8, **apls_settings):
    return {**geometry_metrics(reference, predicted, tolerance),
            **topology_metrics(reference, predicted, tolerance, minimum_branch),
            'apls': apls(reference, predicted, tolerance=tolerance, **apls_settings)}
