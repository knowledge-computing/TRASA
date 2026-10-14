"""Reference topology labels, direct graph decoding and map-level merging."""
from collections import deque
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch


def canonical_edges(edges, num_nodes):
    raw = torch.as_tensor(edges)
    if raw.numel() == 0:
        if not (raw.ndim == 1 and raw.shape[0] == 0 or raw.ndim == 2 and raw.shape == (0, 2)):
            raise ValueError('Edges must have shape (E,2), including empty arrays')
        return torch.empty((0, 2), dtype=torch.long, device=raw.device)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError('Edges must have shape (E,2)')
    if raw.dtype == torch.bool or not torch.isfinite(raw).all() or (raw != raw.long()).any():
        raise ValueError('Edge indices must be finite integers')
    edges = raw.long()
    if len(edges) and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError('Edge indices outside the node array')
    if len(edges) and (edges[:, 0] == edges[:, 1]).any():
        raise ValueError('Reference graphs cannot contain self loops')
    return torch.unique(edges.sort(-1).values, dim=0)


def node_types(num_nodes, edges):
    edges = canonical_edges(edges, num_nodes)
    degree = torch.bincount(edges.flatten(), minlength=num_nodes)
    if (degree == 0).any():
        raise ValueError('Isolated reference nodes have no valid paper type; remove or annotate them')
    return torch.where(degree == 1, 1, torch.where(degree == 2, 2, 3)).long()


def connectivity_labels(num_nodes, edges, hops=5):
    """Shortest-path distance in [1,N] inclusive; self/disconnected pairs are false."""
    if hops < 1:
        raise ValueError('hops must be positive')
    adjacency = [[] for _ in range(num_nodes)]
    for a, b in canonical_edges(edges, num_nodes).cpu().tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    labels = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for start in range(num_nodes):
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth == hops:
                continue
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    labels[start, neighbor] = True
                    queue.append((neighbor, depth + 1))
    return labels


@dataclass
class Graph:
    nodes: np.ndarray
    edges: np.ndarray
    types: np.ndarray
    node_scores: np.ndarray
    edge_scores: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        if self.nodes.ndim == 1 and self.nodes.size == 0:
            self.nodes = self.nodes.reshape(0, 2)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2 or not np.isfinite(self.nodes).all():
            raise ValueError('Graph nodes must be finite (N,2) coordinates')
        raw = np.asarray(self.edges)
        if raw.ndim == 1 and raw.size == 0:
            raw = raw.reshape(0, 2)
        checked = canonical_edges(raw, len(self.nodes)).numpy()
        self.types, self.node_scores, self.edge_scores = map(np.asarray, (self.types, self.node_scores, self.edge_scores))
        for name, array, count in (('types', self.types, len(self.nodes)), ('node_scores', self.node_scores, len(self.nodes)),
                                   ('edge_scores', self.edge_scores, len(raw))):
            if array.ndim != 1 or len(array) not in (0, count) or not np.isfinite(array).all():
                raise ValueError(f'Invalid graph {name}')
        if len(raw) and len(self.edge_scores):
            grouped = {}
            for edge, score in zip(np.sort(raw.astype(int), axis=1), self.edge_scores):
                key = tuple(edge)
                grouped[key] = max(grouped.get(key, float('-inf')), float(score))
            self.edge_scores = np.array([grouped[tuple(e)] for e in checked])
        self.edges = checked

    def asdict(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}


@torch.no_grad()
def decode_graphs(model, outputs, image_size=None):
    """Only validity and direct edges construct graphs; NTC is an attribute."""
    config = model.config
    height, width = image_size or (config.patch_size, config.patch_size)
    graphs = []
    for batch, logits in enumerate(outputs['pred_logits']):
        confidence = logits.softmax(-1)[:, 1]
        keep = torch.where(confidence >= config.node_threshold)[0]
        local_pairs = torch.triu_indices(len(keep), len(keep), offset=1, device=keep.device).T
        pairs = keep[local_pairs]
        scores = model.pair_logits(outputs, batch, pairs).softmax(-1)[:, 1]
        retain_edges = scores >= config.edge_threshold
        coords = outputs['pred_nodes'][batch, keep] * logits.new_tensor([width - 1, height - 1])
        types = (outputs['pred_types'][batch, keep].argmax(-1) if 'pred_types' in outputs
                 else torch.full_like(keep, -1))
        graphs.append(Graph(coords.cpu().numpy(), local_pairs[retain_edges].cpu().numpy(),
                            types.cpu().numpy(), confidence[keep].cpu().numpy(),
                            scores[retain_edges].cpu().numpy()))
    return graphs


def ownership_boxes(origins, patch_size=256, image_shape=None):
    """Partition intentional overlaps at midpoints of patch centers."""
    xs, ys = sorted({x for x, y in origins}), sorted({y for x, y in origins})
    boxes = []
    for x, y in origins:
        box = [0., 0., float(patch_size - 1), float(patch_size - 1)]
        for axis, value, ordered in ((0, x, xs), (1, y, ys)):
            index = ordered.index(value)
            if index:
                box[axis] = max(0., (ordered[index - 1] - value + patch_size - 1) / 2)
            if index + 1 < len(ordered):
                box[axis + 2] = min(float(patch_size - 1), (ordered[index + 1] - value + patch_size - 1) / 2)
        if image_shape is not None:
            height, width = image_shape
            box[2] = min(box[2], width - 1 - x)
            box[3] = min(box[3], height - 1 - y)
        boxes.append(box)
    return boxes


def clip_graph(graph, bounds):
    """Clip existing straight edges to a rectangle; never add connections."""
    lo, hi = np.asarray(bounds[:2]), np.asarray(bounds[2:])
    if (hi < lo).any():
        return Graph(np.empty((0, 2)), np.empty((0, 2), int), np.empty(0, int), np.empty(0), np.empty(0))
    nodes, types, scores, edges, edge_scores, mapping = [], [], [], [], [], {}
    for i, point in enumerate(graph.nodes):
        if (point >= lo).all() and (point <= hi).all():
            mapping[i] = len(nodes)
            nodes.append(point)
            types.append(graph.types[i])
            scores.append(graph.node_scores[i])
    for (a, b), score in zip(graph.edges, graph.edge_scores):
        start, vector = graph.nodes[a], graph.nodes[b] - graph.nodes[a]
        lower, upper = 0., 1.
        for axis in range(2):
            if abs(vector[axis]) < 1e-12:
                if start[axis] < lo[axis] or start[axis] > hi[axis]:
                    upper = -1
                    break
            else:
                first, last = sorted(((lo[axis] - start[axis]) / vector[axis], (hi[axis] - start[axis]) / vector[axis]))
                lower, upper = max(lower, first), min(upper, last)
        if upper <= lower or np.linalg.norm(vector) < 1e-12:
            continue
        endpoints = []
        for original, fraction in ((int(a), lower), (int(b), upper)):
            if original in mapping:
                endpoints.append(mapping[original])
            else:
                endpoints.append(len(nodes))
                nodes.append(start + fraction * vector)
                types.append(-1)  # clipping creates no NTC prediction
                scores.append(min(graph.node_scores[a], graph.node_scores[b]))
        edges.append(endpoints)
        edge_scores.append(score)
    return Graph(np.asarray(nodes).reshape(-1, 2), np.asarray(edges, int).reshape(-1, 2),
                 np.asarray(types, int), np.asarray(scores), np.asarray(edge_scores),
                 {**graph.metadata, 'clip_bounds': list(map(float, bounds))})


def merge_patch_graphs(graphs, origins, patch_size=256, tolerance=5.0, boundary_band=5.0, boundary_boxes=None):
    """Merge nearby boundary nodes from different patches; do not fuse interiors.

    Origins use (x,y). Greedy bounded clusters avoid transitive chains that could
    collapse distinct nearby roads. Duplicate edges retain maximum confidence.
    """
    if len(graphs) != len(origins) or not np.isfinite([tolerance, boundary_band]).all() or tolerance < 0 or boundary_band < 0:
        raise ValueError('Invalid patch origins or merge settings')
    if any(np.asarray(o).shape != (2,) or not np.isfinite(o).all() for o in origins):
        raise ValueError('Origins must be finite [x,y] pairs')
    boundary_boxes = boundary_boxes if boundary_boxes is not None else [[0, 0, patch_size - 1, patch_size - 1] for _ in graphs]
    if len(boundary_boxes) != len(graphs):
        raise ValueError('One boundary rectangle required per patch')
    nodes, types, scores, members, edges = [], [], [], [], {}
    bins, cell = defaultdict(list), max(tolerance, 1.)
    for patch_id, (graph, origin) in enumerate(zip(graphs, origins)):
        mapping = []
        xmin, ymin, xmax, ymax = boundary_boxes[patch_id]
        for i, local in enumerate(graph.nodes):
            point = np.asarray(local) + origin
            boundary = min(local[0] - xmin, local[1] - ymin, xmax - local[0], ymax - local[1]) <= boundary_band
            candidates = []
            key = tuple(np.floor(point / cell).astype(int))
            if boundary:
                neighboring = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in bins[(key[0] + dx, key[1] + dy)]]
                for j in neighboring:
                    cluster = members[j]
                    if all(p != patch_id and is_boundary and np.linalg.norm(point - q) <= tolerance
                           for p, q, is_boundary in cluster):
                        candidates.append(j)
            if candidates:
                target = min(candidates, key=lambda j: np.linalg.norm(point - nodes[j]))
                members[target].append((patch_id, point, boundary))
                nodes[target] = np.mean([q for _, q, _ in members[target]], axis=0)
                if graph.node_scores[i] > scores[target]:
                    types[target], scores[target] = graph.types[i], graph.node_scores[i]
            else:
                target = len(nodes)
                nodes.append(point)
                types.append(graph.types[i])
                scores.append(graph.node_scores[i])
                members.append([(patch_id, point, boundary)])
                if boundary:
                    bins[key].append(target)
            mapping.append(target)
        for (a, b), score in zip(graph.edges, graph.edge_scores):
            a, b = mapping[a], mapping[b]
            if a != b:
                edge = tuple(sorted((a, b)))
                edges[edge] = max(edges.get(edge, 0), float(score))
    return Graph(np.asarray(nodes, dtype=float).reshape(-1, 2),
                 np.asarray(list(edges), dtype=int).reshape(-1, 2), np.asarray(types, dtype=int),
                 np.asarray(scores, dtype=float), np.asarray(list(edges.values()), dtype=float))


def georeference_graph(graph, affine, crs):
    """GDAL affine (x0,a,b,y0,d,e); pixel coordinates map to pixel centers."""
    if len(affine) != 6 or not crs:
        raise ValueError('Provide six GDAL affine coefficients and an explicit CRS')
    if not np.isfinite(affine).all() or abs(affine[1] * affine[5] - affine[2] * affine[4]) < 1e-15:
        raise ValueError('Affine must be finite and invertible')
    x0, a, b, y0, d, e = affine
    xy = graph.nodes + 0.5
    nodes = np.column_stack((x0 + a * xy[:, 0] + b * xy[:, 1],
                             y0 + d * xy[:, 0] + e * xy[:, 1]))
    return Graph(nodes, graph.edges.copy(), graph.types.copy(), graph.node_scores.copy(),
                 graph.edge_scores.copy(), {**graph.metadata, 'crs': crs, 'affine': list(affine), 'coordinate_space': 'geospatial'})
