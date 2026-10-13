"""Reference topology labels, direct graph decoding and map-level merging."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


def canonical_edges(edges, num_nodes):
    edges = torch.as_tensor(edges, dtype=torch.long).reshape(-1, 2)
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


def merge_patch_graphs(graphs, origins, patch_size=256, tolerance=5.0, boundary_band=5.0):
    """Merge nearby boundary nodes from different patches; do not fuse interiors.

    Origins use (x,y). Greedy bounded clusters avoid transitive chains that could
    collapse distinct nearby roads. Duplicate edges retain maximum confidence.
    """
    if len(graphs) != len(origins) or tolerance < 0 or boundary_band < 0:
        raise ValueError('Invalid patch origins or merge settings')
    nodes, types, scores, members, edges = [], [], [], [], {}
    for patch_id, (graph, origin) in enumerate(zip(graphs, origins)):
        mapping = []
        for i, local in enumerate(graph.nodes):
            point = np.asarray(local) + origin
            boundary = min(local[0], local[1], patch_size - 1 - local[0], patch_size - 1 - local[1]) <= boundary_band
            candidates = []
            if boundary:
                for j, cluster in enumerate(members):
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
    x0, a, b, y0, d, e = affine
    xy = graph.nodes + 0.5
    nodes = np.column_stack((x0 + a * xy[:, 0] + b * xy[:, 1],
                             y0 + d * xy[:, 0] + e * xy[:, 1]))
    return Graph(nodes, graph.edges.copy(), graph.types.copy(), graph.node_scores.copy(),
                 graph.edge_scores.copy(), {**graph.metadata, 'crs': crs, 'affine': list(affine)})
