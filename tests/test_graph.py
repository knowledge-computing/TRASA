import numpy as np
import pytest
import torch

from trasa.graph import Graph, connectivity_labels, georeference_graph, merge_patch_graphs, node_types
from trasa.infer import tile_origins


def make_graph(nodes, edges):
    return Graph(np.array(nodes, float).reshape(-1, 2), np.array(edges, int).reshape(-1, 2),
                 np.ones(len(nodes), int), np.ones(len(nodes)), np.ones(len(edges)))


def test_roles_use_patch_degree_and_deduplicate_edges():
    edges = [[0, 1], [1, 0], [1, 2], [1, 3], [3, 4]]
    assert node_types(5, edges).tolist() == [1, 3, 1, 2, 1]
    with pytest.raises(ValueError, match='Isolated'):
        node_types(1, [])
    assert node_types(0, []).shape == (0,)


def test_hops_include_five_but_exclude_six_and_disconnected():
    edges = [[i, i + 1] for i in range(6)]
    labels = connectivity_labels(8, edges, hops=5)
    assert labels[0, 5] and not labels[0, 6] and not labels[0, 7]
    assert not labels.diagonal().any()
    assert torch.equal(labels, labels.T)


def test_merge_only_cross_patch_boundary_nodes_and_duplicate_edges():
    a = make_graph([[250, 20], [255, 20], [100, 100]], [[0, 1]])
    b = make_graph([[0, 20], [5, 20], [100, 100]], [[0, 1]])
    merged = merge_patch_graphs([a, b], [(0, 0), (250, 0)])
    assert len(merged.nodes) == 4
    assert len(merged.edges) == 1
    assert (merged.edges[:, 0] != merged.edges[:, 1]).all()
    interior = make_graph([[100, 100]], [])
    assert len(merge_patch_graphs([interior, interior], [(0, 0), (0, 0)]).nodes) == 2


def test_empty_merge_affine_and_tile_coverage():
    assert merge_patch_graphs([], []).edges.shape == (0, 2)
    graph = make_graph([[0, 0], [1, 2]], [[0, 1]])
    transformed = georeference_graph(graph, [100, 2, 0, 200, 0, -2], 'EPSG:4326')
    np.testing.assert_allclose(transformed.nodes, [[101, 199], [103, 195]])
    assert tile_origins(300, 300, 256, 256) == [(0, 0), (256, 0), (0, 256), (256, 256)]
    with pytest.raises(ValueError):
        tile_origins(300, 300, 256, 300)
