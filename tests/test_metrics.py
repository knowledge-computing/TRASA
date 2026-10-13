import numpy as np

from trasa.graph import Graph
from trasa.metrics import apls, evaluate, match_nodes, type_macro_f1


def graph(nodes, edges):
    return Graph(np.array(nodes, float).reshape(-1, 2), np.array(edges, int).reshape(-1, 2),
                 np.ones(len(nodes), int), np.ones(len(nodes)), np.ones(len(edges)))


def test_perfect_graph_and_identical_geometry_different_subdivision():
    ref = graph([[0, 0], [20, 0]], [[0, 1]])
    pred = graph([[0, 0], [10, 0], [20, 0]], [[0, 1], [1, 2]])
    assert apls(ref, pred) == 1
    scores = evaluate(ref, ref)
    for key in ('completeness', 'correctness', 'quality', 'apls', 'connectivity'):
        assert scores[key] == 1
    assert scores['split_errors'] == scores['spurious_branches'] == 0


def test_split_and_unsupported_branch_have_topological_penalties():
    ref = graph([[0, 0], [10, 0], [20, 0], [30, 0]], [[0, 1], [1, 2], [2, 3]])
    pred = graph([[0, 0], [10, 0], [20, 0], [30, 0], [0, 100], [20, 100]], [[0, 1], [2, 3], [4, 5]])
    scores = evaluate(ref, pred)
    assert scores['split_errors'] == 1 and scores['spurious_branches'] == 1
    assert scores['connectivity'] == 1/3
    assert 0 <= scores['apls'] < 1


def test_degree_accuracy_uses_graph_not_predicted_types():
    ref = graph([[10, 10], [0, 10], [20, 10], [10, 0]], [[0, 1], [0, 2], [0, 3]])
    ref.types[:] = 0
    result = evaluate(ref, ref)
    assert result['intersection_f1'] == result['intersection_degree_accuracy'] == 1
    assert type_macro_f1([0, 1, 2, 3], [0, 1, 2, 3]) == 1


def test_matching_is_one_to_one_and_threshold_constrained():
    pairs = match_nodes([[0, 0], [1, 0]], [[0.5, 0], [100, 0]], 5)
    assert len(pairs) == 1
    assert len(np.unique(pairs[:, 1])) == 1
