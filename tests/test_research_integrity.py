import importlib.util
import json
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from trasa.config import Config
from trasa.data import PatchDataset
from trasa.graph import Graph, canonical_edges, clip_graph, merge_patch_graphs, ownership_boxes
from trasa.metrics import apls_control_points, evaluate, evaluation_protocol, type_metrics
from trasa.model import TRASA, position_encoding
from trasa.prepare_data import convert_graph
from trasa.validate_data import validate_dataset
from trasa.vendor.ldtr.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch

from test_data_runtime import manifest


def graph(nodes, edges):
    return Graph(np.asarray(nodes, float).reshape(-1, 2), np.asarray(edges).reshape(-1, 2),
                 np.full(len(nodes), -1), np.ones(len(nodes)), np.ones(len(edges)))


def test_positional_encoding_matches_actual_pinned_ldtr(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('reference_ldtr')
    package.__path__ = []
    utilities = types.ModuleType('reference_ldtr.utils')
    utilities.NestedTensor = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, 'reference_ldtr', package)
    monkeypatch.setitem(sys.modules, 'reference_ldtr.utils', utilities)
    spec = importlib.util.spec_from_file_location('reference_ldtr.position', root / 'third_party/ldtr/models/position_encoding_2D.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    x = torch.zeros(2, 64, 5, 7)
    nested = types.SimpleNamespace(tensors=x, mask=torch.zeros(2, 5, 7, dtype=torch.bool))
    expected = original.PositionEmbeddingSine(32, normalize=True)(nested)
    torch.testing.assert_close(position_encoding(x), expected, rtol=0, atol=0)
    assert not torch.equal(position_encoding(x, 'legacy_corner'), expected)


def test_attention_bilinear_values_and_gradients_by_hand():
    value = torch.tensor([1., 2., 3., 4.], dtype=torch.double).reshape(1, 4, 1, 1).requires_grad_()
    locations = torch.tensor([.4, .4, .6, .6], dtype=torch.double).reshape(1, 1, 1, 1, 2, 2).requires_grad_()
    weights = torch.tensor([.25, .75], dtype=torch.double).reshape(1, 1, 1, 1, 2).requires_grad_()
    output = ms_deform_attn_core_pytorch(value, [(2, 2)], locations, weights)
    assert output.item() == pytest.approx(2.8)
    output.sum().backward()
    torch.testing.assert_close(value.grad.flatten(), torch.tensor([.19, .21, .21, .39], dtype=torch.double))
    torch.testing.assert_close(weights.grad.flatten(), torch.tensor([1.9, 3.1], dtype=torch.double))
    assert torch.autograd.gradcheck(lambda v, l, w: ms_deform_attn_core_pytorch(v, [(2, 2)], l, w),
                                    (value.detach().requires_grad_(), locations.detach().requires_grad_(), weights.detach().requires_grad_()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware/runtime not available')
def test_attention_cuda_matches_cpu_reference():
    value = torch.randn(1, 4, 2, 4)
    locations = torch.rand(1, 3, 2, 1, 2, 2)
    weights = torch.rand(1, 3, 2, 1, 2)
    cpu = ms_deform_attn_core_pytorch(value, [(2, 2)], locations, weights)
    gpu = ms_deform_attn_core_pytorch(value.cuda(), [(2, 2)], locations.cuda(), weights.cuda())
    torch.testing.assert_close(cpu, gpu.cpu())


def test_ablation_common_parameters_have_identical_initialization():
    cfg = Config(pretrained_backbone=False, hidden_dim=64, encoder_layers=1, decoder_layers=1, num_queries=8)
    torch.manual_seed(17)
    baseline = TRASA(replace(cfg, guidance=False, ntc=False)).state_dict()
    torch.manual_seed(17)
    complete = TRASA(cfg).state_dict()
    for key, value in baseline.items():
        actual = complete[key][:, :3] if key == 'backbone.conv1.weight' else complete[key]
        torch.testing.assert_close(value, actual, rtol=0, atol=0)


class ZeroPairs:
    def pair_logits(self, outputs, batch, pairs, connectivity=False, symmetrize=True):
        return torch.zeros(len(pairs), 2, requires_grad=True)


def test_loss_reductions_and_total_objective_numerically():
    from trasa.losses import GraphCriterion
    outputs = {'pred_logits': torch.zeros(1, 3, 2, requires_grad=True),
               'pred_nodes': torch.tensor([[[.1, .1], [.8, .5], [.9, .9]]], requires_grad=True),
               'pred_types': torch.zeros(1, 3, 4, requires_grad=True),
               'node_embeddings': torch.zeros(1, 3, 8, requires_grad=True)}
    target = {'nodes': torch.tensor([[.1, .1], [.7, .5]]), 'edges': [[0, 1]]}
    loss = GraphCriterion(Config())(ZeroPairs(), outputs, [target])
    assert loss['node'].item() == pytest.approx(math.log(2) + .05)
    assert loss['edge'].item() == pytest.approx(math.log(2))
    assert loss['connectivity'].item() == pytest.approx(math.log(2))
    assert loss['type'].item() == pytest.approx(math.log(4))
    assert loss['total'].item() == pytest.approx(4 * math.log(2) + .05)
    loss['total'].backward()
    assert outputs['pred_nodes'].grad[0, 1, 0].item() == pytest.approx(.5)
    assert outputs['pred_types'].grad[0, 0, 1].item() == pytest.approx(-.125)


def test_explicit_internal_node_weights_and_sum_ntc_reduction():
    from trasa.losses import GraphCriterion
    output = {'pred_logits': torch.zeros(1, 2, 2), 'pred_nodes': torch.tensor([[[.1, .1], [.8, .5]]]),
              'pred_types': torch.zeros(1, 2, 4), 'node_embeddings': torch.zeros(1, 2, 8)}
    target = {'nodes': torch.tensor([[.1, .1], [.7, .5]]), 'edges': [[0, 1]]}
    loss = GraphCriterion(Config(validity_loss_weight=3, coordinate_loss_weight=5, type_reduction='sum'))(
        ZeroPairs(), output, [target])
    assert loss['node'].item() == pytest.approx(3 * math.log(2) + 5 * .05)
    assert loss['type'].item() == pytest.approx(2 * math.log(4))


def test_one_node_reference_is_rejected_with_defined_type_error():
    from trasa.losses import GraphCriterion
    output = {'pred_logits': torch.zeros(1, 2, 2), 'pred_nodes': torch.zeros(1, 2, 2),
              'node_embeddings': torch.zeros(1, 2, 8)}
    with pytest.raises(ValueError, match='Isolated reference nodes'):
        GraphCriterion(Config())(ZeroPairs(), output, [{'nodes': torch.zeros(1, 2), 'edges': []}])


@pytest.mark.parametrize('change', [{'hidden_dim': 0}, {'epochs': 1.5}, {'learning_rate': float('nan')},
                                    {'edge_class_weights': [1, float('inf')]}, {'merge_tolerance': -1}])
def test_invalid_experiment_configuration_fails_before_training(change):
    with pytest.raises(ValueError):
        Config(**change)


@pytest.mark.parametrize('offsets,connectivity', [('3 2', '0 1 2'), ('2', '0 1.5'), ('1', '0')])
def test_malformed_ascii_vtp_rejected(tmp_path, offsets, connectivity):
    from trasa.prepare_data import read_vtp
    source = tmp_path / 'bad.vtp'
    source.write_text('<VTKFile><PolyData><Piece NumberOfPoints="3"><Points>'
                      '<DataArray NumberOfComponents="3" format="ascii">0 0 0 1 0 0 2 0 0</DataArray></Points>'
                      '<Lines><DataArray Name="connectivity" format="ascii">' + connectivity + '</DataArray>'
                      '<DataArray Name="offsets" format="ascii">' + offsets + '</DataArray></Lines>'
                      '</Piece></PolyData></VTKFile>')
    with pytest.raises(ValueError):
        read_vtp(source)


def test_road_type_widths_and_correction_metadata_survive_conversion(tmp_path):
    from trasa.train_geometry import geometry_targets
    source = tmp_path / 'corrected.json'
    source.write_text(json.dumps({'nodes': [[2, 8], [28, 8], [28, 24]],
                                 'edges': [[1, 0], [1, 2], [0, 1]], 'edge_widths': [3, 7, 3],
                                 'metadata': {'historical_correction': 'reviewed'}}))
    converted = convert_graph(source, 'pixels', 32)
    assert converted['edges'] == [[0, 1], [1, 2]] and converted['edge_widths'] == [3, 7]
    assert converted['metadata']['historical_correction'] == 'reviewed'
    target = {'nodes': torch.tensor(converted['nodes']) / 31, 'edges': torch.tensor(converted['edges']),
              'types': torch.tensor(converted['types']), 'edge_widths': torch.tensor(converted['edge_widths'])}
    labels = geometry_targets(target, 32)
    assert labels[1, :, 10].sum() == 3 and labels[1, 20].sum() == 7
    assert labels[0].sum() == 0  # intersection-only, degree-two bend and endpoints excluded


def test_random_order_ce_is_not_ce_of_averaged_logits():
    from trasa.losses import GraphCriterion
    class Asymmetric:
        def pair_logits(self, outputs, batch, pairs, connectivity=False, symmetrize=True):
            if symmetrize or connectivity:
                return torch.zeros(len(pairs), 2)
            positive = pairs[:, 0] < pairs[:, 1]
            return torch.stack((~positive, positive), -1).float() * 10
    out = {'pred_logits': torch.zeros(1, 2, 2), 'pred_nodes': torch.tensor([[[.1, .1], [.9, .9]]]),
           'pred_types': torch.zeros(1, 2, 4), 'node_embeddings': torch.zeros(1, 2, 8)}
    target = {'nodes': out['pred_nodes'][0], 'edges': [[0, 1]]}
    torch.manual_seed(0)
    actual = GraphCriterion(Config())(Asymmetric(), out, [target])
    legacy = GraphCriterion(Config(pair_training='legacy_average_logits'))(Asymmetric(), out, [target])
    assert actual['edge'].item() == pytest.approx(10., abs=.001)
    assert legacy['edge'].item() == pytest.approx(math.log(2))


@pytest.mark.parametrize('edges', [[[0, .5]], [[0, float('nan')]], [[False, True]], [0, 1], [[0, 3]], [[1, 1]], [[]], [[], []]])
def test_bad_edge_indices_are_not_silently_cast(edges):
    with pytest.raises((ValueError, TypeError)):
        canonical_edges(edges, 3)


@pytest.mark.parametrize('value', [
    {'nodes': [0, 1, 2, 3], 'edges': [[0, 1]]},
    {'nodes': [[0, 0], [1, 1]], 'edges': [[0, 1.5]]},
    {'nodes': [[0, 0]], 'edges': []},
    {'nodes': [[0, 0], [0, 0]], 'edges': [[0, 1]]},
    {'nodes': [[0, 0], [256, 0]], 'edges': [[0, 1]]},
    {'nodes': [[], []], 'edges': []},
])
def test_reference_annotations_reject_malformed_graphs(tmp_path, value):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        convert_graph(path, 'pixels')


def test_dataset_validation_collects_all_bad_samples(tmp_path):
    path, document = manifest(tmp_path)
    document['samples'][0]['image'] = 'missing_a.png'
    document['samples'][1]['image'] = 'missing_b.png'
    path.write_text(json.dumps(document))
    report = validate_dataset(path, guided=False)
    assert not report['valid'] and len(report['invalid_samples']) == 2
    assert {row['sample_id'] for row in report['invalid_samples']} == {'a', 'b'}


@pytest.mark.parametrize('bad_provenance', [None, [], 'not_an_object'])
def test_malformed_correction_provenance_is_reported_not_crashed(tmp_path, bad_provenance):
    path, document = manifest(tmp_path)
    document['reference_provenance'] = bad_provenance
    path.write_text(json.dumps(document))
    report = validate_dataset(path, guided=False, paper=True)
    assert not report['valid'] and 'reference_provenance must be an object' in report['errors']


def test_malformed_empty_nodes_fail_all_public_graph_readers(tmp_path):
    from trasa.export import read_graph
    from trasa.evaluate import load_graph
    path, _ = manifest(tmp_path)
    annotation = tmp_path / 'graph.json'
    annotation.write_text(json.dumps({'nodes': [[]], 'edges': []}))
    assert not validate_dataset(path, guided=False)['valid']
    for reader in (read_graph, load_graph):
        with pytest.raises(ValueError, match='nodes'):
            reader(annotation)


@pytest.mark.parametrize('declared,points', [('0', '0 0 0'), ('1', '0 0 0 trailing_invalid_text')])
def test_ascii_vtp_does_not_discard_invalid_point_data(tmp_path, declared, points):
    from trasa.prepare_data import read_vtp
    source = tmp_path / 'points.vtp'
    source.write_text(f'<VTKFile><PolyData><Piece NumberOfPoints="{declared}"><Points>'
                      f'<DataArray NumberOfComponents="3" format="ascii">{points}</DataArray></Points>'
                      '</Piece></PolyData></VTKFile>')
    with pytest.raises(ValueError):
        read_vtp(source)


@pytest.mark.parametrize('declared_points', [0, 2])
def test_vtp_inconsistent_line_declarations_are_not_empty_reference_graphs(tmp_path, declared_points):
    from trasa.prepare_data import read_vtp
    source = tmp_path / 'inconsistent_lines.vtp'
    points = '' if not declared_points else '0 0 0 1 0 0'
    source.write_text(f'<VTKFile><PolyData><Piece NumberOfPoints="{declared_points}" NumberOfLines="0">'
                      f'<Points><DataArray NumberOfComponents="3" format="ascii">{points}</DataArray></Points>'
                      '<Lines><DataArray Name="connectivity" format="ascii">0 1</DataArray>'
                      '<DataArray Name="offsets" format="ascii">2</DataArray></Lines>'
                      '</Piece></PolyData></VTKFile>')
    with pytest.raises(ValueError, match='polylines|line count'):
        read_vtp(source)


def test_cache_binding_rejects_wrong_sample_and_image(tmp_path):
    path, _ = manifest(tmp_path)
    with np.load(tmp_path / 'masks.npz') as original:
        changed = {name: original[name] for name in original.files}
    changed['map_id'] = np.array('wrong_map')
    np.savez(tmp_path / 'masks.npz', **changed)
    with pytest.raises(ValueError, match='binding mismatch: map_id'):
        PatchDataset(path, 'train', Config())[0]


def test_geometry_selection_on_test_map_is_leakage(tmp_path):
    path, document = manifest(tmp_path)
    document['samples'][1]['split'] = 'test'
    document['guidance_provenance']['validation_map_ids'] = ['B']
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='selection.*test'):
        PatchDataset(path, 'train', Config())


def test_geometry_training_and_validation_declarations_cannot_overlap(tmp_path):
    path, document = manifest(tmp_path)
    document['guidance_provenance']['validation_map_ids'] = ['A']
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='must be disjoint'):
        PatchDataset(path, 'train', Config())
    from trasa.geometry import SAMRoadGeometry
    # The validation method needs metadata only, not an allocated ViT encoder.
    checker = types.SimpleNamespace(provenance=None)
    with pytest.raises(ValueError, match='must be disjoint'):
        SAMRoadGeometry.validate_provenance(checker, document['guidance_provenance'])


def test_clip_and_overlap_reconstruction_preserve_only_existing_edges():
    first = graph([[0, 10], [255, 10]], [[0, 1]])
    second = graph([[0, 10], [255, 10]], [[0, 1]])
    origins = [(0, 0), (128, 0)]
    boxes = ownership_boxes(origins, 256, (32, 300))
    clipped = [clip_graph(g, b) for g, b in zip((first, second), boxes)]
    result = merge_patch_graphs(clipped, origins, boundary_boxes=boxes)
    assert len(result.nodes) == 3 and len(result.edges) == 2
    assert result.nodes[:, 0].min() == 0 and result.nodes[:, 0].max() == 299
    assert sum(np.linalg.norm(result.nodes[a] - result.nodes[b]) for a, b in result.edges) == pytest.approx(299)


def test_close_parallel_roads_do_not_collapse_transitively():
    first = graph([[100, 10], [255, 10], [100, 14], [255, 14]], [[0, 1], [2, 3]])
    second = graph([[0, 10], [100, 10], [0, 14], [100, 14]], [[0, 1], [2, 3]])
    merged = merge_patch_graphs([first, second], [(0, 0), (256, 0)])
    assert len(merged.nodes) == 6 and len(merged.edges) == 4
    from trasa.metrics import sparse_graph
    from scipy.sparse.csgraph import connected_components
    assert connected_components(sparse_graph(merged), directed=False)[0] == 2


def test_graph_duplicate_edges_do_not_change_degree_or_scores():
    duplicate = Graph([[0, 0], [20, 0]], [[0, 1], [1, 0]], [1, 1], [1, 1], [.4, .8])
    assert duplicate.edges.tolist() == [[0, 1]] and duplicate.edge_scores.tolist() == [.8]
    assert evaluate(duplicate, duplicate)['connectivity'] == 1


def test_attached_spurious_branch_is_counted():
    reference = graph([[0, 0], [20, 0]], [[0, 1]])
    prediction = graph([[0, 0], [10, 0], [20, 0], [10, 30]], [[0, 1], [1, 2], [1, 3]])
    result = evaluate(reference, prediction)
    assert result['spurious_branches'] == 1
    assert result['completeness'] == 1 and result['correctness'] < 1


def test_duplicate_intersection_nodes_do_not_both_match_one_reference():
    reference = graph([[20, 20], [0, 20], [40, 20], [20, 0]], [[0, 1], [0, 2], [0, 3]])
    duplicate = graph([[20, 20], [0, 20], [40, 20], [20, 0], [20, 20]],
                      [[0, 1], [0, 2], [0, 3], [4, 1], [4, 2], [4, 3]])
    scores = evaluate(reference, duplicate)
    assert scores['intersection_precision'] == .5 and scores['intersection_recall'] == 1
    assert scores['intersection_f1'] == pytest.approx(2 / 3)
    assert scores['intersection_degree_accuracy'] == 1


def test_both_empty_graphs_have_explicit_undefined_topology_conventions():
    empty = graph([], [])
    scores = evaluate(empty, empty)
    assert scores['completeness'] == scores['correctness'] == scores['quality'] == scores['apls'] == 1
    assert scores['connectivity'] is None and scores['intersection_degree_accuracy'] is None
    assert scores['split_errors'] == scores['spurious_branches'] == 0


@pytest.mark.parametrize('kind', ['perfect', 'empty', 'missing_intersection', 'spurious_intersection', 'shifted_parallel', 'disconnected'])
def test_topology_and_geometry_cases(kind):
    reference = graph([[20, 20], [0, 20], [40, 20], [20, 0]], [[0, 1], [0, 2], [0, 3]])
    if kind == 'perfect':
        prediction = reference
    elif kind == 'empty':
        prediction = graph([], [])
    elif kind == 'missing_intersection':
        prediction = graph([[0, 20], [40, 20]], [[0, 1]])
    elif kind == 'spurious_intersection':
        prediction = graph([[20, 20], [0, 20], [40, 20], [20, 0], [20, 40]], [[0, 1], [0, 2], [0, 3], [0, 4]])
    elif kind == 'shifted_parallel':
        prediction = graph(reference.nodes + [20, 40], reference.edges)
    else:
        prediction = graph(reference.nodes, [[0, 1], [0, 2]])
    scores = evaluate(reference, prediction)
    assert all(value is None or np.isfinite(value) for value in scores.values())
    if kind == 'perfect':
        assert scores['intersection_f1'] == scores['intersection_degree_accuracy'] == scores['apls'] == 1
    if kind in ('empty', 'missing_intersection', 'shifted_parallel'):
        assert scores['intersection_recall'] == 0
    if kind == 'spurious_intersection':
        assert scores['intersection_f1'] == 1 and scores['intersection_degree_accuracy'] == 0
    if kind == 'empty':
        assert scores['apls'] == scores['quality'] == 0 and scores['connectivity'] is None


def test_apls_projected_path_lengths_and_zero_tolerance():
    reference = graph([[0, 0], [10, 0]], [[0, 1]])
    predicted = graph([[0, 0], [10, 10]], [[0, 1]])
    assert apls_control_points(reference, predicted, tolerance=100, spacing=100, max_controls=2) == pytest.approx(1 / math.sqrt(2))
    assert evaluate(reference, reference, tolerance=0)['quality'] == 1
    assert evaluation_protocol()['apls_algorithm'] == 'control_point_apls_v1'


def test_four_class_report_and_invalid_labels():
    result = type_metrics([0, 1, 2, 3], [0, 1, 1, 3])
    assert result['type_f1_endpoint'] == pytest.approx(2 / 3)
    assert result['type_recall_intermediate'] == 0
    assert result['type_macro_f1'] == pytest.approx((1 + 2 / 3 + 0 + 1) / 4)
    with pytest.raises(ValueError):
        type_metrics([4], [0])
