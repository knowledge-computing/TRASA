from dataclasses import replace

import pytest
import torch
from torch import nn

from trasa.config import Config
from trasa.graph import decode_graphs
from trasa.losses import GraphCriterion
from trasa.model import TRASA, guidance_stem


def config(**kwargs):
    return replace(Config(pretrained_backbone=False), **kwargs)


def test_five_channel_stem_preserves_pretrained_filters():
    conv = nn.Conv2d(3, 8, 7, 2, 3, bias=False)
    stem = guidance_stem(conv)
    torch.testing.assert_close(stem.weight[:, :3], conv.weight)
    for channel in (3, 4):
        torch.testing.assert_close(stem.weight[:, channel], conv.weight.mean(1))


def test_full_paper_forward_backward_and_frozen_masks():
    model = TRASA(config())
    guidance = torch.rand(1, 2, 128, 128, requires_grad=True)
    out = model(torch.rand(1, 3, 256, 256), guidance)
    assert out['pred_logits'].shape == (1, 100, 2)
    assert out['pred_nodes'].shape == (1, 100, 2)
    assert out['pred_types'].shape == (1, 100, 4)
    targets = [{'nodes': torch.tensor([[.1, .2], [.3, .4], [.4, .5], [.6, .7]]),
                'edges': torch.tensor([[0, 1], [1, 2], [1, 3]])}]
    losses = GraphCriterion(model.config)(model, out, targets)
    expected = losses['node'] + losses['edge'] + losses['connectivity'] + .5 * losses['type']
    torch.testing.assert_close(losses['total'], expected)
    losses['total'].backward()
    assert guidance.grad is None
    for module in (model.valid_head, model.coordinate_head, model.type_head, model.edge_head, model.connectivity_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.backbone.conv1.weight.grad[:, 3:].abs().sum() > 0


@pytest.mark.parametrize('guidance,ntc', [(False, False), (False, True), (True, False), (True, True)])
def test_ablation_and_negative_image(guidance, ntc):
    model = TRASA(config(guidance=guidance, ntc=ntc, num_queries=8, hidden_dim=64,
                         encoder_layers=1, decoder_layers=1, feedforward_dim=128))
    out = model(torch.rand(1, 3, 256, 256), torch.rand(1, 2, 256, 256) if guidance else None)
    losses = GraphCriterion(model.config)(model, out, [{'nodes': torch.empty(0, 2), 'edges': torch.empty(0, 2, dtype=torch.long)}])
    assert all(torch.isfinite(v) for v in losses.values())
    losses['total'].backward()
    assert ('pred_types' in out) == ntc


def test_pair_symmetry_and_inference_ignore_auxiliary_heads():
    model = TRASA(config(num_queries=3, hidden_dim=64, encoder_layers=1, decoder_layers=1))
    out = {'node_embeddings': torch.randn(1, 3, 64), 'relation_embedding': torch.randn(1, 64),
           'pred_logits': torch.tensor([[[-10., 10.], [-10., 10.], [10., -10.]]]),
           'pred_nodes': torch.tensor([[[0., 0.], [1., 1.], [.5, .5]]]),
           'pred_types': torch.tensor([[[10., 0., 0., 0.], [0., 0., 0., 10.], [0., 0., 10., 0.]]])}
    pairs = torch.tensor([[0, 1], [1, 0]])
    pred = model.pair_logits(out, 0, pairs)
    torch.testing.assert_close(pred[0], pred[1])
    def forbidden(*args, **kwargs):
        raise AssertionError('Connectivity head must not run at inference')
    model.connectivity_head.forward = forbidden
    graph = decode_graphs(model, out)[0]
    assert len(graph.nodes) == 2  # retained even though NTC predicts no-node for query 0
    assert graph.types.tolist() == [0, 3]
    assert graph.nodes.tolist() == [[0., 0.], [255., 255.]]


def test_matching_overflow_and_missing_guidance_fail():
    model = TRASA(config(num_queries=2))
    with pytest.raises(ValueError, match='guidance'):
        model(torch.rand(1, 3, 256, 256))
    criterion = GraphCriterion(model.config)
    out = {'pred_logits': torch.zeros(1, 2, 2), 'pred_nodes': torch.zeros(1, 2, 2)}
    with pytest.raises(ValueError, match='capacity'):
        criterion.match(out, [{'nodes': torch.zeros(3, 2)}])
