import torch

from trasa.config import Config
from trasa.losses import GraphCriterion


class PredictedPairs:
    def pair_logits(self, outputs, batch, pairs, connectivity=False):
        # Query 2 is the central intersection; all star nodes are within 2 hops.
        positive = torch.ones(len(pairs), dtype=torch.bool) if connectivity else (pairs == 2).any(1)
        return torch.stack((~positive, positive), -1).float() * 30


def test_hungarian_permutation_applies_to_types_and_both_pair_targets():
    target = {'nodes': torch.tensor([[.1, .1], [.2, .2], [.3, .3], [.4, .4]]),
              'edges': torch.tensor([[0, 1], [0, 2], [0, 3]])}
    # Query/reference mapping: 0->2, 1->3, 2->0, 3->1; fifth query is unmatched.
    outputs = {'pred_nodes': torch.tensor([[[.3, .3], [.4, .4], [.1, .1], [.2, .2], [.9, .9]]]),
               'pred_logits': torch.tensor([[[0., 30.]] * 4 + [[30., 0.]]]),
               'pred_types': torch.tensor([[[0., 30., 0., 0.], [0., 30., 0., 0.],
                                           [0., 0., 0., 30.], [0., 30., 0., 0.], [30., 0., 0., 0.]]]),
               'node_embeddings': torch.zeros(1, 5, 16)}
    losses = GraphCriterion(Config())(PredictedPairs(), outputs, [target])
    for key in ('node', 'edge', 'connectivity', 'type', 'total'):
        assert losses[key] < 1e-6
