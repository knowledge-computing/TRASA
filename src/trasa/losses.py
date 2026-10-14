"""Hungarian node matching + LDTR direct edge/N-hop objectives + separate NTC."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .graph import canonical_edges, connectivity_labels, node_types


class GraphCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def match(self, outputs, targets):
        assignments = []
        for logits, predictions, target in zip(outputs['pred_logits'], outputs['pred_nodes'], targets):
            nodes = target['nodes'].to(predictions)
            if len(nodes) > len(predictions):
                raise ValueError('Reference graph exceeds node-query capacity; do not silently truncate')
            if nodes.ndim != 2 or nodes.shape[1] != 2 or not torch.isfinite(nodes).all() or (nodes < 0).any() or (nodes > 1).any():
                raise ValueError('Target nodes must be Nx2 finite normalized (x,y) coordinates')
            cost = (self.config.matcher_node_cost * torch.cdist(predictions, nodes, p=1)
                    - self.config.matcher_class_cost * logits.softmax(-1)[:, 1, None])
            query, reference = linear_sum_assignment(cost.cpu().numpy())
            assignments.append((torch.as_tensor(query, device=predictions.device),
                                torch.as_tensor(reference, device=predictions.device)))
        return assignments

    def forward(self, model, outputs, targets):
        if len(targets) != len(outputs['pred_logits']) or not targets:
            raise ValueError('One target graph is required per input image')
        indices = self.match(outputs, targets)
        valid = torch.zeros(outputs['pred_logits'].shape[:2], dtype=torch.long,
                            device=outputs['pred_logits'].device)
        types = torch.zeros_like(valid)
        coordinate_loss = outputs['pred_nodes'].sum() * 0
        edge_logits, edge_labels, conn_logits, conn_labels = [], [], [], []
        node_count = 0
        for batch, (target, (query, reference)) in enumerate(zip(targets, indices)):
            nodes = target['nodes'].to(outputs['pred_nodes'])
            edges = canonical_edges(target['edges'], len(nodes)).cpu()
            roles = node_types(len(nodes), edges).to(types.device)
            if 'types' in target:
                supplied = torch.as_tensor(target['types'], device=roles.device)
                if supplied.shape != roles.shape or not torch.isfinite(supplied).all() or (supplied != supplied.long()).any() or not torch.equal(supplied.long(), roles):
                    raise ValueError('Reference node types must be integers matching the patch graph degrees')
            valid[batch, query] = 1
            types[batch, query] = roles[reference]
            coordinate_loss = coordinate_loss + F.l1_loss(outputs['pred_nodes'][batch, query], nodes[reference], reduction='sum')
            node_count += len(nodes)
            if len(nodes) >= 2:
                local = torch.triu_indices(len(nodes), len(nodes), offset=1).T
                query_pairs = query[local.to(query.device)]
                reference_pairs = reference[local.to(reference.device)].cpu()
                adjacency = torch.zeros((len(nodes), len(nodes)), dtype=torch.long)
                adjacency[edges[:, 0], edges[:, 1]] = 1
                adjacency[edges[:, 1], edges[:, 0]] = 1
                connectivity = connectivity_labels(len(nodes), edges, self.config.hops)
                edge_labels.append(adjacency[reference_pairs[:, 0], reference_pairs[:, 1]])
                conn_labels.append(connectivity[reference_pairs[:, 0], reference_pairs[:, 1]].long())
            elif len(nodes) == 0:
                # Empty images: supervise all query pairs as absent (no random fabricated graph).
                query_pairs = torch.triu_indices(valid.shape[1], valid.shape[1], offset=1, device=valid.device).T
                edge_labels.append(torch.zeros(len(query_pairs), dtype=torch.long))
                conn_labels.append(torch.zeros(len(query_pairs), dtype=torch.long))
            else:
                continue
            if self.config.pair_training == 'random_order':
                def ordered_pairs(pairs):
                    swap = torch.rand(len(pairs), device=pairs.device) < 0.5
                    return torch.where(swap[:, None], pairs.flip(-1), pairs)
                edge_logits.append(model.pair_logits(outputs, batch, ordered_pairs(query_pairs), symmetrize=False))
                conn_logits.append(model.pair_logits(outputs, batch, ordered_pairs(query_pairs), connectivity=True, symmetrize=False))
            else:
                edge_logits.append(model.pair_logits(outputs, batch, query_pairs))
                conn_logits.append(model.pair_logits(outputs, batch, query_pairs, connectivity=True))
        classification = F.cross_entropy(outputs['pred_logits'].transpose(1, 2), valid,
                                         weight=outputs['pred_logits'].new_tensor(self.config.node_class_weights))
        losses = {'node': self.config.validity_loss_weight * classification +
                  self.config.coordinate_loss_weight * coordinate_loss / max(node_count, 1)}
        zero = outputs['node_embeddings'].sum() * 0
        if edge_logits and sum(len(x) for x in edge_logits):
            pred_edges, pred_conn = torch.cat(edge_logits), torch.cat(conn_logits)
            losses['edge'] = F.cross_entropy(pred_edges, torch.cat(edge_labels).to(pred_edges.device),
                                             weight=pred_edges.new_tensor(self.config.edge_class_weights))
            losses['connectivity'] = F.cross_entropy(pred_conn, torch.cat(conn_labels).to(pred_conn.device))
        else:
            losses['edge'] = zero + sum(p.sum() * 0 for p in model.edge_head.parameters())
            losses['connectivity'] = zero + sum(p.sum() * 0 for p in model.connectivity_head.parameters())
        if self.config.ntc:
            losses['type'] = F.cross_entropy(outputs['pred_types'].transpose(1, 2), types, reduction=self.config.type_reduction)
        else:
            losses['type'] = zero
        losses['total'] = (self.config.node_weight * losses['node'] + self.config.edge_weight * losses['edge']
                           + self.config.connectivity_weight * losses['connectivity']
                           + self.config.type_weight * losses['type'])
        return losses
