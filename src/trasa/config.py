from dataclasses import asdict, dataclass
import math
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    patch_size: int = 256
    num_queries: int = 100
    hidden_dim: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 3
    attention_heads: int = 8
    feature_levels: int = 4
    feedforward_dim: int = 1024
    dropout: float = 0.1
    sampling_points: int = 4
    hops: int = 5
    guidance: bool = True
    ntc: bool = True
    pretrained_backbone: bool = True
    node_weight: float = 1.0
    edge_weight: float = 1.0
    connectivity_weight: float = 1.0
    type_weight: float = 0.5
    matcher_class_cost: float = 3.0
    matcher_node_cost: float = 5.0
    node_class_weights: tuple = (0.2, 0.8)
    edge_class_weights: tuple = (1.0, 20.0)
    node_threshold: float = 0.5
    edge_threshold: float = 0.5
    merge_tolerance: float = 5.0
    boundary_band: float = 5.0
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 10
    position_encoding: str = 'ldtr_centered'
    pair_training: str = 'random_order'
    require_bound_guidance: bool = True
    type_reduction: str = 'mean'
    rln_attention: bool = True
    validity_loss_weight: float = 1.0
    coordinate_loss_weight: float = 1.0

    def __post_init__(self):
        integer_fields = ('patch_size', 'num_queries', 'hidden_dim', 'encoder_layers', 'decoder_layers',
                          'attention_heads', 'feature_levels', 'feedforward_dim', 'sampling_points',
                          'hops', 'batch_size', 'epochs', 'seed')
        if any(isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int)
               for name in integer_fields):
            raise ValueError('Architecture counts, epoch budget and seed must be integers')
        for name, value in self.asdict().items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.hidden_dim < 32 or self.feedforward_dim < 1 or self.seed < 0:
            raise ValueError('Positive hidden/feedforward dimensions and nonnegative seed required')
        if min(self.matcher_class_cost, self.matcher_node_cost, self.merge_tolerance, self.boundary_band) < 0:
            raise ValueError('Matching costs and spatial tolerances must be nonnegative')
        if self.matcher_class_cost + self.matcher_node_cost <= 0:
            raise ValueError('At least one Hungarian matching cost must be positive')
        if self.attention_heads < 1 or self.patch_size < 32 or self.patch_size % 16:
            raise ValueError('Positive attention heads and patch size >=32 divisible by 16 required')
        if self.hidden_dim % 32 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by 32 and attention_heads")
        if self.feature_levels != 4 or self.hops < 1 or self.num_queries < 1:
            raise ValueError("Use four LDTR feature levels, positive hops and queries")
        if min(self.encoder_layers, self.decoder_layers, self.sampling_points, self.batch_size, self.epochs) < 1:
            raise ValueError('Layer counts, sampling points, batch size and epochs must be positive')
        if not 0 <= self.node_threshold <= 1 or not 0 <= self.edge_threshold <= 1:
            raise ValueError('Confidence thresholds must be in [0,1]')
        if self.position_encoding not in ('ldtr_centered', 'legacy_corner'):
            raise ValueError('Unknown positional encoding convention')
        if self.pair_training not in ('random_order', 'legacy_average_logits'):
            raise ValueError('Unknown pair-training objective')
        if self.type_reduction not in ('mean', 'sum'):
            raise ValueError('NTC reduction must be mean or sum')
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid dropout, learning rate or weight decay')
        if min(self.node_weight, self.edge_weight, self.connectivity_weight, self.type_weight,
               self.validity_loss_weight, self.coordinate_loss_weight) < 0:
            raise ValueError('Loss weights must be nonnegative')
        if len(self.node_class_weights) != 2 or len(self.edge_class_weights) != 2 or any(
                not math.isfinite(w) or w <= 0 for w in (*self.node_class_weights, *self.edge_class_weights)):
            raise ValueError('Two positive class weights required')

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            return cls(**yaml.safe_load(stream))

    def asdict(self):
        return asdict(self)
