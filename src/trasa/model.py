"""LDTR deformable graph decoder with TRASA dual-mask guidance and NTC."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .config import Config
from .vendor.ldtr.deformable_detr_2D import DeformableTransformer


class MLP(nn.Sequential):
    def __init__(self, inputs, hidden, outputs):
        super().__init__(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def guidance_stem(conv):
    """Preserve RGB filters; initialize both guidance filters with their mean."""
    devices = [conv.weight.device.index or 0] if conv.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        stem = nn.Conv2d(5, conv.out_channels, conv.kernel_size, conv.stride,
                         conv.padding, bias=False, device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        stem.weight[:, :3].copy_(conv.weight)
        stem.weight[:, 3:].copy_(conv.weight.mean(1, keepdim=True).expand(-1, 2, -1, -1))
    return stem


def position_encoding(x, convention='ldtr_centered'):
    """Normalized 2D sine encoding, as in LDTR/Deformable DETR."""
    b, dim, h, w = x.shape
    offset, eps = (0.5, 1e-6) if convention == 'ldtr_centered' else (0., 0.)
    y = (torch.arange(1, h + 1, device=x.device, dtype=torch.float32) - offset) / (h + eps) * 2 * math.pi
    z = (torch.arange(1, w + 1, device=x.device, dtype=torch.float32) - offset) / (w + eps) * 2 * math.pi
    frequency = 10000 ** (2 * torch.div(torch.arange(dim // 2, device=x.device), 2,
                                      rounding_mode='floor') / (dim // 2))
    py, px = y[:, None] / frequency, z[:, None] / frequency
    py = torch.stack((py[:, 0::2].sin(), py[:, 1::2].cos()), -1).flatten(1)
    px = torch.stack((px[:, 0::2].sin(), px[:, 1::2].cos()), -1).flatten(1)
    pos = torch.cat((py[:, None].expand(h, w, -1), px[None].expand(h, w, -1)), -1)
    return pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)


class TRASA(nn.Module):
    def __init__(self, config=Config(), geometry=None):
        super().__init__()
        self.config = config
        self.geometry = geometry
        if geometry is not None:
            geometry.requires_grad_(False).eval()
        weights = ResNet50_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
        backbone = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        if config.guidance:
            backbone.conv1 = guidance_stem(backbone.conv1)
        self.backbone = IntermediateLayerGetter(backbone, {'layer2': '0', 'layer3': '1', 'layer4': '2'})
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, config.hidden_dim, 1), nn.GroupNorm(32, config.hidden_dim))
            for c in (512, 1024, 2048)])
        self.input_proj.append(nn.Sequential(nn.Conv2d(2048, config.hidden_dim, 3, 2, 1),
                                            nn.GroupNorm(32, config.hidden_dim)))
        self.transformer = DeformableTransformer(
            d_model=config.hidden_dim, nhead=config.attention_heads,
            num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.feedforward_dim, dropout=config.dropout,
            return_intermediate_dec=False, num_feature_levels=config.feature_levels,
            dec_n_points=config.sampling_points, enc_n_points=config.sampling_points,
            two_stage=False, rln_attn=config.rln_attention)
        # One LDTR relation token in addition to 100 node queries.
        self.query_embed = nn.Embedding(config.num_queries + 1, config.hidden_dim * 2)
        self.valid_head = nn.Linear(config.hidden_dim, 2)
        self.coordinate_head = MLP(config.hidden_dim, config.hidden_dim, 2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() + 104729)
            self.type_head = MLP(config.hidden_dim, config.hidden_dim, 4) if config.ntc else None
        self.edge_head = MLP(config.hidden_dim * 3, config.hidden_dim, 2)
        self.connectivity_head = MLP(config.hidden_dim * 2, config.hidden_dim, 2)
        self.register_buffer('rgb_mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        if self.geometry is not None:
            self.geometry.eval()
        return self

    def forward(self, rgb, guidance=None):
        """RGB: B,3,H,W float [0,1]. Guidance: B,2,H,W [road, intersection]."""
        if rgb.ndim != 4 or rgb.shape[1:] != (3, self.config.patch_size, self.config.patch_size):
            raise ValueError('Expected fixed-size B,3,H,W RGB patches')
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('RGB must be finite and in [0,1]')
        x = (rgb - self.rgb_mean) / self.rgb_std
        if self.config.guidance:
            if guidance is None:
                if self.geometry is None:
                    raise ValueError('Frozen SAM-Road geometry or cached probability guidance is required')
                with torch.no_grad():
                    guidance = self.geometry(rgb)
            if guidance.ndim != 4 or guidance.shape[:2] != (len(rgb), 2):
                raise ValueError('Guidance must be B,2,H,W in [road, intersection] order')
            if not torch.isfinite(guidance).all() or guidance.min() < 0 or guidance.max() > 1:
                raise ValueError('Guidance must contain finite probabilities in [0,1]')
            guidance = F.interpolate(guidance.detach(), rgb.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, guidance), 1)
        elif guidance is not None:
            raise ValueError('Guidance supplied to an RGB-only ablation')
        features = list(self.backbone(x).values())
        sources = [proj(feat) for proj, feat in zip(self.input_proj[:3], features)]
        sources.append(self.input_proj[3](features[-1]))
        masks = [torch.zeros((len(rgb), *s.shape[-2:]), dtype=torch.bool, device=s.device) for s in sources]
        positions = [position_encoding(s, self.config.position_encoding).to(s.dtype) for s in sources]
        h = self.transformer(sources, masks, self.query_embed.weight, positions)[0]
        nodes = h[:, :self.config.num_queries]
        out = {'pred_logits': self.valid_head(nodes), 'pred_nodes': self.coordinate_head(nodes).sigmoid(),
               'node_embeddings': nodes, 'relation_embedding': h[:, -1]}
        if self.type_head is not None:
            out['pred_types'] = self.type_head(nodes)
        return out

    def pair_logits(self, outputs, batch, pairs, connectivity=False, symmetrize=True):
        """Average both endpoint orders, following LDTR's undirected inference."""
        h = outputs['node_embeddings'][batch]
        a, b = h[pairs[:, 0]], h[pairs[:, 1]]
        if connectivity:
            ordered = self.connectivity_head(torch.cat((a, b), -1))
            return (ordered + self.connectivity_head(torch.cat((b, a), -1))) / 2 if symmetrize else ordered
        relation = outputs['relation_embedding'][batch].expand(len(pairs), -1)
        ordered = self.edge_head(torch.cat((a, b, relation), -1))
        return (ordered + self.edge_head(torch.cat((b, a, relation), -1))) / 2 if symmetrize else ordered
