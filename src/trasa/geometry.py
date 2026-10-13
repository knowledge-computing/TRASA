"""Frozen SAM-Road geometry, without Lightning, TopoNet, NMS or GT inputs."""
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.segment_anything.modeling.image_encoder import ImageEncoderViT
from .vendor.segment_anything.modeling.common import LayerNorm2d
from .vendor.segment_anything.modeling.mask_decoder import MaskDecoder
from .vendor.segment_anything.modeling.prompt_encoder import PromptEncoder
from .vendor.segment_anything.modeling.transformer import TwoWayTransformer


class SAMRoadGeometry(nn.Module):
    """Architecture/state keys match pinned SAM-Road's geometry modules.

    Inputs are RGB [0,1]; upstream normalization uses pixels [0,255].
    Upstream mask order [keypoint, road] is returned as [road, intersection].
    A trained geometry checkpoint is mandatory; vanilla SAM has no road decoder.
    """
    def __init__(self, patch_size=256, variant='vit_b', use_sam_decoder=False):
        super().__init__()
        variants = {'vit_b': (768, 12, 12, [2, 5, 8, 11]),
                    'vit_l': (1024, 24, 16, [5, 11, 17, 23]),
                    'vit_h': (1280, 32, 16, [7, 15, 23, 31])}
        if variant not in variants or patch_size % 16:
            raise ValueError('Unknown SAM variant or patch size not divisible by 16')
        dim, depth, heads, global_attention = variants[variant]
        self.patch_size, self.variant, self.use_sam_decoder = patch_size, variant, use_sam_decoder
        self.image_encoder = ImageEncoderViT(
            depth=depth, embed_dim=dim, img_size=patch_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=heads,
            patch_size=16, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=global_attention, window_size=14, out_chans=256)
        if use_sam_decoder:
            self.prompt_encoder = PromptEncoder(256, (patch_size // 16, patch_size // 16),
                                                (patch_size, patch_size), 16)
            self.mask_decoder = MaskDecoder(
                num_multimask_outputs=2, transformer=TwoWayTransformer(2, 256, 8, 2048),
                transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
        else:
            # Exact default geometry decoder from htcr/sam_road/model.py.
            self.map_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, 2), LayerNorm2d(128), nn.GELU(),
                nn.ConvTranspose2d(128, 64, 2, 2), nn.GELU(),
                nn.ConvTranspose2d(64, 32, 2, 2), nn.GELU(),
                nn.ConvTranspose2d(32, 2, 2, 2))
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53])[None, :, None, None], persistent=False)
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375])[None, :, None, None], persistent=False)
        self._checkpoint_loaded = False
        self.checkpoint_path = None
        self.provenance = None
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen means both parameter gradients AND train-time stochasticity disabled.
        return super().train(False)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        own = self.state_dict()
        selected = {}
        for key, value in state.items():
            # Accept plain upstream Lightning state or an exported geometry state.
            for prefix in ('module.', 'model.', 'geometry.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key in own:
                selected[key] = value
        # Strict: raw SAM, missing decoder, wrong resolution, and LoRA fail loudly.
        self.load_state_dict(selected, strict=True)
        if any('linear_a_' in k or 'linear_b_' in k for k in state):
            raise ValueError('LoRA geometry checkpoints require explicit conversion; not the default paper geometry')
        self._checkpoint_loaded = True
        self.checkpoint_path = str(Path(path).resolve())
        self.provenance = checkpoint.get('provenance')
        return self

    def validate_provenance(self, declared):
        if self.provenance:
            for key in ('training_map_ids', 'keypoint_semantics'):
                actual, supplied = self.provenance.get(key), declared.get(key)
                if key == 'training_map_ids':
                    actual, supplied = sorted(actual or []), sorted(supplied or [])
                if actual != supplied:
                    raise ValueError(f'Geometry checkpoint provenance disagrees with {key}')

    @torch.no_grad()
    def forward(self, rgb):
        if not self._checkpoint_loaded:
            raise RuntimeError('Load a trained SAM-Road geometry checkpoint before generating guidance')
        return self.mask_logits(rgb).sigmoid()[:, [1, 0]]

    def mask_logits(self, rgb):
        """SAM-Road channel order [keypoint, road]; gradients used in pretraining."""
        if rgb.ndim != 4 or rgb.shape[1:] != (3, self.patch_size, self.patch_size):
            raise ValueError('RGB shape must match the geometry checkpoint patch resolution')
        if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('RGB must be finite in [0,1]')
        embeddings = self.image_encoder((rgb * 255 - self.pixel_mean) / self.pixel_std)
        if self.use_sam_decoder:
            sparse, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
            # SAM's decoder repeats a single image for prompts; decode images individually.
            logits = torch.cat([self.mask_decoder(
                image_embeddings=e[None], image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True)[0] for e in embeddings])
        else:
            logits = self.map_decoder(embeddings)
        logits = F.interpolate(logits, rgb.shape[-2:], mode='bilinear', align_corners=False)
        return logits
