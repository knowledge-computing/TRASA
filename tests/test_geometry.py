import pytest
import torch
from torch import nn

from trasa.geometry import SAMRoadGeometry


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return torch.zeros(len(x), 256, 16, 16)


def test_geometry_checkpoint_channel_order_normalization_and_freezing(monkeypatch, tmp_path):
    monkeypatch.setattr('trasa.geometry.ImageEncoderViT', TinyEncoder)
    geometry = SAMRoadGeometry()
    checkpoint = tmp_path / 'geometry.ckpt'
    with torch.no_grad():
        for p in geometry.map_decoder.parameters():
            p.zero_()
        geometry.map_decoder[-1].bias.copy_(torch.tensor([-2., 2.]))
    torch.save({'state_dict': {**geometry.state_dict(), 'topo_net.unused': torch.ones(1)}}, checkpoint)
    geometry.load_checkpoint(checkpoint)
    geometry.train(True)
    assert not geometry.training and not any(p.requires_grad for p in geometry.parameters())
    rgb = torch.ones(2, 3, 256, 256, requires_grad=True)
    masks = geometry(rgb)
    assert masks.shape == (2, 2, 256, 256) and not masks.requires_grad
    torch.testing.assert_close(masks[:, 0], torch.full((2, 256, 256), torch.sigmoid(torch.tensor(2.)).item()))
    torch.testing.assert_close(masks[:, 1], torch.full((2, 256, 256), torch.sigmoid(torch.tensor(-2.)).item()))
    torch.testing.assert_close(geometry.image_encoder.seen, (rgb.detach() * 255 - geometry.pixel_mean) / geometry.pixel_std)


def test_geometry_rejects_random_or_vanilla_sam(monkeypatch, tmp_path):
    monkeypatch.setattr('trasa.geometry.ImageEncoderViT', TinyEncoder)
    geometry = SAMRoadGeometry()
    with pytest.raises(RuntimeError, match='trained'):
        geometry(torch.zeros(1, 3, 256, 256))
    checkpoint = tmp_path / 'vanilla.pth'
    torch.save({'image_encoder.weight': torch.ones(1)}, checkpoint)
    with pytest.raises(RuntimeError, match='Missing key'):
        geometry.load_checkpoint(checkpoint)


@pytest.mark.parametrize('patch_size', [32, 256])
def test_real_sam_encoder_and_geometry_decoder_forward(patch_size):
    # Synthetic initialization verifies architecture/imports, not pretrained accuracy.
    geometry = SAMRoadGeometry(patch_size=patch_size)
    geometry.map_decoder.requires_grad_(True)
    logits = geometry.mask_logits(torch.rand(1, 3, patch_size, patch_size))
    assert logits.shape == (1, 2, patch_size, patch_size)
    assert torch.isfinite(logits).all() and len(geometry.image_encoder.blocks) == 12
    logits.square().mean().backward()
    assert any(p.grad is not None for p in geometry.map_decoder.parameters())
    assert all(p.grad is None for p in geometry.image_encoder.parameters())
