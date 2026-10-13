"""Fine-tune SAM-Road geometry on training maps before freezing it for TRASA."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .data import PatchDataset
from .geometry import SAMRoadGeometry
from .runtime import seed_everything, sha256


def geometry_targets(target, patch_size=256, road_width=3, radius=3, semantics='degree_ne_2'):
    if road_width < 1 or radius < 0 or semantics not in ('degree_ne_2', 'intersection_only'):
        raise ValueError('Invalid geometry label settings')
    nodes = target['nodes'].numpy() * (patch_size - 1)
    roads, points = Image.new('L', (patch_size, patch_size)), Image.new('L', (patch_size, patch_size))
    draw_road, draw_point = ImageDraw.Draw(roads), ImageDraw.Draw(points)
    for a, b in target['edges'].tolist():
        draw_road.line([tuple(np.rint(nodes[a]).astype(int)), tuple(np.rint(nodes[b]).astype(int))], fill=255, width=road_width)
    roles = target['types'].numpy()
    selected = roles != 2 if semantics == 'degree_ne_2' else roles == 3
    for x, y in nodes[selected]:
        x, y = round(float(x)), round(float(y))
        draw_point.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return torch.from_numpy(np.stack((np.asarray(points), np.asarray(roads))).astype(np.float32) / 255)


class GeometryDataset(Dataset):
    def __init__(self, manifest, split, patch_size=256, road_width=3, radius=3, semantics='degree_ne_2'):
        # Geometry training never consumes cached guidance or held-out labels.
        self.patches = PatchDataset(manifest, split, Config(patch_size=patch_size, guidance=False, ntc=False), False)
        self.patch_size, self.road_width, self.radius, self.semantics = patch_size, road_width, radius, semantics

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        sample = self.patches[index]
        row = self.patches.samples[index]
        width = int(row.get('road_width', self.road_width))
        labels = geometry_targets(sample['target'], self.patch_size, width, self.radius, self.semantics)
        if row.get('road_mask'):
            with Image.open(self.patches.path.parent / row['road_mask']) as image:
                mask = np.asarray(image.convert('L'), dtype=np.float32).copy()
            if mask.max() > 1:
                mask /= 255
            if mask.shape != (self.patch_size, self.patch_size):
                raise ValueError('Annotated road mask dimensions differ from patch')
            labels[1] = torch.from_numpy(mask)
        return sample['rgb'], labels


def initialize_encoder(model, checkpoint):
    """Load every SAM image-encoder parameter with upstream positional resizing."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state.get('state_dict', state)
    selected = {}
    own = model.image_encoder.state_dict()
    for key, expected in own.items():
        full = 'image_encoder.' + key
        if full not in state:
            raise ValueError(f'SAM checkpoint lacks {full}')
        value = state[full]
        if key == 'pos_embed' and value.shape != expected.shape:
            value = F.interpolate(value.permute(0, 3, 1, 2), expected.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        elif ('rel_pos_h' in key or 'rel_pos_w' in key) and value.shape != expected.shape:
            value = F.interpolate(value[None, None], expected.shape, mode='bilinear', align_corners=False)[0, 0]
        if value.shape != expected.shape:
            raise ValueError(f'SAM encoder shape mismatch at {full}')
        selected[key] = value
    model.image_encoder.load_state_dict(selected, strict=True)


def geometry_epoch(model, loader, device, optimizer=None):
    # The frozen adapter's public train() always remains eval; here pretraining
    # deliberately sets module mode and calls the differentiable logits method.
    nn.Module.train(model, optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for rgb, labels in loader:
            loss = F.binary_cross_entropy_with_logits(model.mask_logits(rgb.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite geometry objective')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(rgb)
            count += len(rgb)
    return total / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--encoder-lr-factor', type=float, default=0.1)
    parser.add_argument('--road-width', type=int, default=3)
    parser.add_argument('--keypoint-radius', type=int, default=3)
    parser.add_argument('--keypoint-semantics', default='degree_ne_2', choices=['degree_ne_2', 'intersection_only'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', help='Trusted geometry training checkpoint')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.encoder_lr_factor <= 0:
        raise ValueError('Positive geometry-training settings required')
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ValueError('Output exists; choose a new directory')
    seed_everything(args.seed)
    train = GeometryDataset(args.manifest, 'train', args.patch_size, args.road_width, args.keypoint_radius, args.keypoint_semantics)
    val = GeometryDataset(args.manifest, 'val', args.patch_size, args.road_width, args.keypoint_radius, args.keypoint_semantics)
    model = SAMRoadGeometry(args.patch_size, args.variant).to(args.device)
    if not args.resume:
        initialize_encoder(model, args.sam_checkpoint)
    model.requires_grad_(True)
    optimizer = torch.optim.Adam([
        {'params': model.image_encoder.parameters(), 'lr': args.learning_rate * args.encoder_lr_factor},
        {'params': model.map_decoder.parameters(), 'lr': args.learning_rate}])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, args.batch_size, shuffle=True, generator=generator)
    validation = DataLoader(val, args.batch_size, shuffle=False)
    provenance = {'training_map_ids': sorted({r['map_id'] for r in train.patches.samples}),
                  'validation_map_ids': sorted({r['map_id'] for r in val.patches.samples}),
                  'sam_checkpoint_sha256': sha256(args.sam_checkpoint),
                  'manifest_sha256': sha256(args.manifest), 'training_data_sha256': train.patches.fingerprint(),
                  'validation_data_sha256': val.patches.fingerprint(), 'keypoint_semantics': args.keypoint_semantics}
    settings = {k: v for k, v in vars(args).items() if k not in ('manifest', 'sam_checkpoint', 'output', 'resume')}
    output.mkdir(parents=True, exist_ok=True)
    best, start = float('inf'), 0
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        comparable = lambda values: {k: v for k, v in values.items() if k != 'epochs'}
        if previous['provenance'] != provenance or comparable(previous['settings']) != comparable(settings):
            raise ValueError('Geometry resume settings or data provenance changed')
        model.load_state_dict(previous['state_dict'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        random.setstate(previous['rng_python'])
        saved_numpy = previous['rng_numpy']
        np.random.set_state((saved_numpy[0], np.asarray(saved_numpy[1], dtype=np.uint32), *saved_numpy[2:]))
        torch.set_rng_state(previous['rng_torch'])
        if torch.cuda.is_available() and previous['rng_cuda']:
            torch.cuda.set_rng_state_all(previous['rng_cuda'])
        generator.set_state(previous['loader_rng'])
        best, start = previous['best_val'], previous['epoch'] + 1
    for epoch in range(start, args.epochs):
        training_loss = geometry_epoch(model, loader, args.device, optimizer)
        validation_loss = geometry_epoch(model, validation, args.device)
        improved = validation_loss < best
        best = min(best, validation_loss)
        numpy_state = np.random.get_state()
        state = {'state_dict': model.state_dict(), 'provenance': provenance, 'settings': settings,
                 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'best_val': best,
                 'rng_python': random.getstate(),
                 'rng_numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                 'rng_torch': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'loader_rng': generator.get_state()}
        torch.save(state, output / 'last.ckpt')
        if improved:
            torch.save(state, output / 'best.ckpt')
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train_loss': training_loss, 'val_loss': validation_loss}) + '\n')
        print(f'epoch={epoch} train={training_loss:.6f} val={validation_loss:.6f}', flush=True)


if __name__ == '__main__':
    main()
