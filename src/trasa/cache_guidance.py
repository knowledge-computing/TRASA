"""Cache frozen geometry probabilities; this creates no reference annotations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import SAMRoadGeometry
from .runtime import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New cache directory')
    parser.add_argument('--training-map-ids', nargs='*', required=True,
                        help='Maps used to train geometry; pass flag alone for external pretrained data')
    parser.add_argument('--keypoint-semantics', choices=['intersection_only', 'degree_ne_2'], required=True)
    parser.add_argument('--sam-variant', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--sam-decoder', action='store_true')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    manifest = Path(args.manifest).resolve()
    document = json.loads(manifest.read_text())
    if document.get('schema_version') != 1:
        raise ValueError('Expected manifest schema_version 1')
    held_out = {row['map_id'] for row in document['samples'] if row['split'] != 'train'}
    if held_out & set(args.training_map_ids):
        raise ValueError('Geometry training includes validation/test maps')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Cache output exists; choose a new directory')
    geometry = SAMRoadGeometry(args.patch_size, args.sam_variant, args.sam_decoder).load_checkpoint(args.checkpoint).to(args.device)
    output.mkdir(parents=True)
    document['guidance_provenance'] = {'checkpoint_sha256': sha256(args.checkpoint),
        'channel_order': ['road', 'intersection'], 'keypoint_semantics': args.keypoint_semantics,
        'training_map_ids': args.training_map_ids, 'sam_variant': args.sam_variant,
        'use_sam_decoder': args.sam_decoder, 'patch_size': args.patch_size}
    for index, row in enumerate(document['samples']):
        image_path = (manifest.parent / row['image']).resolve()
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert('RGB'), np.float32).copy() / 255
        masks = geometry(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device))[0].cpu().numpy()
        cache = output / f'{index:08d}.npz'
        np.savez_compressed(cache, road=masks[0], intersection=masks[1])
        row['image'] = str(image_path)
        row['graph'] = str((manifest.parent / row['graph']).resolve())
        row['guidance'] = cache.name
    (output / 'manifest.json').write_text(json.dumps(document, indent=2) + '\n')


if __name__ == '__main__':
    main()
