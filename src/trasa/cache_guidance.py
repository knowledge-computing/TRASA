"""Cache frozen geometry probabilities; this creates no reference annotations."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import SAMRoadGeometry
from .runtime import sha256
from .config import Config
from .data import PatchDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New cache directory')
    parser.add_argument('--training-map-ids', nargs='*',
                        help='Maps used to train geometry; pass flag alone for external pretrained data')
    parser.add_argument('--keypoint-semantics', choices=['intersection_only', 'degree_ne_2'])
    parser.add_argument('--validation-map-ids', nargs='*', help='Maps used for geometry checkpoint selection; flag alone for none')
    parser.add_argument('--sam-variant', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--sam-decoder', action='store_true')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    manifest = Path(args.manifest).resolve()
    document = json.loads(manifest.read_text())
    if document.get('schema_version') != 1:
        raise ValueError('Expected manifest schema_version 1')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Cache output exists; choose a new directory')
    geometry = SAMRoadGeometry(args.patch_size, args.sam_variant, args.sam_decoder).load_checkpoint(args.checkpoint).to(args.device)
    actual = geometry.provenance or {}
    training_maps = args.training_map_ids if args.training_map_ids is not None else actual.get('training_map_ids')
    validation_maps = args.validation_map_ids if args.validation_map_ids is not None else actual.get('validation_map_ids')
    semantics = args.keypoint_semantics or actual.get('keypoint_semantics')
    if training_maps is None or validation_maps is None or semantics is None:
        raise ValueError('External geometry checkpoints need explicit training/validation-map IDs and keypoint semantics')
    declared = {'training_map_ids': training_maps, 'validation_map_ids': validation_maps, 'keypoint_semantics': semantics}
    geometry.validate_provenance(declared)
    held_out = {row['map_id'] for row in document['samples'] if row['split'] != 'train'}
    if held_out & set(training_maps):
        raise ValueError('Geometry training includes validation/test maps')
    test_maps = {row['map_id'] for row in document['samples'] if row['split'] == 'test'}
    if test_maps & set(validation_maps):
        raise ValueError('Geometry checkpoint selection includes held-out test maps')
    # Known producer metadata can verify unchanged training/selection annotations.
    for split, key in (('train', 'training_data_sha256'), ('val', 'validation_data_sha256')):
        if key in actual:
            dataset = PatchDataset(manifest, split, Config(patch_size=args.patch_size, guidance=False), False)
            if dataset.fingerprint() != actual[key]:
                raise ValueError(f'Geometry checkpoint {split} data provenance changed; fine-tune on the current annotations')
    output.mkdir(parents=True)
    document['guidance_provenance'] = {'checkpoint_sha256': sha256(args.checkpoint),
        'channel_order': ['road', 'intersection'], 'keypoint_semantics': semantics,
        'training_map_ids': training_maps, 'validation_map_ids': validation_maps, 'sam_variant': args.sam_variant,
        'use_sam_decoder': args.sam_decoder, 'patch_size': args.patch_size, 'cache_schema_version': 2,
        'training_manifest_sha256': actual.get('manifest_sha256'), 'recipe': actual.get('recipe', 'external_declared')}
    for index, row in enumerate(document['samples']):
        image_path = (manifest.parent / row['image']).resolve()
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert('RGB'), np.float32).copy() / 255
        masks = geometry(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device))[0].cpu().numpy()
        cache = output / f'{index:08d}.npz'
        np.savez_compressed(cache, road=masks[0], intersection=masks[1],
                            checkpoint_sha256=document['guidance_provenance']['checkpoint_sha256'], image_sha256=sha256(image_path),
                            keypoint_semantics=semantics, sample_id=row['id'], map_id=row['map_id'])
        row['image'] = os.path.relpath(image_path, output)
        row['graph'] = os.path.relpath((manifest.parent / row['graph']).resolve(), output)
        if row.get('road_mask'):
            row['road_mask'] = os.path.relpath((manifest.parent / row['road_mask']).resolve(), output)
        row['guidance'] = cache.name
    (output / 'manifest.json').write_text(json.dumps(document, indent=2) + '\n')


if __name__ == '__main__':
    main()
