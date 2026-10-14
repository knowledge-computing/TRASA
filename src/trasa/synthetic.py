"""Generate explicitly synthetic road images/annotations for demos, never a benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate(output, patch_size=256, seed=10):
    output = Path(output)
    if output.exists():
        raise ValueError('Synthetic output exists; choose a new directory')
    if patch_size < 32 or patch_size % 16:
        raise ValueError('Patch size must be at least 32 and divisible by 16')
    for folder in ('images', 'graphs', 'labels', 'maps'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, maps = [], {}
    for number in range(8):
        map_id = f'synthetic_{number:02d}'
        maps[map_id] = {'width': patch_size * 2, 'height': patch_size, 'source': 'synthetic_generator'}
        y = patch_size // 2 + int(rng.integers(-patch_size // 8, patch_size // 8 + 1))
        full_map = Image.new('RGB', (patch_size * 2, patch_size))
        for column in range(2):
            sample_id = f'{map_id}_{column}'
            nodes = [[0, y], [patch_size // 2, y], [patch_size - 1, y], [patch_size // 2, 0]]
            edges = [[0, 1], [1, 2], [1, 3]]
            if number == 7 and column == 1:
                nodes, edges = [], []
            rgb = Image.new('RGB', (patch_size, patch_size), (240, 236, 225))
            mask = Image.new('L', rgb.size)
            for a, b in edges:
                ImageDraw.Draw(rgb).line([tuple(nodes[a]), tuple(nodes[b])], fill=(80, 45, 35), width=3)
                ImageDraw.Draw(mask).line([tuple(nodes[a]), tuple(nodes[b])], fill=255, width=3)
            rgb.save(output / f'images/{sample_id}.png')
            full_map.paste(rgb, (column * patch_size, 0))
            mask.save(output / f'labels/{sample_id}.png')
            (output / f'graphs/{sample_id}.json').write_text(json.dumps({'nodes': nodes, 'edges': edges}) + '\n')
            rows.append({'id': sample_id, 'map_id': map_id, 'split': 'train',
                         'image': f'images/{sample_id}.png', 'graph': f'graphs/{sample_id}.json',
                         'road_mask': f'labels/{sample_id}.png', 'origin': [column * patch_size, 0]})
        full_map.save(output / f'maps/{map_id}.png')
    document = {'schema_version': 1, 'dataset_kind': 'synthetic', 'seed': seed,
                'description': 'Generated execution demonstration; not USGS data or the TRASA benchmark.',
                'maps': maps, 'samples': rows}
    (output / 'index.json').write_text(json.dumps(document, indent=2) + '\n')
    return output / 'index.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args(argv)
    print(generate(args.output, args.patch_size, args.seed))


if __name__ == '__main__':
    main()
