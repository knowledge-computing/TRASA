"""Export a fold for the native baseline codebases, and import their predictions back.

``cityscale`` (SAM-Road, Sat2Graph): each map is stitched from its patches and
cut into square regions (2048 px by default) written as
``20cities/region_<i>_sat.png`` with the reference graph as
``region_<i>_refine_gt_graph.p`` and ``region_<i>_graph_gt.pickle`` (adjacency
dict of (row, col) nodes). Both codebases split regions by index
(train: i % 10 < 8, validation: i % 20 == 18, test: i % 10 == 9 or i % 20 == 8),
so training/validation maps get training-or-validation indices and the test map
gets test indices; no upstream file needs editing. ``trasa_regions.json``
records which map and offset each index came from.

``vtp`` (RelationFormer, LDTR): patches as ``<split>/raw/sample_<n>_data.png``,
``<split>/seg/sample_<n>_seg.png`` and ``<split>/vtp/sample_<n>_graph.vtp``
with x, y normalized by (patch_size - 1), matching ``trasa.baselines --kind vtp
--coordinates normalized`` on the way back.

``import-cityscale`` turns region graph pickles from SAM-Road/Sat2Graph
inference back into complete-map pixel graphs, and ``import-vtp`` turns per-patch
VTP predictions back into pixel graphs; both write an index for
``trasa.baseline_evaluate``.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .data import PatchDataset
from .export import write_graph
from .graph import Graph
from .predict_dataset import merge_map, reference_graph


def train_indices():
    return [i for i in range(100000) if i % 10 < 8 or i % 20 == 18]


def test_indices():
    return [i for i in range(100000) if i % 10 == 9 or i % 20 == 8]


def stitched_maps(manifest, patch_size):
    """{map_id: (rgb image, merged reference graph, split)} from a prepared fold manifest."""
    config = Config(patch_size=patch_size, guidance=False, ntc=False)
    result = {}
    for split in ('train', 'val', 'test'):
        try:
            dataset = PatchDataset(manifest, split, config, False)
        except ValueError as error:
            if 'No samples' in str(error):
                continue
            raise
        groups = defaultdict(list)
        for index, row in enumerate(dataset.samples):
            groups[row['map_id']].append((row, dataset[index]))
        for map_id, items in groups.items():
            spatial = dataset.document.get('maps', {}).get(map_id, {})
            width = spatial.get('width') or max(r['origin'][0] for r, _ in items) + patch_size
            height = spatial.get('height') or max(r['origin'][1] for r, _ in items) + patch_size
            canvas = np.full((height, width, 3), 255, np.uint8)
            for row, sample in items:
                x, y = row['origin']
                patch = (sample['rgb'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                h, w = min(patch_size, height - y), min(patch_size, width - x)
                canvas[y:y + h, x:x + w] = patch[:h, :w]
            graphs = [reference_graph(sample['target'], patch_size) for _, sample in items]
            reference = merge_map(graphs, [row['origin'] for row, _ in items], config, spatial)
            result[map_id] = (canvas, reference, split)
    return result


def adjacency(graph, offset, size):
    """Sat2Graph adjacency {(r, c): [(r, c), ...]} of the part of a graph inside one region."""
    from shapely.geometry import LineString, box
    x0, y0 = offset
    window = box(x0, y0, x0 + size - 1, y0 + size - 1)
    result = defaultdict(set)
    for a, b in graph.edges:
        piece = LineString(graph.nodes[[a, b]]).intersection(window)
        if piece.is_empty or piece.geom_type != 'LineString':
            continue
        (xa, ya), (xb, yb) = piece.coords[0], piece.coords[-1]
        u = (int(round(ya - y0)), int(round(xa - x0)))
        v = (int(round(yb - y0)), int(round(xb - x0)))
        if u != v:
            result[u].add(v)
            result[v].add(u)
    return {node: sorted(neighbors) for node, neighbors in sorted(result.items())}


def export_cityscale(manifest, output, patch_size=256, region=2048, max_index=180):
    maps = stitched_maps(manifest, patch_size)
    folder = Path(output) / '20cities'
    folder.mkdir(parents=True)
    pools = {'train': iter(train_indices()), 'test': iter(test_indices())}
    regions, split_lists = [], {'train': [], 'validation': [], 'test': []}
    for map_id in sorted(maps):
        image, reference, split = maps[map_id]
        pool = 'test' if split == 'test' else 'train'
        height, width = image.shape[:2]
        for y in range(0, height, region):
            for x in range(0, width, region):
                index = next(pools[pool])
                tile = np.full((region, region, 3), 255, np.uint8)
                crop = image[y:y + region, x:x + region]
                tile[:crop.shape[0], :crop.shape[1]] = crop
                Image.fromarray(tile).save(folder / f'region_{index}_sat.png')
                graph = adjacency(reference, (x, y), region)
                for name in (f'region_{index}_refine_gt_graph.p', f'region_{index}_graph_gt.pickle'):
                    with (folder / name).open('wb') as stream:
                        pickle.dump(graph, stream)
                regions.append({'index': index, 'map_id': map_id, 'split': split, 'origin': [x, y],
                                'valid_size': [int(crop.shape[1]), int(crop.shape[0])]})
                split_lists['test' if split == 'test' else 'validation' if index % 20 == 18 else 'train'].append(index)
    largest = max(r['index'] for r in regions)
    if largest >= max_index:
        raise ValueError(f'Region index {largest} exceeds the upstream cityscale partition (indices < {max_index}). '
                         'Use a larger --region-size, a coarser dataset downsample, or extend range(180) upstream '
                         'and pass --max-index.')
    document = {'region_size': region, 'patch_size': patch_size, 'manifest': str(Path(manifest).resolve()),
                'regions': regions, 'data_split': split_lists}
    (Path(output) / 'trasa_regions.json').write_text(json.dumps(document, indent=2) + '\n')
    (Path(output) / 'data_split.json').write_text(json.dumps(split_lists) + '\n')
    return document


def write_vtp(path, nodes, edges):
    points = ' '.join(f'{x:.6f} {y:.6f} 0' for x, y in nodes)
    connectivity = ' '.join(f'{a} {b}' for a, b in edges)
    offsets = ' '.join(str(2 * (i + 1)) for i in range(len(edges)))
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n<PolyData>\n'
        f'<Piece NumberOfPoints="{len(nodes)}" NumberOfVerts="0" NumberOfLines="{len(edges)}" NumberOfStrips="0" NumberOfPolys="0">\n'
        f'<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{points}</DataArray></Points>\n'
        f'<Lines><DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'<DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray></Lines>\n'
        '</Piece>\n</PolyData>\n</VTKFile>\n')


def export_vtp(manifest, output, patch_size=256):
    config = Config(patch_size=patch_size, guidance=False, ntc=False)
    ids = {}
    for split in ('train', 'val', 'test'):
        try:
            dataset = PatchDataset(manifest, split, config, False)
        except ValueError as error:
            if 'No samples' in str(error):
                continue
            raise
        root = Path(output) / split
        for folder in ('raw', 'seg', 'vtp'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(dataset.samples):
            sample = dataset[index]
            name = f'sample_{len(ids):06d}'
            ids[name] = {'id': row['id'], 'map_id': row['map_id'], 'split': split, 'origin': row['origin']}
            rgb = (sample['rgb'].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(root / 'raw' / f'{name}_data.png')
            if row.get('road_mask'):
                from .data import read_road_mask
                mask = read_road_mask(dataset.path.parent / row['road_mask'], patch_size)
            else:
                from .train_geometry import geometry_targets
                mask = geometry_targets(sample['target'], patch_size)[1].numpy()
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(root / 'seg' / f'{name}_seg.png')
            nodes = sample['target']['nodes'].numpy()  # already normalized by (patch_size - 1)
            write_vtp(root / 'vtp' / f'{name}_graph.vtp', nodes, sample['target']['edges'].numpy())
    (Path(output) / 'trasa_samples.json').write_text(json.dumps({'patch_size': patch_size, 'normalized_by': patch_size - 1,
                                                                 'samples': ids}, indent=2) + '\n')
    return ids


def import_cityscale(regions_file, predictions, output, method, pattern='{index}.p', axis_order='yx'):
    document = json.loads(Path(regions_file).read_text())
    groups = defaultdict(list)
    for region in document['regions']:
        if region['split'] == 'test':
            groups[region['map_id']].append(region)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    maps = []
    for map_id, regions in sorted(groups.items()):
        nodes, index, edges = [], {}, set()
        for region in regions:
            path = Path(predictions) / pattern.format(index=region['index'])
            with path.open('rb') as stream:
                graph = pickle.load(stream)  # trusted output of your own baseline run
            x0, y0 = region['origin']
            w, h = region['valid_size']

            def key(point):
                r, c = (point[0], point[1]) if axis_order == 'yx' else (point[1], point[0])
                if not (0 <= c < w and 0 <= r < h):
                    return None
                location = (float(c) + x0, float(r) + y0)
                if location not in index:
                    index[location] = len(nodes)
                    nodes.append(location)
                return index[location]
            for point, neighbors in graph.items():
                for other in neighbors:
                    a, b = key(point), key(other)
                    if a is not None and b is not None and a != b:
                        edges.add(tuple(sorted((a, b))))
        edge_list = sorted(edges)
        merged = Graph(np.asarray(nodes, float).reshape(-1, 2), np.asarray(edge_list, int).reshape(-1, 2),
                       np.full(len(nodes), -1), np.ones(len(nodes)), np.ones(len(edge_list)),
                       {'coordinate_space': 'pixel', 'external_method': method, 'map_id': map_id})
        name = f'{map_id}.json'
        write_graph(merged, output / name)
        maps.append({'map_id': map_id, 'graph': name})
    return maps


def import_vtp(samples_file, predictions, output, method, pattern='{name}_graph.vtp'):
    """Per-patch VTP predictions (normalized coordinates) -> pixel graphs + sample list."""
    from .prepare_data import read_vtp
    document = json.loads(Path(samples_file).read_text())
    scale = document['normalized_by']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for name, sample in sorted(document['samples'].items()):
        if sample['split'] != 'test':
            continue
        nodes, edges = read_vtp(Path(predictions) / pattern.format(name=name))
        graph = Graph(np.asarray(nodes, float).reshape(-1, 2) * scale, edges, np.full(len(nodes), -1),
                      np.ones(len(nodes)), np.ones(len(edges)), {'coordinate_space': 'pixel', 'external_method': method})
        write_graph(graph, output / f"{sample['id']}.json")
        samples.append({'id': sample['id'], 'graph': f"{sample['id']}.json"})
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('cityscale', 'vtp'):
        command = sub.add_parser(name)
        command.add_argument('--manifest', required=True, help='Fold manifest (prepared data)')
        command.add_argument('--output', required=True)
        command.add_argument('--patch-size', type=int, default=256)
        if name == 'cityscale':
            command.add_argument('--region-size', type=int, default=2048)
            command.add_argument('--max-index', type=int, default=180, help='Upstream cityscale_data_partition range')
    back = sub.add_parser('import-cityscale')
    back.add_argument('--regions', required=True, help='trasa_regions.json written by the cityscale export')
    back.add_argument('--predictions', required=True, help='Folder with one graph pickle per test region')
    back.add_argument('--pattern', default='{index}.p', help="File name pattern, e.g. '{index}.p' (SAM-Road graph/)")
    back.add_argument('--axis-order', choices=['yx', 'xy'], default='yx')
    back.add_argument('--method', required=True)
    back.add_argument('--output', required=True)
    vtp_back = sub.add_parser('import-vtp')
    vtp_back.add_argument('--samples', required=True, help='trasa_samples.json written by the vtp export')
    vtp_back.add_argument('--predictions', required=True, help='Folder with one predicted VTP per test patch')
    vtp_back.add_argument('--pattern', default='{name}_graph.vtp')
    vtp_back.add_argument('--method', required=True)
    vtp_back.add_argument('--output', required=True)
    vtp_back.add_argument('--manifest', required=True, help='Fold manifest the export came from')
    vtp_back.add_argument('--source-revision')
    vtp_back.add_argument('--checkpoint-sha256')
    back.add_argument('--training-map-ids', nargs='*', default=None)
    back.add_argument('--validation-map-ids', nargs='*', default=None)
    back.add_argument('--source-revision')
    back.add_argument('--checkpoint-sha256')
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise ValueError('Output exists; choose a new directory')
    if args.command == 'cityscale':
        export_cityscale(args.manifest, args.output, args.patch_size, args.region_size, args.max_index)
    elif args.command == 'vtp':
        export_vtp(args.manifest, args.output, args.patch_size)
    elif args.command == 'import-vtp':
        samples = import_vtp(args.samples, args.predictions, args.output, args.method, args.pattern)
        fold = json.loads(Path(args.manifest).read_text())
        train = sorted({r['map_id'] for r in fold['samples'] if r['split'] == 'train'})
        val = sorted({r['map_id'] for r in fold['samples'] if r['split'] == 'val'})
        index = {'training_map_ids': train, 'validation_map_ids': val, 'example_map_ids': [],
                 'source_revision': args.source_revision, 'checkpoint_sha256': args.checkpoint_sha256,
                 'configuration': {}, 'samples': samples}
        (Path(args.output) / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    else:
        regions = json.loads(Path(args.regions).read_text())
        maps = import_cityscale(args.regions, args.predictions, args.output, args.method, args.pattern, args.axis_order)
        fold = json.loads(Path(regions['manifest']).read_text())
        train = sorted({r['map_id'] for r in fold['samples'] if r['split'] == 'train'})
        val = sorted({r['map_id'] for r in fold['samples'] if r['split'] == 'val'})
        index = {'training_map_ids': args.training_map_ids if args.training_map_ids is not None else train + val,
                 'validation_map_ids': args.validation_map_ids if args.validation_map_ids is not None else [],
                 'example_map_ids': [], 'source_revision': args.source_revision,
                 'checkpoint_sha256': args.checkpoint_sha256,
                 'configuration': {'region_size': regions['region_size']}, 'maps': maps}
        (Path(args.output) / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
