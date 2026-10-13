"""Convert corrected JSON/VTP patch graphs into the training manifest contract."""
import argparse
import json
import os
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image

from .graph import canonical_edges, node_types


def read_vtp(path):
    """Read ASCII XML directly; use the official VTK reader for compressed data."""
    try:
        root = ET.parse(path).getroot()
        piece = root.find('./PolyData/Piece')
        if piece is None:
            raise ValueError('Expected VTK PolyData with one Piece')
        if len(root.findall('./PolyData/Piece')) != 1:
            raise ValueError('Multi-piece graphs require consolidation before conversion')
        if int(piece.get('NumberOfPoints', 0)) == 0:
            return np.empty((0, 2)), np.empty((0, 2), dtype=int)
        point_array = piece.find('./Points/DataArray')
        lines = piece.find('./Lines')
        arrays = {} if lines is None else {a.get('Name'): a for a in lines.findall('DataArray')}
        if point_array is not None and all(a.get('format', 'ascii') == 'ascii' for a in [point_array, *arrays.values()]):
            dimensions = int(point_array.get('NumberOfComponents', 3))
            points = np.fromstring(point_array.text or '', sep=' ').reshape(-1, dimensions)[:, :2]
            connectivity = np.fromstring(arrays['connectivity'].text or '', sep=' ', dtype=int) if arrays else np.empty(0, int)
            offsets = np.fromstring(arrays['offsets'].text or '', sep=' ', dtype=int) if arrays else np.empty(0, int)
            edges, begin = [], 0
            for end in offsets:
                line = connectivity[begin:end]
                edges.extend(zip(line[:-1], line[1:]))
                begin = int(end)
            if begin != len(connectivity):
                raise ValueError('Invalid polyline offsets')
            return points, np.array(edges, int).reshape(-1, 2)
    except ET.ParseError:
        pass  # Raw appended XML data is handled by VTK, not a text parser.
    try:
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        from vtkmodules.util.numpy_support import vtk_to_numpy
    except ImportError as error:
        raise RuntimeError("Compressed/appended VTP requires: pip install -e '.[data]'") from error
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetPoints() is None:
        raise ValueError('VTK did not produce point geometry')
    points = vtk_to_numpy(mesh.GetPoints().GetData())[:, :2]
    packed = vtk_to_numpy(mesh.GetLines().GetData())
    edges, position = [], 0
    while position < len(packed):
        length = int(packed[position])
        line = packed[position + 1:position + 1 + length]
        edges.extend(zip(line[:-1], line[1:]))
        position += length + 1
    return points, np.array(edges, int).reshape(-1, 2)


def convert_graph(path, coordinates, patch_size=256, drop_isolated=False):
    path = Path(path)
    if path.suffix.lower() == '.vtp':
        nodes, edges = read_vtp(path)
    elif path.suffix.lower() == '.json':
        document = json.loads(path.read_text())
        nodes = np.array(document['nodes'], dtype=float).reshape(-1, 2)
        edges = np.array(document['edges'], dtype=int).reshape(-1, 2)
    else:
        raise ValueError('Use JSON node/edge graphs or VTP; export other serialized formats first')
    if coordinates == 'normalized':
        nodes = nodes * (patch_size - 1)
    elif coordinates != 'pixels':
        raise ValueError('Choose normalized or pixels explicitly')
    if not np.isfinite(nodes).all() or (nodes < 0).any() or (nodes > patch_size - 1).any():
        raise ValueError('Graph coordinates outside patch; crop/align corrected annotations first')
    edges = canonical_edges(edges, len(nodes))
    if drop_isolated and len(nodes):
        degree = torch.bincount(edges.flatten(), minlength=len(nodes)).numpy()
        keep = degree > 0
        mapping = np.cumsum(keep) - 1
        nodes = nodes[keep]
        edges = torch.tensor(mapping[edges.numpy()], dtype=torch.long).reshape(-1, 2)
    roles = node_types(len(nodes), edges)
    return {'nodes': nodes.tolist(), 'edges': edges.tolist(), 'types': roles.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, help='JSON samples with id, map_id, split, image, graph and origin')
    parser.add_argument('--output', required=True, help='New prepared-manifest directory')
    parser.add_argument('--coordinates', choices=['normalized', 'pixels'], required=True)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--max-nodes', type=int, default=100)
    parser.add_argument('--drop-isolated', action='store_true', help='Explicitly remove unused VTP points')
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args(argv)
    index = Path(args.index).resolve()
    source = json.loads(index.read_text())
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    prepared, map_splits, ids = [], {}, set()
    # Validate every sample before materializing the output.
    for number, row in enumerate(source['samples']):
        if row['id'] in ids or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Duplicate sample id or invalid split')
        ids.add(row['id'])
        previous = map_splits.setdefault(row['map_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Map-level split leakage')
        if len(row.get('origin', [])) != 2:
            raise ValueError('Map-level evaluation requires the patch origin [x,y]')
        image = (index.parent / row['image']).resolve()
        with Image.open(image) as pixels:
            if pixels.size != (args.patch_size, args.patch_size):
                raise ValueError('Image dimensions differ from patch size')
        if row.get('road_mask'):
            with Image.open(index.parent / row['road_mask']) as mask:
                if mask.size != (args.patch_size, args.patch_size):
                    raise ValueError('Road annotation mask dimensions differ from patch size')
        graph = convert_graph(index.parent / row['graph'], args.coordinates, args.patch_size, args.drop_isolated)
        if len(graph['nodes']) > args.max_nodes:
            raise ValueError('Graph exceeds query capacity; revise annotation sampling, never truncate')
        prepared.append((row, image, graph))
    (output / 'graphs').mkdir(parents=True)
    rows = []
    for number, (row, image, graph) in enumerate(prepared):
        graph_name = f'graphs/{number:08d}.json'
        (output / graph_name).write_text(json.dumps(graph) + '\n')
        image_name = os.path.relpath(image, output)
        if args.copy_images:
            (output / 'images').mkdir(exist_ok=True)
            image_name = f'images/{number:08d}{image.suffix}'
            shutil.copy2(image, output / image_name)
        prepared_row = {**row, 'image': image_name, 'graph': graph_name}
        if row.get('road_mask'):
            mask = (index.parent / row['road_mask']).resolve()
            prepared_row['road_mask'] = os.path.relpath(mask, output)
            if args.copy_images:
                (output / 'labels').mkdir(exist_ok=True)
                prepared_row['road_mask'] = f'labels/road_{number:08d}{mask.suffix}'
                shutil.copy2(mask, output / prepared_row['road_mask'])
        rows.append(prepared_row)
    (output / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'samples': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
