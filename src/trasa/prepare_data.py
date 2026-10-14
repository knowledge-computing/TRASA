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
from .data import manifest_errors, read_road_mask


def _integer_array(text):
    values = np.asarray((text or '').split(), dtype=float)
    if not np.isfinite(values).all() or (values != np.floor(values)).any():
        raise ValueError('VTP connectivity and offsets must be finite integers')
    return values.astype(np.int64)


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
            points = piece.find('./Points/DataArray')
            if points is not None and points.get('format', 'ascii') == 'ascii' and (points.text or '').strip():
                raise ValueError('VTP point count disagrees with Points array')
            if int(piece.get('NumberOfLines', 0)) or any(
                    array.get('Name') in ('connectivity', 'offsets') and array.get('format', 'ascii') == 'ascii'
                    and (array.text or '').strip() for array in piece.findall('./Lines/DataArray')):
                raise ValueError('VTP with no points cannot contain polylines')
            return np.empty((0, 2)), np.empty((0, 2), dtype=int)
        point_array = piece.find('./Points/DataArray')
        lines = piece.find('./Lines')
        arrays = {} if lines is None else {a.get('Name'): a for a in lines.findall('DataArray')}
        if point_array is not None and all(a.get('format', 'ascii') == 'ascii' for a in [point_array, *arrays.values()]):
            dimensions = int(point_array.get('NumberOfComponents', 3))
            points = np.asarray((point_array.text or '').split(), dtype=float).reshape(-1, dimensions)[:, :2]
            connectivity = _integer_array(arrays['connectivity'].text) if arrays else np.empty(0, int)
            offsets = _integer_array(arrays['offsets'].text) if arrays else np.empty(0, int)
            if 'NumberOfLines' in piece.attrib and int(piece.get('NumberOfLines')) != len(offsets):
                raise ValueError('VTP line count disagrees with polyline offsets')
            if len(points) != int(piece.get('NumberOfPoints', 0)):
                raise ValueError('VTP point count disagrees with Points array')
            if len(offsets) and ((np.diff(np.r_[0, offsets]) < 2).any() or offsets[-1] != len(connectivity)):
                raise ValueError('Invalid polyline offsets')
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
    extras, width_map = {}, None
    if path.suffix.lower() == '.vtp':
        nodes, edges = read_vtp(path)
    elif path.suffix.lower() == '.json':
        document = json.loads(path.read_text())
        extras = {key: value for key, value in document.items() if key not in ('nodes', 'edges', 'types', 'edge_widths')}
        nodes = np.array(document['nodes'], dtype=float)
        if nodes.ndim == 1 and nodes.size == 0:
            nodes = nodes.reshape(0, 2)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError('Graph nodes must have shape (N,2)')
        edges = document['edges']
        if 'edge_widths' in document:
            raw = np.asarray(edges)
            widths = np.asarray(document['edge_widths'], dtype=float)
            if widths.shape != (len(raw),) or not np.isfinite(widths).all() or (widths < 1).any() or (widths != np.floor(widths)).any():
                raise ValueError('One positive integer road width required per edge')
            width_map = {}
            for edge, width in zip(raw, widths):
                key = tuple(sorted(edge))
                if key in width_map and width_map[key] != width:
                    raise ValueError('Duplicate edges have conflicting road widths')
                width_map[key] = int(width)
    else:
        raise ValueError('Use JSON node/edge graphs or VTP; export other serialized formats first')
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError('Graph nodes must have shape (N,2)')
    if coordinates == 'normalized':
        nodes = nodes * (patch_size - 1)
    elif coordinates != 'pixels':
        raise ValueError('Choose normalized or pixels explicitly')
    if not np.isfinite(nodes).all() or (nodes < 0).any() or (nodes > patch_size - 1).any():
        raise ValueError('Graph coordinates outside patch; crop/align corrected annotations first')
    edges = canonical_edges(edges, len(nodes))
    indices = edges.numpy()
    if len(edges) and np.any(np.linalg.norm(nodes[indices[:, 0]] - nodes[indices[:, 1]], axis=1) <= 1e-12):
        raise ValueError('Reference edges must have positive geometric length')
    if drop_isolated and len(nodes):
        degree = torch.bincount(edges.flatten(), minlength=len(nodes)).numpy()
        keep = degree > 0
        mapping = np.cumsum(keep) - 1
        nodes = nodes[keep]
        edges = torch.tensor(mapping[edges.numpy()], dtype=torch.long).reshape(-1, 2)
    roles = node_types(len(nodes), edges)
    if width_map is not None:
        extras['edge_widths'] = [width_map[tuple(edge)] for edge in indices]
    return {**extras, 'nodes': nodes.tolist(), 'edges': edges.tolist(), 'types': roles.tolist()}


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
    errors = manifest_errors({**source, 'schema_version': 1}, index.parent)
    if errors:
        raise ValueError('\n'.join(errors))
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
            if pixels.mode != 'RGB':
                raise ValueError(f"Sample {row['id']}: image must be RGB")
            if pixels.size != (args.patch_size, args.patch_size):
                raise ValueError('Image dimensions differ from patch size')
        if row.get('road_mask'):
            read_road_mask(index.parent / row['road_mask'], args.patch_size)
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
    document = {**source, 'schema_version': 1, 'samples': rows}
    document.pop('guidance_provenance', None)
    for row in rows:
        row.pop('guidance', None)
        row.pop('guidance_provenance', None)
    (output / 'manifest.json').write_text(json.dumps(document, indent=2) + '\n')


if __name__ == '__main__':
    main()
