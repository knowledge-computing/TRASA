"""Export graph JSON to research JSON, WGS84 GeoJSON or CRS-preserving GeoPackage."""
import argparse
import json
from pathlib import Path

import numpy as np

from .graph import Graph, georeference_graph


def read_graph(path):
    value = json.loads(Path(path).read_text())
    count, edge_count = len(value['nodes']), len(value['edges'])
    return Graph(value['nodes'], value['edges'], np.asarray(value.get('types', [-1] * count)),
                 np.asarray(value.get('node_scores', [1.] * count)), np.asarray(value.get('edge_scores', [1.] * edge_count)),
                 value.get('metadata', {}))


def write_graph(graph, path):
    path = Path(path)
    if path.exists():
        raise ValueError('Graph output exists; choose a new file')
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.json':
        path.write_text(json.dumps(graph.asdict(), indent=2, allow_nan=False) + '\n')
        return
    if graph.metadata.get('coordinate_space', 'pixel') == 'pixel':
        if not graph.metadata.get('affine') or not graph.metadata.get('crs'):
            raise ValueError('GIS export requires an affine and CRS; pixel coordinates are not longitude/latitude')
        graph = georeference_graph(graph, graph.metadata['affine'], graph.metadata['crs'])
    crs = graph.metadata.get('crs')
    if not crs:
        raise ValueError('GIS export requires an explicit CRS')
    if suffix == '.geojson':
        try:
            from pyproj import Transformer
        except ImportError as error:
            raise RuntimeError("GeoJSON CRS transformation requires pip install -e '.[gis]'") from error
        transform = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        x, y = transform.transform(graph.nodes[:, 0], graph.nodes[:, 1])
        nodes = np.column_stack((x, y))
        if not np.isfinite(nodes).all():
            raise ValueError('CRS transformation produced nonfinite coordinates')
        if len(nodes) and ((np.abs(nodes[:, 0]) > 180).any() or (np.abs(nodes[:, 1]) > 90).any()):
            raise ValueError('GeoJSON coordinates outside longitude/latitude bounds; verify source CRS')
        features = []
        for index, point in enumerate(nodes):
            features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': point.tolist()},
                             'properties': {'node_id': index, 'node_type': int(graph.types[index]), 'score': float(graph.node_scores[index])}})
        for index, (a, b) in enumerate(graph.edges):
            features.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': nodes[[a, b]].tolist()},
                             'properties': {'edge_id': index, 'source': int(a), 'target': int(b), 'score': float(graph.edge_scores[index])}})
        collection = {'type': 'FeatureCollection', 'features': features,
                      'properties': {'source_graph_metadata': graph.metadata, 'output_crs': 'EPSG:4326'}}
        path.write_text(json.dumps(collection, indent=2, allow_nan=False) + '\n')
    elif suffix == '.gpkg':
        try:
            import fiona
            from fiona.crs import CRS
        except ImportError as error:
            raise RuntimeError("GeoPackage export requires pip install -e '.[gis]'") from error
        with fiona.open(path, 'w', driver='GPKG', layer='roads', crs=CRS.from_user_input(crs),
                        schema={'geometry': 'LineString', 'properties': {'source': 'int', 'target': 'int', 'score': 'float'}}) as layer:
            for index, (a, b) in enumerate(graph.edges):
                layer.write({'geometry': {'type': 'LineString', 'coordinates': graph.nodes[[a, b]].tolist()},
                             'properties': {'source': int(a), 'target': int(b), 'score': float(graph.edge_scores[index])}})
        with fiona.open(path, 'w', driver='GPKG', layer='nodes', crs=CRS.from_user_input(crs),
                        schema={'geometry': 'Point', 'properties': {'node_id': 'int', 'node_type': 'int', 'score': 'float'}}) as layer:
            for index, point in enumerate(graph.nodes):
                layer.write({'geometry': {'type': 'Point', 'coordinates': point.tolist()},
                             'properties': {'node_id': index, 'node_type': int(graph.types[index]), 'score': float(graph.node_scores[index])}})
    else:
        raise ValueError('Choose .json, .geojson or .gpkg output')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--affine', type=float, nargs=6)
    parser.add_argument('--crs')
    args = parser.parse_args(argv)
    graph = read_graph(args.input)
    if args.affine is not None:
        graph = georeference_graph(graph, args.affine, args.crs)
    elif args.crs:
        raise ValueError('A CRS override requires an affine')
    write_graph(graph, args.output)


if __name__ == '__main__':
    main()
