"""Build the patch benchmark from USGS sheets and corrected road annotations.

For every map in the spec this command
1. reads the USGS GeoTIFF (fetched by ``trasa.fetch_sources``, SHA256-checked),
2. finds the map frame from the quadrangle neatline (or explicit pixel corners),
3. cuts the frame into 256x256 patches on a fixed grid and keeps patches whose
   area is mostly inside the frame,
4. turns the corrected road lines into a road graph (nodes at road ends and at
   shared vertices; crossings without a shared vertex stay unconnected, as
   bridges/overpasses should), clips it to every patch, simplifies it and
   places intermediate nodes at most ``node_spacing`` pixels apart,
5. writes RGB patches, patch graphs with road types and widths, line-width
   road masks, map-level reference graphs, ``index.json`` for
   ``trasa.prepare_data``, the fold schedule and SHA256 checksums.

All steps are deterministic: same spec + same source bytes + pinned
libraries give the same files, which ``trasa.verify_dataset`` checks.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset_spec import Spec, default_cache, file_sha256, raster_file, source_path
from .road_types import ROAD_TYPES, code as road_type_code

CHECKSUM_EXCLUDE = ('checksums.json', 'build_report.json', 'previews/')
OVERLAY = (83, 111, 191)  # #536FBF


# ---------------------------------------------------------------- utilities

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format='PNG', optimize=False, compress_level=6)


def checked_transformer(source, target, allow_ballpark=False):
    """CRS transformer that refuses a silent 'ballpark' datum shift (~100 m error for NAD27)."""
    from pyproj import CRS, Transformer
    source, target = CRS.from_user_input(source), CRS.from_user_input(target)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    description = transformer.description or ''
    if 'ballpark' in description.lower() and not allow_ballpark:
        raise ValueError(f'Only a ballpark transformation is available from {source.name} to {target.name}. '
                         'Install the PROJ datum grids (e.g. projsync --area-of-use "United States"), '
                         'give the neatline/annotation in the raster CRS, or set allow_ballpark_transforms.')
    return transformer, description


class Grid:
    """Patch-grid pixel coordinates: x = source column / downsample - 0.5 (pixel centers)."""

    def __init__(self, transform, width, height, downsample):
        self.transform, self.d = transform, downsample
        self.inverse = ~transform
        self.width, self.height = width // downsample, height // downsample

    def from_world(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        col, row = self.inverse * (xy[:, 0], xy[:, 1])
        return np.column_stack((np.asarray(col) / self.d - .5, np.asarray(row) / self.d - .5))

    def from_source_pixels(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return (xy + .5) / self.d - .5

    def gdal_affine(self, x0, y0):
        from affine import Affine
        grid = self.transform * Affine.translation(x0 * self.d, y0 * self.d) * Affine.scale(self.d)
        return [grid.c, grid.a, grid.b, grid.f, grid.d, grid.e]


def densify_ring(west, south, east, north, steps=64):
    t = np.linspace(0, 1, steps, endpoint=False)
    edges = [np.column_stack((west + (east - west) * t, np.full(steps, south))),
             np.column_stack((np.full(steps, east), south + (north - south) * t)),
             np.column_stack((east - (east - west) * t, np.full(steps, north))),
             np.column_stack((np.full(steps, west), north - (north - south) * t))]
    return np.concatenate(edges)


def neatline_polygon(entry, grid, raster_crs, allow_ballpark):
    from shapely.geometry import Polygon, box
    neatline = entry.get('neatline')
    if neatline is None:
        return box(-.5, -.5, grid.width - .5, grid.height - .5), 'whole_raster'
    if 'pixels' in neatline:
        polygon = Polygon(grid.from_source_pixels(neatline['pixels']))
        description = 'source_pixels'
    else:
        ring = densify_ring(neatline['west'], neatline['south'], neatline['east'], neatline['north'])
        transformer, description = checked_transformer(neatline['crs'], raster_crs, allow_ballpark)
        x, y = transformer.transform(ring[:, 0], ring[:, 1])
        polygon = Polygon(grid.from_world(np.column_stack((x, y))))
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"{entry['map_id']}: neatline does not form a valid polygon in raster pixels")
    return polygon, description


# --------------------------------------------------------- annotation graph

def read_annotation(path, entry, spec, raster_crs, grid):
    """Corrected road lines as (grid coordinates, road type code) pairs."""
    import fiona
    from shapely.geometry import shape
    settings = spec.settings
    options = {'layer': entry['annotation']['layer']} if entry['annotation'].get('layer') else {}
    field, mapping, drop = settings['road_type_field'], settings['road_type_map'], set(settings['drop_types'])
    lines, unknown, transformer, description = [], set(), None, 'same_crs'
    coordinates = entry['annotation'].get('coordinates', 'crs')
    with fiona.open(path, **options) as collection:
        if coordinates == 'crs':
            crs = entry['annotation'].get('crs') or collection.crs_wkt
            if not crs:
                raise ValueError(f"{entry['map_id']}: annotation has no CRS; set annotation.crs")
            from pyproj import CRS
            if not CRS.from_user_input(crs).equals(CRS.from_user_input(raster_crs)):
                transformer, description = checked_transformer(crs, raster_crs, settings['allow_ballpark_transforms'])
        elif coordinates != 'source_pixels':
            raise ValueError('annotation.coordinates must be crs or source_pixels')
        for feature in collection:
            value = feature['properties'].get(field)
            if value in drop:
                continue
            road_type = mapping.get(value, value)
            if road_type not in ROAD_TYPES:
                unknown.add(value)
                continue
            geometry = shape(feature['geometry']) if feature['geometry'] else None
            if geometry is None or geometry.is_empty:
                continue
            parts = list(geometry.geoms) if hasattr(geometry, 'geoms') else [geometry]
            for part in parts:
                if part.geom_type != 'LineString':
                    raise ValueError(f"{entry['map_id']}: annotation geometry must be lines, found {part.geom_type}")
                xy = np.asarray(part.coords, float)[:, :2]
                if transformer is not None:
                    x, y = transformer.transform(xy[:, 0], xy[:, 1])
                    xy = np.column_stack((x, y))
                xy = grid.from_world(xy) if coordinates == 'crs' else grid.from_source_pixels(xy)
                lines.append((xy, road_type_code(road_type)))
    if unknown:
        raise ValueError(f"{entry['map_id']}: annotation road types {sorted(map(str, unknown))} are not paper types; "
                         'add them to road_type_map or drop_types')
    return lines, description


def _thin(xy, tolerance):
    """Drop repeated vertices and vertices closer than the snap tolerance (endpoints kept)."""
    keep = [xy[0]]
    for point in xy[1:-1]:
        if np.hypot(*(point - keep[-1])) > max(tolerance, 1e-9):
            keep.append(point)
    if np.hypot(*(xy[-1] - keep[-1])) > 1e-9 or len(keep) == 1:
        if len(keep) > 1 and np.hypot(*(xy[-1] - keep[-1])) <= tolerance:
            keep[-1] = xy[-1]
        else:
            keep.append(xy[-1])
    return np.asarray(keep)


def _insert(xy, distances):
    """Insert vertices at the given distances along a polyline."""
    if not distances:
        return xy
    lengths = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    points = [(float(d), None) for d in distances] + [(float(l), i) for i, l in enumerate(lengths)]
    out = []
    for distance, index in sorted(points, key=lambda p: (p[0], p[1] is None)):
        if index is not None:
            out.append(xy[index])
            continue
        segment = min(max(int(np.searchsorted(lengths, distance, side='right')) - 1, 0), len(xy) - 2)
        span = lengths[segment + 1] - lengths[segment]
        t = 0 if span <= 0 else (distance - lengths[segment]) / span
        point = xy[segment] + t * (xy[segment + 1] - xy[segment])
        if not out or np.hypot(*(point - out[-1])) > 1e-9:
            out.append(point)
    return _thin(np.asarray(out), 0)


def map_graph(lines, frame, snap):
    """Node the corrected lines: returns node positions and typed chains between nodes."""
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    clipped = []
    for xy, road_type in lines:
        xy = _thin(xy, 0)
        if len(xy) < 2:
            continue
        geometry = LineString(xy).intersection(frame)
        parts = list(geometry.geoms) if hasattr(geometry, 'geoms') else [geometry]
        for part in parts:
            if part.geom_type == 'LineString' and part.length > 0:
                clipped.append((_thin(np.asarray(part.coords), snap), road_type))
    clipped = [(xy, t) for xy, t in clipped if len(xy) >= 2 and LineString(xy).length > 0]
    if not clipped:
        return np.empty((0, 2)), []
    # T-junctions: a road end that touches another road between its vertices gets a vertex there.
    geometries = [LineString(xy) for xy, _ in clipped]
    tree = STRtree(geometries)
    inserts = defaultdict(list)
    for index, (xy, _) in enumerate(clipped):
        for end in (xy[0], xy[-1]):
            point = Point(end)
            for other in tree.query(point, predicate='dwithin', distance=snap):
                if other == index:
                    continue
                line = geometries[other]
                position = line.project(point)
                if snap < position < line.length - snap and line.distance(point) <= snap:
                    inserts[int(other)].append(position)
    clipped = [(_insert(xy, sorted(set(round(d, 9) for d in inserts.get(i, [])))), t) for i, (xy, t) in enumerate(clipped)]
    # Cluster vertices: road ends, and vertices shared by different roads, become nodes.
    owners = [(line, vertex) for line, (xy, _) in enumerate(clipped) for vertex in range(len(xy))]
    points = np.concatenate([xy for xy, _ in clipped])
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    is_end = np.array([vertex in (0, len(clipped[line][0]) - 1) for line, vertex in owners])
    for a, b in sorted(cKDTree(points).query_pairs(snap)) if snap > 0 else []:
        same_line = owners[a][0] == owners[b][0]
        if same_line and not (is_end[a] and is_end[b]):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    if snap == 0:
        # Exact coordinate equality still joins roads that share a vertex.
        seen = {}
        for i, point in enumerate(points):
            key = (float(point[0]), float(point[1]))
            if key in seen:
                ra, rb = find(i), find(seen[key])
                parent[max(ra, rb)] = min(ra, rb)
            else:
                seen[key] = i
    clusters = defaultdict(list)
    for i in range(len(points)):
        clusters[find(i)].append(i)
    node_of, nodes = {}, []
    for root in sorted(clusters):
        members = clusters[root]
        lines_in = {owners[i][0] for i in members}
        if is_end[members].any() or len(lines_in) > 1 or len(members) > 1:
            node_of.update({i: len(nodes) for i in members})
            nodes.append(points[sorted(members)].mean(axis=0))
    nodes = np.asarray(nodes).reshape(-1, 2)
    chains, offset = [], 0
    for xy, road_type in clipped:
        ids = list(range(offset, offset + len(xy)))
        offset += len(xy)
        cut = [k for k, i in enumerate(ids) if i in node_of]
        for start, end in zip(cut[:-1], cut[1:]):
            a, b = node_of[ids[start]], node_of[ids[end]]
            coords = np.vstack([nodes[a], xy[start + 1:end], nodes[b]])
            if a == b and len(coords) < 4:
                continue
            if LineString(coords).length > 0:
                chains.append((a, b, coords, road_type))
    return nodes, chains


# ------------------------------------------------------------ patch graphs

def _simplify_and_space(coords, tolerance, spacing):
    from shapely.geometry import LineString
    if tolerance > 0 and len(coords) > 2:
        coords = np.asarray(LineString(coords).simplify(tolerance, preserve_topology=False).coords)
    out = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        steps = max(1, int(math.ceil(np.hypot(*(b - a)) / spacing - 1e-9)))
        for k in range(1, steps + 1):
            out.append(a + (b - a) * k / steps)
    return np.asarray(out)


def patch_graph(nodes, chains, tree, geometries, origin, settings, widths):
    """Clip the map graph to one patch; returns the graph document or raises if too dense."""
    from shapely.geometry import box
    size = settings['patch_size']
    ox, oy = origin
    window = box(ox, oy, ox + size - 1, oy + size - 1)
    candidates = sorted(int(i) for i in tree.query(window, predicate='intersects')) if geometries else []
    parts = []
    for index in candidates:
        a, b, coords, road_type = chains[index]
        clipped = geometries[index].intersection(window)
        pieces = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]
        for piece in pieces:
            if piece.geom_type != 'LineString' or piece.length <= 0:
                continue
            xy = np.asarray(piece.coords, float)
            ends = []
            for end in (xy[0], xy[-1]):
                if np.hypot(*(end - nodes[a])) < 1e-6:
                    ends.append(('node', a))
                elif np.hypot(*(end - nodes[b])) < 1e-6:
                    ends.append(('node', b))
                else:
                    ends.append(('cut', index, round(float(end[0]), 6), round(float(end[1]), 6)))
            parts.append((xy - [ox, oy], ends, road_type))
    spacing = float(settings['node_spacing'])
    for attempt in range(25):
        document = _assemble(parts, settings, widths, spacing)
        if len(document['nodes']) <= settings['max_nodes']:
            document['metadata']['node_spacing_px'] = spacing
            return document
        spacing *= 1.25
    raise ValueError(f'Patch at {origin} needs more than {settings["max_nodes"]} nodes even at spacing {spacing:.1f}px')


def _assemble(parts, settings, widths, spacing):
    decimals, size = settings['coordinate_decimals'], settings['patch_size']
    positions, keys, edges = [], {}, {}

    def node(key, xy):
        xy = np.clip(np.round(np.asarray(xy, float), decimals), 0, size - 1)
        location = (float(xy[0]), float(xy[1]))
        if key is not None and key in keys:
            return keys[key]
        if location in keys:  # rounding can make two nodes coincide: they become one
            index = keys[location]
        else:
            index = len(positions)
            positions.append(location)
            keys[location] = index
        if key is not None:
            keys[key] = index
        return index

    for xy, ends, road_type in parts:
        line = _simplify_and_space(xy, settings['simplify_tolerance'], spacing)
        ids = [node(ends[0], line[0])] + [node(None, p) for p in line[1:-1]] + [node(ends[1], line[-1])]
        width = widths[ROAD_TYPES[road_type]]
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            current = edges.get(key)
            # Overlapping roads keep the wider symbol; equal widths keep the lower type code.
            if current is None or (width, -road_type) > (current[1], -current[0]):
                edges[key] = (road_type, width)
    used = sorted({i for edge in edges for i in edge}, key=lambda i: (positions[i][1], positions[i][0]))
    remap = {old: new for new, old in enumerate(used)}
    canonical = sorted((min(remap[a], remap[b]), max(remap[a], remap[b]), road_type, width)
                       for (a, b), (road_type, width) in edges.items())
    return {'nodes': [list(positions[i]) for i in used], 'edges': [[a, b] for a, b, _, _ in canonical],
            'edge_types': [ROAD_TYPES[t] for _, _, t, _ in canonical], 'edge_widths': [w for _, _, _, w in canonical],
            'metadata': {}}


def road_mask(document, size):
    """Line-width road mask drawn from the patch graph only (so it is reproducible from the JSON)."""
    image = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(image)
    nodes = [tuple(p) for p in document['nodes']]
    joints = defaultdict(int)
    for (a, b), width in zip(document['edges'], document['edge_widths']):
        draw.line([nodes[a], nodes[b]], fill=255, width=int(width))
        joints[a], joints[b] = max(joints[a], width), max(joints[b], width)
    for index, width in sorted(joints.items()):
        if width > 2:
            x, y = nodes[index]
            r = (width - 1) / 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(image)


# -------------------------------------------------------------- raster I/O

def palette(src):
    try:
        colormap = src.colormap(1)
    except ValueError:
        return None
    table = np.zeros((256, 3), np.uint8)
    for index, color in colormap.items():
        if 0 <= index < 256:
            table[index] = color[:3]
    return table


def read_patch(src, lut, origin, size, downsample):
    from rasterio.windows import Window
    ox, oy = origin
    window = Window(ox * downsample, oy * downsample, size * downsample, size * downsample)
    if src.count >= 3:
        data = src.read([1, 2, 3], window=window)
    else:
        data = src.read(1, window=window)
        data = lut[data].transpose(2, 0, 1) if lut is not None else np.repeat(data[None], 3, 0)
    if data.dtype != np.uint8:
        raise ValueError(f'Expected 8-bit map rasters, found {data.dtype}')
    if downsample > 1:
        data = np.rint(data.reshape(3, size, downsample, size, downsample).astype(np.float64).mean((2, 4))).astype(np.uint8)
    return np.ascontiguousarray(data.transpose(1, 2, 0))


def frame_mask(frame, origin, size):
    image = Image.new('L', (size, size), 0)
    polygon = [(x - origin[0], y - origin[1]) for x, y in np.asarray(frame.exterior.coords)]
    ImageDraw.Draw(image).polygon(polygon, fill=255)
    return np.asarray(image) > 0


# ---------------------------------------------------------------- the build

def build_map(spec, entry, cache, output, preview=False, log=print):
    import rasterio
    from shapely.geometry import LineString, box
    from shapely.strtree import STRtree
    settings, widths = spec.settings, spec.widths
    size, stride, downsample = settings['patch_size'], settings['stride'], settings['downsample']
    map_id = entry['map_id']
    raster_path = raster_file(spec, entry, cache)
    annotation_path = source_path(spec, entry, 'annotation', cache)
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f'{map_id}: raster has no CRS')
        raster_crs = src.crs.to_wkt()
        grid = Grid(src.transform, src.width, src.height, downsample)
        frame, frame_transform = neatline_polygon(entry, grid, raster_crs, settings['allow_ballpark_transforms'])
        lines, annotation_transform = read_annotation(annotation_path, entry, spec, raster_crs, grid)
        nodes, chains = map_graph(lines, frame, settings['snap_tolerance'])
        minx, miny, maxx, maxy = frame.bounds
        x0, y0 = max(0, math.floor(minx)), max(0, math.floor(miny))
        x1, y1 = min(grid.width, math.ceil(maxx) + 1), min(grid.height, math.ceil(maxy) + 1)
        geometries = [LineString(coords) for _, _, coords, _ in chains]
        tree = STRtree(geometries) if geometries else None
        lut = palette(src) if src.count < 3 else None
        samples, stats = [], defaultdict(float)
        for oy in range(y0, y1, stride):
            for ox in range(x0, x1, stride):
                if ox + size > grid.width or oy + size > grid.height:
                    stats['tiles_outside_raster'] += 1
                    continue
                coverage = frame.intersection(box(ox - .5, oy - .5, ox + size - .5, oy + size - .5)).area / size ** 2
                if coverage < settings['min_frame_coverage']:
                    stats['tiles_low_coverage'] += 1
                    continue
                document = patch_graph(nodes, chains, tree, geometries, (ox, oy), settings, widths)
                if not document['nodes'] and not settings['keep_empty_patches']:
                    stats['tiles_empty_dropped'] += 1
                    continue
                origin = [ox - x0, oy - y0]
                sample_id = f'{map_id}__x{origin[0]:05d}_y{origin[1]:05d}'
                document['metadata'].update({'map_id': map_id, 'origin': origin, 'frame_coverage': round(coverage, 6),
                                             'producer': 'trasa.build_dataset'})
                rgb = read_patch(src, lut, (ox, oy), size, downsample)
                if settings['outside_frame'] == 'white' and coverage < 1:
                    rgb = rgb.copy()
                    rgb[~frame_mask(frame, (ox, oy), size)] = 255
                relative = Path(map_id) / f'{sample_id}.png'
                write_png(output / 'images' / relative, rgb)
                write_png(output / 'masks' / relative, road_mask(document, size))
                write_json(output / 'graphs' / map_id / f'{sample_id}.json', document)
                samples.append({'id': sample_id, 'map_id': map_id, 'split': 'train', 'origin': origin,
                                'image': f'images/{relative.as_posix()}', 'road_mask': f'masks/{relative.as_posix()}',
                                'graph': f'graphs/{map_id}/{sample_id}.json'})
                stats['patches'] += 1
                stats['empty_patches'] += not document['nodes']
                stats['patch_nodes'] += len(document['nodes'])
                stats['patch_edges'] += len(document['edges'])
                stats['spacing_increased'] += document['metadata']['node_spacing_px'] > settings['node_spacing']
        if preview:
            write_preview(src, frame, chains, output / 'previews' / f'{map_id}.png', downsample)
    reference = map_reference(nodes, chains, (x0, y0))
    reference['metadata'] = {'map_id': map_id, 'coordinate_space': 'pixel', 'grid_origin': [x0, y0]}
    write_json(output / 'references' / f'{map_id}.json', reference)
    lengths = defaultdict(float)
    for _, _, coords, road_type in chains:
        lengths[ROAD_TYPES[road_type]] += float(LineString(coords).length)
    metadata = {'width': x1 - x0, 'height': y1 - y0, 'map_year': entry.get('year'), 'title': entry.get('title'),
                'series': entry.get('series'), 'scan_id': entry.get('scan_id'), 'crs': raster_crs,
                'affine': grid.gdal_affine(x0, y0), 'downsample': downsample,
                'raster_sha256': entry['raster']['sha256'], 'annotation_sha256': entry['annotation']['sha256'],
                'raster_url': entry['raster'].get('url')}
    report = {'patches': int(stats['patches']), 'empty_patches': int(stats['empty_patches']),
              'tiles_low_coverage': int(stats['tiles_low_coverage']), 'tiles_outside_raster': int(stats['tiles_outside_raster']),
              'tiles_empty_dropped': int(stats['tiles_empty_dropped']),
              'mean_patch_nodes': stats['patch_nodes'] / max(stats['patches'], 1),
              'patches_with_wider_spacing': int(stats['spacing_increased']),
              'map_nodes': len(nodes), 'map_chains': len(chains),
              'road_length_px': {k: round(v, 3) for k, v in sorted(lengths.items())},
              'neatline_transform': frame_transform, 'annotation_transform': annotation_transform,
              'grid_origin': [x0, y0]}
    log(f"{map_id}: {report['patches']} patches ({report['empty_patches']} empty), {len(chains)} road chains")
    return samples, metadata, report


def map_reference(nodes, chains, origin):
    """Complete-map reference graph (all polyline vertices) in patch-grid pixels."""
    positions, index, edges, classes = [], {}, [], []

    def key(xy):
        point = (round(float(xy[0] - origin[0]), 3), round(float(xy[1] - origin[1]), 3))
        if point not in index:
            index[point] = len(positions)
            positions.append(point)
        return index[point]
    for a, b, coords, road_type in chains:
        ids = [key(p) for p in coords]
        for u, v in zip(ids[:-1], ids[1:]):
            if u != v:
                edges.append(sorted((u, v)))
                classes.append(road_type)
    unique = {}
    for edge, value in zip(edges, classes):
        unique.setdefault(tuple(edge), value)
    ordered = sorted(unique)
    return {'nodes': [list(p) for p in positions], 'edges': [list(e) for e in ordered],
            'edge_types': [ROAD_TYPES[unique[e]] for e in ordered]}


def write_preview(src, frame, chains, path, downsample, longest=2048):
    from rasterio.enums import Resampling
    factor = max(src.width, src.height) / longest
    width, height = max(1, int(src.width / factor)), max(1, int(src.height / factor))
    if src.count >= 3:
        data = src.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.nearest)
    else:
        band = src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        lut = palette(src)
        data = lut[band].transpose(2, 0, 1) if lut is not None else np.repeat(band[None], 3, 0)
    image = Image.fromarray(np.ascontiguousarray(data.transpose(1, 2, 0)))
    draw = ImageDraw.Draw(image)
    scale = downsample / factor
    to_preview = lambda xy: [((x + .5) * scale, (y + .5) * scale) for x, y in xy]
    draw.line(to_preview(np.asarray(frame.exterior.coords)), fill=OVERLAY, width=3)
    for _, _, coords, _ in chains:
        draw.line(to_preview(coords), fill=OVERLAY, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def checksums(output):
    files = {}
    for path in sorted(p for p in output.rglob('*') if p.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in CHECKSUM_EXCLUDE or relative.startswith(tuple(e for e in CHECKSUM_EXCLUDE if e.endswith('/'))):
            continue
        files[relative] = file_sha256(path)
    combined = hashlib.sha256(''.join(f'{k}\t{v}\n' for k, v in files.items()).encode()).hexdigest()
    return {'schema_version': 1, 'dataset_sha256': combined, 'files': files}


def software_versions():
    import importlib.metadata
    versions = {}
    for name in ('trasa', 'numpy', 'shapely', 'pyproj', 'rasterio', 'fiona', 'Pillow', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import rasterio
        import pyproj
        versions['gdal'] = rasterio.__gdal_version__
        versions['proj'] = pyproj.proj_version_str
    except ImportError:
        pass
    return versions


def build(spec, cache, output, maps=None, preview=False, allow_unlocked=False, log=print):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    missing = spec.missing_hashes()
    if missing and not allow_unlocked:
        raise ValueError(f'Sources without SHA256: {missing}. Run trasa.fetch_sources --lock first.')
    from .fetch_sources import download, fetch
    if not missing:
        fetch(spec, cache, maps, log=log)  # downloads what is missing and checks every SHA256
    else:
        # Development build from unlocked sources: hashes are computed and reported, not checked.
        for entry in spec.maps:
            if maps and entry['map_id'] not in maps:
                continue
            for key in ('raster', 'annotation'):
                path = source_path(spec, entry, key, cache)
                if not path.exists():
                    if not entry[key].get('url'):
                        raise ValueError(f"{entry['map_id']} {key}: missing local file {path}")
                    download(entry[key]['url'], path)
                entry[key]['sha256'] = entry[key].get('sha256') or file_sha256(path)
    output.mkdir(parents=True)
    samples, metadata, reports = [], {}, {}
    for entry in spec.maps:
        if maps and entry['map_id'] not in maps:
            continue
        rows, info, report = build_map(spec, entry, cache, output, preview, log)
        samples.extend(rows)
        metadata[entry['map_id']], reports[entry['map_id']] = info, report
    index = {'schema_version': 1, 'dataset_kind': spec.dataset_kind, 'name': spec.name,
             'description': spec.document.get('description', ''),
             'reference_provenance': spec.document.get('reference_provenance', {}),
             'build': {'spec_sha256': spec.sha256(), 'settings': spec.settings, 'road_widths': spec.widths,
                       'producer': 'trasa.build_dataset', 'coordinates': 'pixels'},
             'maps': metadata, 'samples': samples}
    write_json(output / 'index.json', index)
    if spec.folds and not maps:
        write_json(output / 'fold_schedule.json', spec.folds)
    sums = checksums(output)
    write_json(output / 'checksums.json', sums)
    write_json(output / 'build_report.json', {'maps': reports, 'patches': len(samples), 'software': software_versions(),
                                              'dataset_sha256': sums['dataset_sha256'],
                                              'unlocked_sources': [list(m) for m in missing]})
    log(f"{len(samples)} patches; dataset SHA256 {sums['dataset_sha256']}")
    return index, sums


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True, help='datasets/<name>/dataset.yaml')
    parser.add_argument('--output', required=True, help='New directory for the built dataset')
    parser.add_argument('--cache', default=str(default_cache()), help='Download cache (outside the repository)')
    parser.add_argument('--maps', nargs='*', help='Build only these maps (no fold schedule is written)')
    parser.add_argument('--preview', action='store_true', help='Write previews/<map>.png with the frame and roads (#536FBF)')
    parser.add_argument('--allow-unlocked', action='store_true',
                        help='Development only: build from sources that have no recorded SHA256')
    args = parser.parse_args(argv)
    build(Spec(args.spec), Path(args.cache), Path(args.output), args.maps, args.preview, args.allow_unlocked)


if __name__ == '__main__':
    main()
