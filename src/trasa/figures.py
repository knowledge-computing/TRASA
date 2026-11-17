"""Qualitative comparison figure (paper Fig. 3) from saved map-level graphs.

Rows are patches (map ID + patch origin), columns are the map patch, the
reference graph and each method's prediction. Graphs are cropped from the
complete-map JSON files written by ``predict_dataset`` / ``baseline_evaluate``
and drawn in #536FBF over a lightened copy of the patch.

    python -m trasa.figures --manifest data/benchmark/folds/fold_03.json \\
        --row "Broken continuation" MAP_ID 512 768 --row "Missed intersection" MAP_ID 1024 256 \\
        --reference results/trasa/fold_03/map_00_reference.json \\
        --method TRASA results/trasa/fold_03/map_00_prediction.json \\
        --method LDTR runs/benchmark/ldtr/fold_03/test/map_00_prediction.json \\
        --output figures/qualitative.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .export import read_graph
from .graph import clip_graph

OVERLAY = (83, 111, 191)  # #536FBF
NODE = (230, 120, 0)


def crop(graph, origin, size):
    x, y = origin
    shifted = graph.__class__(graph.nodes - [x, y], graph.edges, graph.types, graph.node_scores, graph.edge_scores,
                              dict(graph.metadata), graph.edge_classes)
    return clip_graph(shifted, (0, 0, size - 1, size - 1))


def draw(image, graph, scale, nodes=True):
    canvas = Image.fromarray((np.asarray(image, float) * .55 + 255 * .45).astype(np.uint8)).resize(
        (image.width * scale, image.height * scale), Image.NEAREST)
    pen = ImageDraw.Draw(canvas)
    points = (graph.nodes + .5) * scale
    for a, b in graph.edges:
        pen.line([tuple(points[a]), tuple(points[b])], fill=OVERLAY, width=max(2, scale + 1))
    if nodes:
        r = max(2, scale)
        for x, y in points:
            pen.ellipse((x - r, y - r, x + r, y + r), fill=NODE)
    return canvas


def figure(manifest, rows, reference_files, methods, output, scale=1, gap=6):
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    size = int(document.get('build', {}).get('settings', {}).get('patch_size', 256))
    by_origin = {(r['map_id'], tuple(r['origin'])): r for r in document['samples']}
    graphs = {}

    def load(path):
        if path not in graphs:
            graphs[path] = read_graph(path)
        return graphs[path]
    columns = ['Map patch', 'Reference'] + [name for name, _ in methods]
    font = ImageFont.load_default()
    cell, label_width, header = size * scale, 170, 24
    sheet = Image.new('RGB', (label_width + len(columns) * (cell + gap), header + len(rows) * (cell + gap)), 'white')
    pen = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        pen.text((label_width + c * (cell + gap) + 4, 6), name, fill='black', font=font)
    for r, (label, map_id, x, y) in enumerate(rows):
        row = by_origin.get((map_id, (x, y)))
        if row is None:
            raise ValueError(f'No patch of {map_id} at origin ({x}, {y}) in the manifest')
        with Image.open(manifest.parent / row['image']) as image:
            patch = image.convert('RGB')
        top = header + r * (cell + gap)
        pen.text((4, top + cell // 2), label, fill='black', font=font)
        sheet.paste(patch.resize((cell, cell), Image.NEAREST), (label_width, top))
        reference = load(reference_files[map_id] if isinstance(reference_files, dict) else reference_files)
        panels = [crop(reference, (x, y), size)] + [crop(load(path), (x, y), size) for _, path in methods]
        for c, graph in enumerate(panels, start=1):
            sheet.paste(draw(patch, graph, scale), (label_width + c * (cell + gap), top))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--row', nargs=4, action='append', required=True, metavar=('LABEL', 'MAP_ID', 'X', 'Y'),
                        help='Error case label, map ID and patch origin in map pixels')
    parser.add_argument('--reference', required=True, help='Map-level reference graph JSON')
    parser.add_argument('--method', nargs=2, action='append', required=True, metavar=('NAME', 'GRAPH_JSON'))
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    rows = [(label, map_id, int(x), int(y)) for label, map_id, x, y in args.row]
    figure(args.manifest, rows, args.reference, [tuple(m) for m in args.method], args.output, args.scale)


if __name__ == '__main__':
    main()
