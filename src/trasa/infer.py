"""Patch or tiled-map inference; no reference annotation is required."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import SAMRoadGeometry
from .graph import clip_graph, decode_graphs, georeference_graph, merge_patch_graphs, ownership_boxes
from .runtime import load_model, sha256
from .export import write_graph


def tile_origins(width, height, size, stride):
    if not 1 <= stride <= size:
        raise ValueError('Stride must be positive and cannot exceed patch size')
    def axis(length):
        if length < 1:
            raise ValueError('Image dimensions must be positive')
        final = int(np.ceil(max(length - size, 0) / stride)) * stride
        return list(range(0, final + 1, stride))
    return [(x, y) for y in axis(height) for x in axis(width)]


@torch.no_grad()
def infer_image(model, rgb, stride=None):
    size = model.config.patch_size
    stride = size if stride is None else stride
    height, width = rgb.shape[:2]
    origins = tile_origins(width, height, size, stride)
    boxes = ownership_boxes(origins, size, (height, width))
    graphs = []
    model.eval()
    device = next(model.parameters()).device
    for (x, y), bounds in zip(origins, boxes):
        patch = np.zeros((size, size, 3), dtype=np.float32)
        crop = rgb[y:y + size, x:x + size]
        patch[:len(crop), :crop.shape[1]] = crop
        tensor = torch.from_numpy(patch).permute(2, 0, 1)[None].to(device)
        graph = decode_graphs(model, model(tensor))[0]
        graphs.append(clip_graph(graph, bounds))
    return merge_patch_graphs(graphs, origins, size, model.config.merge_tolerance, model.config.boundary_band, boxes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--guidance', help='NPZ with road and intersection probabilities, for one full-size patch')
    parser.add_argument('--geometry-checkpoint', help='Required for guided map inference from a cached-guidance-trained model')
    parser.add_argument('--sam-variant', default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--sam-decoder', action='store_true')
    parser.add_argument('--stride', type=int, help='Tile stride; defaults to checkpoint patch size')
    parser.add_argument('--affine', type=float, nargs=6, help='GDAL geotransform x0,a,b,y0,d,e')
    parser.add_argument('--crs')
    parser.add_argument('--read-georeferencing', action='store_true', help='Read GeoTIFF CRS/affine with the optional GIS environment')
    args = parser.parse_args(argv)
    if args.read_georeferencing:
        if args.affine is not None or args.crs:
            raise ValueError('Choose source georeferencing or explicit affine/CRS')
        try:
            import rasterio
        except ImportError as error:
            raise RuntimeError("Reading GeoTIFF georeferencing requires pip install -e '.[gis]'") from error
        with rasterio.open(args.image) as raster:
            if raster.crs is None:
                raise ValueError('Input raster has no CRS; provide a verified spatial reference')
            args.affine, args.crs = raster.transform.to_gdal(), raster.crs.to_string()
    if Path(args.output).exists():
        raise ValueError('Output already exists; choose a new file')
    model, state = load_model(args.checkpoint, args.device)
    if args.geometry_checkpoint:
        expected = (state['provenance'].get('guidance_provenance') or {}).get('checkpoint_sha256')
        if sha256(args.geometry_checkpoint) != expected:
            raise ValueError('Geometry checkpoint differs from training guidance provenance')
        model.geometry = SAMRoadGeometry(model.config.patch_size, args.sam_variant, args.sam_decoder).load_checkpoint(args.geometry_checkpoint).to(args.device)
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255
    model.eval()
    if args.guidance:
        if rgb.shape[:2] != (model.config.patch_size, model.config.patch_size):
            raise ValueError('Cached guidance inference accepts one patch only')
        with np.load(args.guidance, allow_pickle=False) as cache:
            if model.config.require_bound_guidance:
                provenance = state['provenance'].get('guidance_provenance') or {}
                expected = {'checkpoint_sha256': provenance.get('checkpoint_sha256'),
                            'keypoint_semantics': provenance.get('keypoint_semantics'),
                            'image_sha256': sha256(args.image)}
                for key, value in expected.items():
                    if value is None or key not in cache or str(cache[key].item()) != value:
                        raise ValueError(f'Inference guidance binding mismatch: {key}')
            guidance = torch.from_numpy(np.stack((cache['road'], cache['intersection'])).astype(np.float32))[None].to(args.device)
            if model.config.require_bound_guidance and guidance.shape != (1, 2, model.config.patch_size, model.config.patch_size):
                raise ValueError('Bound inference guidance must have the checkpoint patch resolution; regenerate the cache')
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device)
        graph = decode_graphs(model, model(tensor, guidance))[0]
    else:
        graph = infer_image(model, rgb, args.stride)
    graph.metadata.update({'image_sha256': sha256(args.image), 'model_sha256': sha256(args.checkpoint),
                           'coordinate_space': 'pixel', 'node_type_schema': ['no-node', 'endpoint', 'intermediate', 'intersection']})
    if args.affine is not None:
        graph = georeference_graph(graph, args.affine, args.crs)
        graph.metadata['coordinate_space'] = 'geospatial'
    elif args.crs:
        raise ValueError('A CRS requires an affine geotransform')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_graph(graph, path)


if __name__ == '__main__':
    main()
