"""SAM3 text-prompt baseline (paper Sec. 5.1, Table 3).

Each held-out patch is segmented once per road type with the text prompt
``"[road type] roads"``. Instance masks are combined into one road probability
map (each pixel keeps the highest instance score covering it), which goes
through the shared mask-to-graph step (threshold 0.5, components >= 20 px,
skeleton, spurs >= 8 px). Probability maps are saved, so post-processing can be
re-run without the model.

Needs the facebookresearch/sam3 package and its (gated) weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .baselines import mask_to_graph
from .export import write_graph
from .road_types import DISPLAY_NAMES, ROAD_TYPES

TEMPLATE = '{road_type} roads'


def sam3_predictor(device='cuda', confidence=0.5):
    """predict(image, prompt) -> (masks [N,H,W] bool, scores [N]) using the official SAM3 processor."""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model().to(device).eval()
    try:
        processor = Sam3Processor(model, confidence_threshold=confidence)
    except TypeError:
        processor = Sam3Processor(model)

    @torch.no_grad()
    def predict(image, prompt):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks, scores = output.get('masks'), output.get('scores')
        masks = np.zeros((0, *image.size[::-1]), bool) if masks is None else masks.detach().float().cpu().numpy()
        scores = np.zeros(len(masks)) if scores is None else scores.detach().float().cpu().numpy().reshape(-1)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks > .5, scores
    return predict


def probability_map(predict, image, road_types, template=TEMPLATE):
    size = image.size[::-1]
    combined, instances = np.zeros(size, np.float32), {}
    for road_type in road_types:
        masks, scores = predict(image, template.format(road_type=DISPLAY_NAMES[road_type]))
        masks, scores = np.asarray(masks, bool).reshape(-1, *size), np.asarray(scores, float).reshape(-1)
        if len(masks) != len(scores):
            raise ValueError('SAM3 returned a different number of masks and scores')
        for mask, score in zip(masks, scores):
            combined = np.maximum(combined, np.where(mask, np.float32(np.clip(score, 0, 1)), 0))
        instances[road_type] = int(len(masks))
    return combined, instances


def run(manifest, output, split='test', predict=None, road_types=ROAD_TYPES, template=TEMPLATE, revision=None,
        checkpoint_sha256=None, limit=None, log=print):
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    targets = [row for row in document['samples'] if row['split'] == split][:limit]
    output = Path(output)
    (output / 'probabilities').mkdir(parents=True, exist_ok=True)
    samples = []
    for number, row in enumerate(targets):
        cache = output / 'probabilities' / f"{row['id']}.npz"
        if cache.exists():
            with np.load(cache) as stored:
                probabilities = stored['road']
        else:
            with Image.open(manifest.parent / row['image']) as image:
                probabilities, instances = probability_map(predict, image.convert('RGB'), road_types, template)
            np.savez_compressed(cache, road=probabilities, instances=json.dumps(instances))
        graph = mask_to_graph(probabilities)
        graph.metadata.update({'coordinate_space': 'pixel', 'external_method': 'sam3'})
        name = output / 'graphs' / f"{row['id']}.json"
        if name.exists():
            name.unlink()
        write_graph(graph, name)
        samples.append({'id': row['id'], 'graph': f"graphs/{row['id']}.json"})
        if number % 100 == 0:
            log(f'{number + 1}/{len(targets)} patches')
    index = {'training_map_ids': [], 'validation_map_ids': [], 'example_map_ids': [],
             'source_revision': revision, 'checkpoint_sha256': checkpoint_sha256,
             'prompt_settings': {'template': template, 'road_types': list(road_types), 'prompt_kind': 'text'},
             'configuration': {'combination': 'max_instance_score', 'mask_threshold': .5, 'min_component': 20, 'min_spur': 8},
             'samples': samples}
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--confidence', type=float, default=.5, help='SAM3 instance confidence threshold')
    parser.add_argument('--revision', default='0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc', help='sam3 source revision used')
    parser.add_argument('--checkpoint-sha256', help='SHA256 of the SAM3 weights file')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    run(args.manifest, args.output, args.split, sam3_predictor(args.device, args.confidence), revision=args.revision,
        checkpoint_sha256=args.checkpoint_sha256, limit=args.limit)


if __name__ == '__main__':
    main()
