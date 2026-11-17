"""GPT-5.5 one-shot road-polyline baseline (paper Sec. 5.1, Table 3).

For every held-out patch and every road type the model gets: one fixed example
patch with its annotation (from a training/validation map, never the test map),
then the query patch and the prompt from Table 3, and must answer
``{"polylines": [[[x, y], ...], ...]}``. Settings follow the paper: model
``gpt-5.5``, temperature 0, top-p 1, at most 4096 output tokens, structured JSON
output, one sample per patch.

Every request and raw answer is saved, so the baseline can be re-scored without
calling the API again. Answers are rasterized with road-type line widths, turned
into graphs with the shared mask-to-graph step, and written as an index for
``trasa.baseline_evaluate``. Needs ``pip install openai`` and OPENAI_API_KEY.
"""
import argparse
import base64
import datetime
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .baselines import mask_to_graph, typed_polylines_to_mask
from .export import write_graph
from .road_types import DEFAULT_WIDTHS, DISPLAY_NAMES, ROAD_TYPES, check_widths

PROMPT = ('Given this {size} × {size} map patch and one example annotation, identify all {road_type} roads. '
          'Return each road as an ordered polyline in image coordinates using JSON format: '
          '{{"polylines": [[[x1,y1],...,[xn,yn]], ...]}}.')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['polylines'],
          'properties': {'polylines': {'type': 'array', 'items': {'type': 'array', 'items': {
              'type': 'array', 'items': {'type': 'number'}}}}}}


def graph_polylines(document, road_type):
    """Split the edges of one road type into ordered polylines (maximal paths)."""
    nodes = np.asarray(document['nodes'], float).reshape(-1, 2)
    edges = [tuple(e) for e, t in zip(document['edges'], document.get('edge_types', [])) if t == road_type]
    neighbors = defaultdict(list)
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    used, lines = set(), []

    def walk(start, nxt):
        path = [start, nxt]
        used.add(frozenset((start, nxt)))
        while len(neighbors[path[-1]]) == 2:
            following = [n for n in neighbors[path[-1]] if frozenset((path[-1], n)) not in used]
            if not following:
                break
            used.add(frozenset((path[-1], following[0])))
            path.append(following[0])
        return path
    starts = sorted(neighbors, key=lambda n: (len(neighbors[n]) == 2, n))
    for start in starts:
        for nxt in sorted(neighbors[start]):
            if frozenset((start, nxt)) not in used:
                lines.append(walk(start, nxt))
    return [[[round(float(nodes[i][0]), 1), round(float(nodes[i][1]), 1)] for i in line] for line in lines]


def image_url(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'data:image/png;base64,{data}'


def request(model, example_image, example_polylines, query_image, road_type, size, settings):
    name = DISPLAY_NAMES[road_type]
    prompt = PROMPT.format(size=size, road_type=name)
    content = [
        {'type': 'input_text', 'text': f'Example map patch. Its {name} roads are annotated below.'},
        {'type': 'input_image', 'image_url': example_image},
        {'type': 'input_text', 'text': 'Example annotation: ' + json.dumps({'polylines': example_polylines})},
        {'type': 'input_text', 'text': prompt},
        {'type': 'input_image', 'image_url': query_image},
    ]
    body = {'model': model, 'input': [{'role': 'user', 'content': content}],
            'max_output_tokens': settings['max_output_tokens'],
            'text': {'format': {'type': 'json_schema', 'name': 'road_polylines', 'schema': SCHEMA, 'strict': True}}}
    if settings['sampling_parameters']:
        body.update({'temperature': settings['temperature'], 'top_p': settings['top_p']})
    return body, prompt


def parse(text, size):
    try:
        document = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None, 'malformed_json'
    if not isinstance(document, dict) or not isinstance(document.get('polylines'), list):
        return None, 'malformed_schema'
    lines = []
    for line in document['polylines']:
        try:
            points = np.asarray(line, float)
            if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.isfinite(points).all():
                lines.append(np.clip(points, 0, size - 1).round(3).tolist())
        except (TypeError, ValueError):
            continue
    return lines, 'ok'


def call(client, body, attempts=5):
    for attempt in range(1, attempts + 1):
        try:
            response = client.responses.create(**body)
            raw = response.model_dump() if hasattr(response, 'model_dump') else response
            text = getattr(response, 'output_text', None)
            if text is None and isinstance(raw, dict):
                text = raw.get('output_text')
            return text, raw
        except Exception as error:  # network and rate-limit errors from the client library
            if attempt == attempts:
                return None, {'error': repr(error)}
            time.sleep(min(60, 2 ** attempt))


def run(manifest, split, example_id, output, model='gpt-5.5', client=None, dry_run=False, limit=None,
        settings=None, road_types=ROAD_TYPES, log=print):
    settings = {'temperature': 0, 'top_p': 1, 'max_output_tokens': 4096, 'sampling_parameters': True, **(settings or {})}
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    rows = {row['id']: row for row in document['samples']}
    if example_id not in rows:
        raise ValueError('Example patch not in the manifest')
    example = rows[example_id]
    if example['split'] == 'test':
        raise ValueError('The one-shot example must come from a training or validation map, never the test map')
    targets = [row for row in document['samples'] if row['split'] == split]
    if limit:
        targets = targets[:limit]
    widths = check_widths({**DEFAULT_WIDTHS, **document.get('build', {}).get('road_widths', {})})
    size = int(document.get('build', {}).get('settings', {}).get('patch_size', 256))
    example_graph = json.loads((manifest.parent / example['graph']).read_text())
    if 'edge_types' not in example_graph:
        raise ValueError('The example patch needs road-type annotations (edge_types)')
    example_lines = {t: graph_polylines(example_graph, t) for t in road_types}
    output = Path(output)
    (output / 'responses').mkdir(parents=True, exist_ok=True)
    if client is None and not dry_run:
        from openai import OpenAI
        client = OpenAI()
    example_image = image_url(manifest.parent / example['image'])
    samples, malformed = [], 0
    for number, row in enumerate(targets):
        record_path = output / 'responses' / f"{row['id']}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
        else:
            record = {'sample_id': row['id'], 'map_id': row['map_id'], 'responses': []}
            query_image = image_url(manifest.parent / row['image'])
            for road_type in road_types:
                body, prompt = request(model, example_image, example_lines[road_type], query_image, road_type, size, settings)
                digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                item = {'road_type': road_type, 'prompt': prompt, 'request_sha256': digest}
                if dry_run:
                    item.update({'status': 'dry_run', 'polylines': None})
                else:
                    text, raw = call(client, body)
                    lines, status = parse(text, size)
                    item.update({'status': status, 'polylines': lines, 'output_text': text, 'raw_response': raw})
                record['responses'].append(item)
            if not dry_run:
                record_path.write_text(json.dumps(record, default=str) + '\n')
        mask, discarded, bad = typed_polylines_to_mask(record, widths, size) if not dry_run else (np.zeros((size, size)), 0, 0)
        malformed += bad
        graph = mask_to_graph(mask)
        graph.metadata.update({'coordinate_space': 'pixel', 'external_method': 'gpt5.5', 'discarded_polylines': discarded,
                               'malformed_responses': bad})
        name = f"graphs/{row['id']}.json"
        if (output / name).exists():
            (output / name).unlink()
        write_graph(graph, output / name)
        samples.append({'id': row['id'], 'graph': name})
        if number % 50 == 0:
            log(f'{number + 1}/{len(targets)} patches')
    index = {'training_map_ids': [], 'validation_map_ids': [], 'example_map_ids': [example['map_id']],
             'example_sample_id': example_id, 'source_revision': model, 'checkpoint_sha256': None,
             'access_date': datetime.date.today().isoformat(),
             'prompt_settings': {'template': PROMPT, 'road_types': list(road_types), 'one_shot_example': example_id,
                                 **settings, 'structured_output_schema': SCHEMA},
             'configuration': {'rasterization_widths': widths, 'mask_threshold': .5, 'min_component': 20, 'min_spur': 8,
                               'malformed_responses': malformed, 'dry_run': dry_run},
             'samples': samples}
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True, help='Fold manifest; test patches are queried')
    parser.add_argument('--example', required=True, help='Sample ID of the fixed one-shot example (train/val map)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--model', default='gpt-5.5')
    parser.add_argument('--road-types', nargs='+', default=list(ROAD_TYPES), choices=ROAD_TYPES)
    parser.add_argument('--no-sampling-parameters', action='store_true',
                        help='Omit temperature/top_p if the model rejects them (recorded in the index)')
    parser.add_argument('--limit', type=int, help='Only the first N patches (smoke test)')
    parser.add_argument('--dry-run', action='store_true', help='Build requests, call nothing')
    args = parser.parse_args(argv)
    run(args.manifest, args.split, args.example, args.output, args.model, dry_run=args.dry_run, limit=args.limit,
        settings={'sampling_parameters': not args.no_sampling_parameters}, road_types=tuple(args.road_types))


if __name__ == '__main__':
    main()
