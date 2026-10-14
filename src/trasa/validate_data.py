"""Validate every selected sample and report all failures before training."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import Config
from .data import guidance_metadata, load_sample, manifest_errors, read_road_mask


def validate_dataset(manifest, config=None, splits=None, guided=None, paper=False):
    path = Path(manifest).resolve()
    config = config or Config()
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        return {'valid': False, 'checked_samples': 0, 'map_count': 0, 'errors': [f'Malformed manifest JSON: {error}'], 'invalid_samples': []}
    if not isinstance(document, dict):
        return {'valid': False, 'checked_samples': 0, 'map_count': 0, 'errors': ['Manifest must be an object'], 'invalid_samples': []}
    rows = document.get('samples') if isinstance(document.get('samples'), list) else []
    if guided is None:
        guided = bool(document.get('guidance_provenance') or any(r.get('guidance') for r in rows if isinstance(r, dict)))
    config = replace(config, guidance=guided)
    errors = manifest_errors(document, path.parent)
    provenance = None
    if guided and isinstance(document.get('samples'), list):
        try:
            provenance = guidance_metadata(document, config)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            errors.append(str(error))
    maps = {r['map_id'] for r in rows if isinstance(r, dict) and isinstance(r.get('map_id'), str)}
    if paper:
        if len(maps) != 8 or config.patch_size != 256 or config.num_queries != 100:
            errors.append('Paper validation requires eight maps, 256-pixel patches and 100 queries')
        if document.get('dataset_kind') != 'benchmark':
            errors.append('Paper benchmark must explicitly declare dataset_kind=benchmark')
        reference = document.get('reference_provenance')
        if not isinstance(reference, dict) or reference.get('historical_correction') != 'manually_corrected':
            errors.append('Declare manually corrected historical references; uncorrected modern vectors are insufficient')
    invalid, checked = [], 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or (splits and row.get('split') not in splits):
            continue
        checked += 1
        try:
            active = replace(config, require_bound_guidance=config.require_bound_guidance and provenance is not None)
            load_sample(row, path.parent, active, guided, provenance)
            if row.get('road_mask'):
                read_road_mask(path.parent / row['road_mask'], config.patch_size)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            invalid.append({'sample_id': row.get('id', index), 'error': str(error)})
    return {'valid': not errors and not invalid, 'dataset_kind': document.get('dataset_kind', 'unverified'),
            'checked_samples': checked, 'map_count': len(maps), 'errors': errors, 'invalid_samples': invalid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--without-guidance', action='store_true')
    parser.add_argument('--paper', action='store_true', help='Require benchmark provenance and paper data dimensions')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    report = validate_dataset(args.manifest, Config.load(args.config), guided=False if args.without_guidance else None, paper=args.paper)
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise ValueError('Validation output exists; choose a new file')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
    print(content, end='')
    if not report['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
