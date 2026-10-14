"""Run the four graph variants on explicit fold manifests and aggregate map metrics."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np

from .config import Config
from .data import guidance_metadata, manifest_errors


VARIANTS = ('ldtr', 'ldtr-sam', 'ldtr-ntc', 'trasa')


def aggregate_reports(paths):
    maps, protocol, identity = [], None, None
    for path in paths:
        report = json.loads(Path(path).read_text())
        if protocol is not None and report['protocol'] != protocol:
            raise ValueError('Cannot average results with different evaluation protocols')
        protocol = report['protocol']
        if report.get('experiment'):
            current = {key: report['experiment'].get(key) for key in ('variant', 'dataset_kind', 'seed', 'config', 'evaluation_implementation_sha256')}
            geometry = report['experiment'].get('geometry_provenance') or {}
            current['geometry_recipe'] = {key: geometry.get(key) for key in ('sam_variant', 'patch_size', 'use_sam_decoder', 'keypoint_semantics', 'recipe')}
            if identity is not None and identity != current:
                raise ValueError('Cannot average incompatible variants, datasets, seeds, configurations or code')
            identity = current
            if any(row.get('split') != 'test' for row in report['maps']):
                raise ValueError('Only held-out test maps may enter result aggregation')
            declared_test = report['experiment'].get('map_partitions', {}).get('test')
            if declared_test is not None and {row['map_id'] for row in report['maps']} != set(declared_test):
                raise ValueError('Reported map IDs disagree with declared held-out test partition')
            from .runtime import sha256
            maps.extend({**row, 'experiment': report['experiment'], 'report_sha256': sha256(path)} for row in report['maps'])
        else:
            maps.extend(report['maps'])
    if not maps:
        raise ValueError('No map evaluation results')
    if len({m['map_id'] for m in maps}) != len(maps):
        raise ValueError('A held-out map must appear exactly once in fold results')
    keys = sorted({k for row in maps for k in row['metrics']})
    means, deviations, counts = {}, {}, {}
    for key in keys:
        values = [row['metrics'][key] for row in maps if row['metrics'].get(key) is not None]
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
            raise ValueError(f'Nonfinite or nonscalar metric: {key}')
        counts[key] = len(values)
        means[key] = float(np.mean(values)) if values else None
        deviations[key] = float(np.std(values, ddof=0)) if values else None
    return {'map_count': len(maps), 'mean': means, 'std_population': deviations, 'eligible_maps': counts,
            'maps': maps, 'protocol': protocol, 'identity': identity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-manifests', nargs='+', required=True, help='Prepared fold-specific cached-guidance manifests')
    parser.add_argument('--config', default='configs/paper.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    manifests = [Path(p).resolve() for p in args.fold_manifests]
    held_out, benchmark_maps = [], None
    for manifest in manifests:
        doc = json.loads(manifest.read_text())
        errors = manifest_errors(doc, manifest.parent)
        if errors:
            raise ValueError('\n'.join(errors))
        if not args.dry_run and any(v in ('trasa', 'ldtr-sam') for v in args.variants):
            provenance = guidance_metadata(doc, replace(Config.load(args.config), guidance=True))
            if provenance.get('cache_schema_version') != 2 or 'validation_map_ids' not in provenance:
                raise ValueError('Benchmark requires bound caches and checkpoint-selection provenance')
        train = {r['map_id'] for r in doc['samples'] if r['split'] == 'train'}
        val = {r['map_id'] for r in doc['samples'] if r['split'] == 'val'}
        test = {r['map_id'] for r in doc['samples'] if r['split'] == 'test'}
        if len(train) != 6 or len(val) != 1 or len(test) != 1 or train & (val | test) or val & test:
            raise ValueError('Paper fold requires six training maps and one distinct validation/test map')
        fold_maps = train | val | test
        if benchmark_maps is not None and fold_maps != benchmark_maps:
            raise ValueError('All folds must partition the same eight benchmark maps')
        benchmark_maps = fold_maps
        held_out.extend(test)
    if len(manifests) != 8 or len(set(held_out)) != 8:
        raise ValueError('Full benchmark requires eight folds with unique held-out maps')
    output = Path(args.output).resolve()
    if output.exists() and not args.dry_run:
        raise ValueError('Output exists; choose a new benchmark directory')
    for variant in args.variants:
        reports = []
        for number, manifest in enumerate(manifests):
            run = output / variant / f'fold_{number:02d}'
            evaluation = run / 'test'
            commands = [
                [sys.executable, '-m', 'trasa.train', '--config', str(Path(args.config).resolve()),
                 '--manifest', str(manifest), '--output', str(run), '--device', args.device, '--ablation', variant],
                [sys.executable, '-m', 'trasa.predict_dataset', '--manifest', str(manifest),
                 '--checkpoint', str(run / 'best.pt'), '--output', str(evaluation), '--device', args.device]]
            for command in commands:
                if args.dry_run:
                    print(json.dumps(command))
                else:
                    subprocess.run(command, check=True)
            reports.append(evaluation / 'metrics.json')
        if not args.dry_run:
            (output / variant / 'summary.json').write_text(json.dumps(aggregate_reports(reports), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
