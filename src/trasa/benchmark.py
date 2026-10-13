"""Run the four graph variants on explicit fold manifests and aggregate map metrics."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VARIANTS = ('ldtr', 'ldtr-sam', 'ldtr-ntc', 'trasa')


def aggregate_reports(paths):
    maps, protocol = [], None
    for path in paths:
        report = json.loads(Path(path).read_text())
        if protocol is not None and report['protocol'] != protocol:
            raise ValueError('Cannot average results with different evaluation protocols')
        protocol = report['protocol']
        maps.extend(report['maps'])
    if not maps:
        raise ValueError('No map evaluation results')
    if len({m['map_id'] for m in maps}) != len(maps):
        raise ValueError('A held-out map must appear exactly once in fold results')
    keys = sorted({k for row in maps for k in row['metrics']})
    means = {key: float(np.mean([row['metrics'][key] for row in maps if row['metrics'].get(key) is not None]))
             if any(row['metrics'].get(key) is not None for row in maps) else None for key in keys}
    return {'map_count': len(maps), 'mean': means, 'maps': maps, 'protocol': protocol}


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
