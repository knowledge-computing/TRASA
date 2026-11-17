"""One command from the published spec to training-ready folds.

Runs, in order: fetch_sources (download + SHA256 check), build_dataset,
verify_dataset (when ``expected_checksums.json`` sits next to the spec),
prepare_data, validate_data and folds. Output layout::

    <output>/built      patches, graphs, masks, references, index.json, checksums.json
    <output>/prepared   manifest.json for training
    <output>/folds      fold_00.json ... fold_07.json
"""
import argparse
import json
from pathlib import Path

from .dataset_spec import Spec, default_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', default='datasets/trasa_ca24/dataset.yaml')
    parser.add_argument('--output', required=True, help='New directory')
    parser.add_argument('--cache', default=str(default_cache()))
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--skip-verify', action='store_true', help='Do not compare with expected_checksums.json')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    spec = Spec(args.spec)
    from .build_dataset import build
    from .folds import main as folds
    from .prepare_data import main as prepare
    from .validate_data import validate_dataset
    from .config import Config
    build(spec, Path(args.cache), output / 'built', preview=args.preview)
    expected = spec.root / 'expected_checksums.json'
    if expected.exists() and not args.skip_verify:
        from .verify_dataset import verify
        report = verify(output / 'built', expected)
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        if not report['valid']:
            raise SystemExit(f'Built dataset differs from {expected}; see {output / "verification.json"}')
        print(f"verified {report['files_checked']} files against {expected}")
    elif not expected.exists():
        print(f'No {expected.name} next to the spec: nothing to compare with (first build?).')
    size = spec.settings['patch_size']
    prepare(['--index', str(output / 'built' / 'index.json'), '--coordinates', 'pixels', '--patch-size', str(size),
             '--max-nodes', str(spec.settings['max_nodes']), '--output', str(output / 'prepared')])
    report = validate_dataset(output / 'prepared' / 'manifest.json', Config(patch_size=size, num_queries=max(100, spec.settings['max_nodes'])),
                              guided=False, paper=spec.dataset_kind == 'benchmark' and size == 256)
    (output / 'data_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['valid']:
        raise SystemExit(f'Prepared data failed validation; see {output / "data_validation.json"}')
    schedule = output / 'built' / 'fold_schedule.json'
    command = ['--manifest', str(output / 'prepared' / 'manifest.json'), '--output', str(output / 'folds')]
    if schedule.exists():
        command += ['--schedule', str(schedule)]
    folds(command)
    print(f'Ready: {output / "folds"}')


if __name__ == '__main__':
    main()
