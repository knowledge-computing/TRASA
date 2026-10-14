"""Regenerate comparison/ablation tables exclusively from evaluation artifacts."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .benchmark import aggregate_reports


MAIN_COLUMNS = ('completeness', 'correctness', 'quality', 'apls', 'intersection_recall', 'intersection_precision',
                'intersection_f1', 'intersection_degree_accuracy', 'connectivity', 'split_errors', 'spurious_branches')
ABLATION_COLUMNS = ('quality', 'apls', 'intersection_f1', 'intersection_degree_accuracy', 'connectivity', 'type_macro_f1')


def regenerate(reports, allow_incomplete=False):
    groups, protocol, dataset_kind, universe = defaultdict(list), None, None, None
    assignments = {}
    for path in reports:
        report = json.loads(Path(path).read_text())
        if report.get('schema_version') != 2 or 'experiment' not in report:
            raise ValueError('Tables require version-2 evaluation reports with experiment provenance')
        identity = report['experiment']
        kind = identity.get('dataset_kind')
        if not kind or kind == 'unverified':
            raise ValueError('Declare benchmark, synthetic or alternative dataset provenance')
        if protocol is not None and report['protocol'] != protocol:
            raise ValueError('Cannot compare different evaluation protocols')
        if dataset_kind is not None and kind != dataset_kind:
            raise ValueError('Cannot mix synthetic and benchmark results')
        partition = {key: sorted(identity['map_partitions'][key]) for key in ('train', 'val', 'test')}
        maps = set().union(*(set(v) for v in partition.values()))
        if [len(partition[key]) for key in ('train', 'val', 'test')] != [6, 1, 1] or len(maps) != 8:
            raise ValueError('Table reports require a valid six/one/one eight-map partition')
        test_id = partition['test'][0]
        if test_id in assignments and assignments[test_id] != partition:
            raise ValueError('Methods must use the same training/validation partition for each test map')
        assignments[test_id] = partition
        if universe is not None and maps != universe:
            raise ValueError('All methods must use the same map collection')
        protocol, dataset_kind, universe = report['protocol'], kind, maps
        groups[identity['variant']].append(Path(path))
    if not groups:
        raise ValueError('No evaluation artifacts supplied')
    result = {}
    expected = None
    for method, paths in sorted(groups.items()):
        summary = aggregate_reports(paths)
        ids = {row['map_id'] for row in summary['maps']}
        if not allow_incomplete and (summary['map_count'] != 8 or ids != universe):
            raise ValueError(f'{method}: eight unique held-out maps required')
        if expected is not None and ids != expected:
            raise ValueError('All methods must report the same held-out maps')
        expected = ids
        result[method] = summary
    return {'dataset_kind': dataset_kind, 'result_origin': 'synthetic_execution' if dataset_kind == 'synthetic' else 'computed_with_documented_protocol',
            'complete_eight_fold': all(v['map_count'] == 8 for v in result.values()),
            'protocol': protocol, 'methods': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-incomplete', action='store_true', help='Label partial/synthetic demonstrations explicitly')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Table output exists; choose a new directory')
    result = regenerate(args.reports, args.allow_incomplete)
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, columns in (('main_results', MAIN_COLUMNS), ('ablations', ABLATION_COLUMNS)):
        rows = []
        for method, summary in result['methods'].items():
            if name == 'ablations' and method not in ('ldtr', 'ldtr-sam', 'ldtr-ntc', 'trasa'):
                continue
            if name == 'main_results' and method in ('ldtr-sam', 'ldtr-ntc'):
                continue
            rows.append([method, *['--' if summary['mean'].get(key) is None else
                        f"{summary['mean'][key]:.6g} ± {summary['std_population'][key]:.6g}" for key in columns]])
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['method', *columns])
            writer.writerows(rows)
        header = '| method | ' + ' | '.join(columns) + ' |\n'
        divider = '|' + '|'.join(['---'] * (len(columns) + 1)) + '|\n'
        body = ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows)
        (output / f'{name}.md').write_text(f"Dataset: {result['dataset_kind']}; origin: {result['result_origin']}.\n\n" + header + divider + body)


if __name__ == '__main__':
    main()
