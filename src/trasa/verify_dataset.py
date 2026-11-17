"""Check a built dataset against published checksums, file by file.

Mismatches are grouped (images / graphs / masks / references / index) so it is
clear whether only decoded pixels differ (e.g. another GDAL/libjpeg build) or
the road labels themselves differ.
"""
import argparse
import json
import sys
from pathlib import Path

from .build_dataset import checksums


def category(path):
    head = path.split('/', 1)[0]
    return head if '/' in path else Path(path).stem


def verify(dataset, expected):
    dataset = Path(dataset)
    expected = json.loads(Path(expected).read_text())
    actual = checksums(dataset)
    want, have = expected['files'], actual['files']
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatched = {}
    for path in sorted(set(want) & set(have)):
        if want[path] != have[path]:
            mismatched.setdefault(category(path), []).append(path)
    return {'valid': not missing and not extra and not mismatched,
            'dataset_sha256': actual['dataset_sha256'], 'expected_dataset_sha256': expected['dataset_sha256'],
            'files_checked': len(want), 'missing': missing, 'extra': extra, 'mismatched': mismatched}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Directory written by trasa.build_dataset')
    parser.add_argument('--expected', required=True, help='Published checksums.json (e.g. datasets/<name>/expected_checksums.json)')
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args(argv)
    report = verify(args.dataset, args.expected)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text)
    summary = {k: (len(v) if isinstance(v, list) else {c: len(f) for c, f in v.items()} if isinstance(v, dict) else v)
               for k, v in report.items()}
    print(json.dumps(summary, indent=2))
    if not report['valid']:
        print('Dataset differs from the published checksums.', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
