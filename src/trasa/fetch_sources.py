"""Download the USGS sheets and corrected road annotations named in a dataset spec.

Files go to a cache outside the repository (``--cache``, ``$TRASA_CACHE`` or
``~/.cache/trasa``). Every file is checked against its SHA256. A source may list
``mirrors`` (e.g. an archived copy; USGS maps are public domain) that are tried
when the USGS URL fails; the same SHA256 applies. Use ``--lock``
once, when a spec is first assembled, to record the hashes of the files you
downloaded in ``sources.lock.json``; commit that file so others get the same
bytes or a clear error.
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from .dataset_spec import Spec, default_cache, file_sha256, source_path


def download(url, target, attempts=4, timeout=120):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'trasa-dataset-fetch/1'})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
            temporary.replace(target)
            return target
        except OSError as error:
            if temporary.exists():
                temporary.unlink()
            if attempt == attempts:
                raise RuntimeError(f'Download failed after {attempts} attempts: {url}: {error}') from error
            time.sleep(min(60, 2 ** attempt))


def fetch(spec, cache, maps=None, lock=False, log=print):
    """Return {map_id: {key: {path, sha256}}}; raises on any hash mismatch."""
    result, lock_entries, problems = {}, json.loads(json.dumps(spec.lock)), []
    for entry in spec.maps:
        if maps and entry['map_id'] not in maps:
            continue
        result[entry['map_id']] = {}
        for key in ('raster', 'annotation'):
            source = entry[key]
            path = source_path(spec, entry, key, cache)
            if not path.exists():
                if not source.get('url'):
                    problems.append(f"{entry['map_id']} {key}: missing local file {path}")
                    continue
                urls = [source['url'], *source.get('mirrors', [])]
                for number, url in enumerate(urls):
                    log(f"download {entry['map_id']} {key}: {url}")
                    try:
                        download(url, path)
                        break
                    except RuntimeError as error:
                        if number + 1 == len(urls):
                            problems.append(str(error))
                if not path.exists():
                    continue
            digest = file_sha256(path)
            expected = source.get('sha256')
            if expected and digest != expected:
                problems.append(f"{entry['map_id']} {key}: SHA256 {digest} differs from expected {expected} ({path})")
            elif not expected and not lock:
                problems.append(f"{entry['map_id']} {key}: no SHA256 recorded; run with --lock once and commit sources.lock.json")
            if lock and not expected:
                record = {'sha256': digest, 'bytes': path.stat().st_size}
                record.update({k: source[k] for k in ('url', 'path') if source.get(k)})
                lock_entries.setdefault(entry['map_id'], {})[key] = record
            result[entry['map_id']][key] = {'path': str(path), 'sha256': digest}
    if problems:
        raise ValueError('\n'.join(problems))
    if lock:
        spec.lock_path.write_text(json.dumps(lock_entries, indent=2, sort_keys=True) + '\n')
        log(f'wrote {spec.lock_path}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--cache', default=str(default_cache()))
    parser.add_argument('--maps', nargs='*', help='Only these map IDs')
    parser.add_argument('--lock', action='store_true', help='Record SHA256 of sources that have none yet')
    args = parser.parse_args(argv)
    spec = Spec(args.spec)
    try:
        result = fetch(spec, Path(args.cache), args.maps, args.lock)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
