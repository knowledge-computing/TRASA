"""Generate explicit complete-map fold manifests with a recorded schedule."""
import argparse
import json
import os
from pathlib import Path


def fold_schedules(map_ids):
    maps = sorted(set(map_ids))
    if len(maps) != 8:
        raise ValueError('The paper benchmark requires eight explicitly identified maps')
    return [{'test': maps[i], 'val': maps[(i + 1) % len(maps)],
             'train': [m for m in maps if m not in (maps[i], maps[(i + 1) % len(maps)])]}
            for i in range(len(maps))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--schedule', help='Exact JSON list of {train:[...],val:map_id,test:map_id}')
    args = parser.parse_args(argv)
    source_path = Path(args.manifest).resolve()
    document = json.loads(source_path.read_text())
    maps = set(row['map_id'] for row in document['samples'])
    schedules = json.loads(Path(args.schedule).read_text()) if args.schedule else fold_schedules(maps)
    if len(schedules) != 8 or {s['test'] for s in schedules} != maps:
        raise ValueError('Each of eight maps must be held out once')
    for schedule in schedules:
        train = set(schedule['train'])
        if len(train) != 6 or schedule['test'] == schedule['val'] or train & {schedule['test'], schedule['val']} or train | {schedule['test'], schedule['val']} != maps:
            raise ValueError('Every fold requires six train maps and separate validation/test maps')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    output.mkdir(parents=True)
    for number, schedule in enumerate(schedules):
        rows = []
        for row in document['samples']:
            sample = {**row, 'split': 'test' if row['map_id'] == schedule['test'] else 'val' if row['map_id'] == schedule['val'] else 'train'}
            for key in ('image', 'graph', 'road_mask'):
                if row.get(key):
                    sample[key] = os.path.relpath((source_path.parent / row[key]).resolve(), output)
            # Guidance must be prepared separately for each fold.
            sample.pop('guidance', None)
            sample.pop('guidance_provenance', None)
            rows.append(sample)
        (output / f'fold_{number:02d}.json').write_text(json.dumps({'schema_version': 1, 'fold': number, 'schedule': schedule, 'samples': rows}, indent=2) + '\n')
    (output / 'schedule.json').write_text(json.dumps(schedules, indent=2) + '\n')


if __name__ == '__main__':
    main()
