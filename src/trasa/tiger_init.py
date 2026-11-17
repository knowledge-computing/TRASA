"""Make a first-draft road annotation for one sheet from 2021 TIGER/Line roads.

This is the starting point described in the paper (Sec. 3): TIGER/Line roads
clipped to the quadrangle, with a default road type from the MTFCC code. The
draft is NOT a reference: it must be corrected against the printed map
(remove roads the map does not show, add missing ones, move centerlines onto
the symbols, set the five paper road types) before it goes into a dataset spec.

Counties are given as 5-digit FIPS codes (state + county), e.g. 06037 for
Los Angeles County. Each county file is downloaded once into the cache.
"""
import argparse
import json
from pathlib import Path

from .dataset_spec import Spec, default_cache, file_sha256

TIGER_URL = 'https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{county}_roads.zip'

# MTFCC -> paper road type. Paved/unpaved light duty roads cannot be told apart
# from TIGER; annotators set light_duty_road_paved from the printed symbol.
MTFCC_TYPES = {
    'S1100': 'highway',            # primary road
    'S1200': 'secondary_highway',  # secondary road
    'S1400': 'light_duty_road',    # local neighborhood road, rural road, city street
    'S1500': 'trail',              # vehicular trail (4WD)
    'S1630': 'secondary_highway',  # ramp
    'S1640': 'light_duty_road',    # service drive
    'S1710': 'trail',              # walkway / pedestrian trail
    'S1730': 'light_duty_road',    # alley
    'S1740': 'light_duty_road',    # private road for service vehicles
    'S1820': 'trail',              # bike path
    'S1830': 'trail',              # bridle path
}
# Not drawn as roads on topographic maps: stairways, census internal use, parking lot roads, medians.
MTFCC_SKIP = {'S1720', 'S1750', 'S1780', 'S2000'}


def draft(county_files, bounds, bounds_crs, output, year=2021):
    import fiona
    from pyproj import Transformer
    from shapely.geometry import box, mapping, shape
    from shapely.ops import transform as reproject
    to_tiger = Transformer.from_crs(bounds_crs, 'EPSG:4269', always_xy=True)
    west, south, east, north = bounds
    corners = [to_tiger.transform(x, y) for x, y in ((west, south), (east, south), (east, north), (west, north))]
    xs, ys = zip(*corners)
    frame = box(min(xs), min(ys), max(xs), max(ys))
    features, counts = [], {}
    for path in county_files:
        with fiona.open(f'zip://{path}') as collection:
            to_nad83 = None
            source_crs = collection.crs_wkt
            if source_crs:
                transformer = Transformer.from_crs(source_crs, 'EPSG:4269', always_xy=True)
                to_nad83 = lambda x, y, z=None: transformer.transform(x, y)
            for feature in collection.filter(bbox=frame.bounds):
                properties = dict(feature['properties'])
                mtfcc = properties.get('MTFCC')
                if mtfcc in MTFCC_SKIP:
                    continue
                geometry = shape(feature['geometry'])
                if to_nad83 is not None:
                    geometry = reproject(to_nad83, geometry)
                clipped = geometry.intersection(frame)
                if clipped.is_empty:
                    continue
                road_type = MTFCC_TYPES.get(mtfcc, 'light_duty_road')
                counts[road_type] = counts.get(road_type, 0) + 1
                features.append({'type': 'Feature', 'geometry': mapping(clipped),
                                 'properties': {'road_type': road_type, 'mtfcc': mtfcc, 'fullname': properties.get('FULLNAME'),
                                                'linearid': properties.get('LINEARID'), 'source': f'tiger{year}',
                                                'correction': 'uncorrected'}})
    features.sort(key=lambda f: (str(f['properties']['linearid']), json.dumps(f['geometry'], sort_keys=True)))
    document = {'type': 'FeatureCollection', 'name': Path(output).stem,
                'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::4269'}}, 'features': features}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(document) + '\n')
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--map', required=True, help='map_id in the spec (its neatline gives the clip box)')
    parser.add_argument('--counties', required=True, nargs='+', help='5-digit county FIPS codes covering the sheet')
    parser.add_argument('--output', required=True, help='Draft GeoJSON to correct (EPSG:4269)')
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--cache', default=str(default_cache()))
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise ValueError('Output exists; choose a new file')
    spec = Spec(args.spec)
    entry = next((m for m in spec.maps if m['map_id'] == args.map), None)
    if entry is None or 'west' not in (entry.get('neatline') or {}):
        raise ValueError('The map needs geographic neatline bounds in the spec')
    from .fetch_sources import download
    files, hashes = [], {}
    for county in args.counties:
        if len(county) != 5 or not county.isdigit():
            raise ValueError(f'County FIPS must have 5 digits: {county}')
        url = TIGER_URL.format(year=args.year, county=county)
        path = Path(args.cache) / 'tiger' / str(args.year) / Path(url).name
        if not path.exists():
            download(url, path)
        files.append(path)
        hashes[path.name] = file_sha256(path)
    n = entry['neatline']
    counts = draft(files, (n['west'], n['south'], n['east'], n['north']), n['crs'], args.output, args.year)
    print(json.dumps({'features_by_type': counts, 'tiger_files_sha256': hashes}, indent=2))


if __name__ == '__main__':
    main()
