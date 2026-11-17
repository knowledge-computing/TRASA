"""Write dataset-spec map entries from the official USGS topo metadata tables.

USGS publishes product tables at
https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/historicaltopo.zip
(HTMC scans) and ustopo_current.zip / ustopo_historical.zip (US Topo). Give the
unzipped CSV and the products you annotated (HTMC ``scan_id`` or the product
file name); this prints YAML ``maps:`` entries with the GeoTIFF URL, year,
title and quadrangle bounds, so nobody types URLs by hand. Check each entry
with ``trasa.build_dataset --preview`` before committing it.
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import yaml

DATUMS = {'NAD27': 'EPSG:4267', 'NAD83': 'EPSG:4269', 'WGS84': 'EPSG:4326'}


def map_id(row):
    state = row.get('primary_state_abbr') or row.get('state') or ''
    name = re.sub(r'[^A-Za-z0-9]+', '', row.get('map_name', 'map'))
    year = row.get('date_on_map') or row.get('imprint_year') or ''
    identifier = row.get('scan_id') or ''
    parts = [p for p in (state, name, identifier, year) if p]
    return '_'.join(parts)


def entry(row, annotation_dir='annotations'):
    url = row.get('geotiff_url') or row.get('product_url')
    if not url:
        raise ValueError(f"No GeoTIFF/product URL for {row.get('map_name')}")
    datum = (row.get('hdatum_printed') or row.get('datum') or '').upper().replace(' ', '')
    identifier = map_id(row)
    result = {'map_id': identifier, 'title': row.get('map_name'), 'series': (row.get('series') or '').lower().replace(' ', '_'),
              'year': int(row['date_on_map']) if (row.get('date_on_map') or '').isdigit() else None,
              'scan_id': int(row['scan_id']) if (row.get('scan_id') or '').isdigit() else None,
              'raster': {'url': url},
              'annotation': {'path': f'{annotation_dir}/{identifier}.geojson'}}
    bounds = [row.get(k) for k in ('westbc', 'southbc', 'eastbc', 'northbc')]
    if all(bounds):
        crs = DATUMS.get(datum)
        result['neatline'] = {'west': float(bounds[0]), 'south': float(bounds[1]), 'east': float(bounds[2]),
                              'north': float(bounds[3]), 'crs': crs or 'TODO: printed datum of the neatline'}
    product = unquote(Path(url.split('?')[0]).name)
    if product.lower().endswith('.zip'):
        result['raster']['member'] = 'TODO: GeoTIFF name inside the zip, or remove if it holds one'
    return result


def lookup(rows, wanted):
    found = {}
    for row in rows:
        keys = {row.get('scan_id', ''), unquote(Path((row.get('geotiff_url') or '').split('?')[0]).name),
                unquote(Path((row.get('product_url') or '').split('?')[0]).name), row.get('product_filename', '')}
        for item in wanted:
            if item in keys:
                found[item] = row
    missing = [item for item in wanted if item not in found]
    return [found[item] for item in wanted if item in found], missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metadata', required=True, nargs='+', help='USGS metadata CSV file(s)')
    parser.add_argument('--products', required=True, nargs='+', help='HTMC scan_id values or product file names')
    args = parser.parse_args(argv)
    rows = []
    for path in args.metadata:
        with open(path, newline='', encoding='utf-8-sig') as stream:
            rows.extend(csv.DictReader(stream))
    found, missing = lookup(rows, args.products)
    if missing:
        print(f'Not found in metadata: {missing}', file=sys.stderr)
    print(yaml.safe_dump({'maps': [entry(row) for row in found]}, sort_keys=False))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
