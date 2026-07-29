#!/usr/bin/env python3
"""
Extract research-relevant metadata from map files for spreadsheet analysis.
Handles both digital (GeoTIFFs) and scanned (historical) topographic maps.
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import rasterio
import re


def parse_filename(filename):
    """Extract location and date from filename like CA_Alberhill_20211211_TM_geo."""
    stem = Path(filename).stem

    # Pattern: STATE_Location_YYYYMMDD or STATE_Location_ID_YYYY_scale
    parts = stem.split('_')

    location = None
    date_str = None
    scale = None
    map_type = None

    # Find date patterns
    for part in parts:
        if len(part) == 8 and part.isdigit():  # YYYYMMDD
            date_str = part
            break
        elif len(part) == 4 and part.isdigit() and '19' in part or '20' in part:  # YYYY
            date_str = part
            break

    # Reconstruct location name (everything between state and date/ID)
    if date_str:
        date_idx = parts.index(date_str)
        location = ' '.join(parts[1:date_idx])
    else:
        # For scanned maps without clear date
        location = ' '.join(parts[1:-2]) if len(parts) > 3 else None
        # Try to find year in remaining parts
        for part in parts:
            if len(part) == 4 and '19' in part or '20' in part:
                date_str = part

    # Extract scale if present (usually '24000', '25000', etc.)
    for part in parts:
        if part.isdigit() and len(part) == 5:
            if part in ['24000', '25000', '20000', '63360']:
                scale = part

    # Determine map type from metadata
    if 'TM' in stem:
        map_type = 'Topographic (Digital)'

    return {
        'filename': filename,
        'location': location,
        'date': date_str,
        'scale': scale,
        'map_type': map_type
    }


def parse_xml_metadata(xml_path):
    """Extract key metadata from FGDC XML metadata file."""
    metadata = {
        'title': None,
        'abstract': None,
        'publisher': None,
        'publication_date': None,
        'projection': None,
        'datum': None,
        'source_date_begin': None,
        'source_date_end': None,
        'coverage_area': None,
        'accuracy_notes': None,
        'themes': []
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Define namespace
        ns = {'': 'http://www.fgdc.gov/metadata/csdgm'}

        # Title
        title_elem = root.find('.//title')
        if title_elem is not None:
            metadata['title'] = title_elem.text

        # Abstract
        abstract_elem = root.find('.//abstract')
        if abstract_elem is not None:
            metadata['abstract'] = abstract_elem.text[:200] if abstract_elem.text else None

        # Publisher
        publish_elem = root.find('.//publish')
        if publish_elem is not None:
            metadata['publisher'] = publish_elem.text

        # Publication date
        pubdate_elem = root.find('.//pubdate')
        if pubdate_elem is not None:
            metadata['publication_date'] = pubdate_elem.text

        # Projection
        projname = root.find('.//mapprojn')
        if projname is not None:
            metadata['projection'] = projname.text

        # Datum
        horizdn = root.find('.//horizdn')
        if horizdn is not None:
            metadata['datum'] = horizdn.text

        # Source dates
        begdate = root.find('.//timeperd/timeinfo/rngdates/begdate')
        enddate = root.find('.//timeperd/timeinfo/rngdates/enddate')
        if begdate is not None:
            metadata['source_date_begin'] = begdate.text
        if enddate is not None:
            metadata['source_date_end'] = enddate.text

        # Accuracy
        horizpar = root.find('.//horizpa/horizpar')
        if horizpar is not None:
            acc_text = horizpar.text
            metadata['accuracy_notes'] = acc_text[:150] if acc_text else None

        # Theme keywords
        theme_keys = root.findall('.//themekey')
        if theme_keys:
            metadata['themes'] = [tk.text for tk in theme_keys[:5]]  # First 5

    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")

    return metadata


def get_raster_metadata(tif_path):
    """Extract raster-specific metadata using rasterio."""
    metadata = {
        'width': None,
        'height': None,
        'resolution': None,
        'crs': None,
        'bounds': None,
        'file_size_mb': None,
        'dtype': None,
        'data_min': None,
        'data_max': None
    }

    try:
        file_size_mb = tif_path.stat().st_size / (1024 * 1024)
        metadata['file_size_mb'] = round(file_size_mb, 2)

        with rasterio.open(tif_path) as src:
            metadata['width'] = src.width
            metadata['height'] = src.height
            metadata['crs'] = str(src.crs) if src.crs else None

            # Get resolution from transform
            if src.transform:
                metadata['resolution'] = f"{src.transform.a}m x {src.transform.e}m"

            # Bounds
            bounds = src.bounds
            metadata['bounds'] = f"W:{bounds.left:.2f}, E:{bounds.right:.2f}, N:{bounds.top:.2f}, S:{bounds.bottom:.2f}"

            # Data type
            if src.dtypes:
                metadata['dtype'] = src.dtypes[0]

            # Get raster stats if available
            if src.count > 0:
                band = src.read(1)
                metadata['data_min'] = float(band.min())
                metadata['data_max'] = float(band.max())

    except Exception as e:
        print(f"Error reading raster {tif_path}: {e}")

    return metadata


def determine_map_category(filename, xml_metadata):
    """Determine map category for research purposes."""
    stem = str(filename).lower()

    # Digital vs Scanned based on date
    if any(c.isdigit() for c in stem):
        date_match = re.search(r'20\d{2}', stem)
        if date_match:
            year = int(date_match.group())
            if year >= 2010:
                return 'Contemporary Topographic'
            else:
                return 'Historical Topographic'

    # Check if historic based on file location
    if 'scanned' in str(filename):
        return 'Historical Topographic (Scanned)'
    elif 'digital' in str(filename):
        return 'Contemporary Topographic'

    return 'Topographic'


def extract_all_maps_metadata(maps_root='maps', output_csv='map_metadata_research.csv'):
    """Extract metadata from all map files and write to CSV."""
    maps_root = Path(maps_root)

    # Find all TIF files (avoid auxiliary files)
    tif_files = sorted([f for f in maps_root.rglob('*.tif') if not f.name.startswith('.')])

    # Filter to only main map files (not auxiliary files for the same map)
    main_tifs = {}
    for tif in tif_files:
        # Keep track of main files (geo.tif) vs auxiliary (ortho/etc)
        if '_geo.tif' in tif.name or tif.name.endswith('.tif'):
            base_name = tif.parent.name
            if base_name not in main_tifs:
                main_tifs[base_name] = tif

    records = []

    for idx, (base_name, tif_path) in enumerate(sorted(main_tifs.items()), 1):
        print(f"[{idx}/{len(main_tifs)}] Processing {tif_path.name}...", end=' ')

        # Parse filename
        filename_meta = parse_filename(tif_path.name)

        # Look for XML metadata
        xml_path = tif_path.parent / f"{tif_path.stem}.tif.xml"
        if not xml_path.exists():
            xml_path = tif_path.parent / "metadata.xml"

        xml_meta = parse_xml_metadata(xml_path) if xml_path.exists() else {}

        # Get raster metadata
        raster_meta = get_raster_metadata(tif_path)

        # Determine category
        category = determine_map_category(tif_path, xml_meta)

        # Build record
        record = {
            'Map_ID': base_name,
            'Location': filename_meta['location'],
            'Map_Type': category,
            'Publication_Year': filename_meta['date'] if filename_meta['date'] else xml_meta.get('publication_date', ''),
            'Map_Scale': filename_meta['scale'] or xml_meta.get('scale', '1:24000'),
            'Title': xml_meta.get('title', ''),
            'Publisher': xml_meta.get('publisher', ''),
            'Geographic_Area': xml_meta.get('coverage_area', ''),
            'Themes': ', '.join(xml_meta.get('themes', [])),
            'Projection': xml_meta.get('projection', ''),
            'Datum': xml_meta.get('datum', ''),
            'Source_Date_Begin': xml_meta.get('source_date_begin', ''),
            'Source_Date_End': xml_meta.get('source_date_end', ''),
            'Positional_Accuracy': xml_meta.get('accuracy_notes', ''),
            'Raster_Width_px': raster_meta['width'],
            'Raster_Height_px': raster_meta['height'],
            'Resolution': raster_meta['resolution'],
            'CRS': raster_meta['crs'],
            'Geographic_Bounds': raster_meta['bounds'],
            'File_Size_MB': raster_meta['file_size_mb'],
            'File_Path': str(tif_path)
        }

        records.append(record)
        print("✓")

    # Write to CSV
    if records:
        keys = records[0].keys()
        output_path = Path(output_csv)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n✅ Extracted metadata for {len(records)} maps")
        print(f"📊 Output saved to: {output_path.absolute()}")
        return output_path
    else:
        print("❌ No map files found")
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract research metadata from California topographic maps'
    )
    parser.add_argument(
        '--maps-root',
        default='maps',
        help='Root directory containing map files (default: maps)'
    )
    parser.add_argument(
        '--output',
        default='map_metadata_research.csv',
        help='Output CSV filename (default: map_metadata_research.csv)'
    )

    args = parser.parse_args()
    extract_all_maps_metadata(args.maps_root, args.output)
