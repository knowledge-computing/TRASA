#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-la_metro_topo_metadata_cleaned_v2}"
GPKG="${OUT_DIR}/la_metro_topo_map_editions_clean.gpkg"
CACHE_DIR="la_metro_topo_metadata/_cache"

mkdir -p "$CACHE_DIR"

TMP_PY="/tmp/add_la_county_boundary_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import sys
import requests
import geopandas as gpd

out_dir = Path(sys.argv[1])
gpkg = out_dir / "la_metro_topo_map_editions_clean.gpkg"
cache_dir = Path("la_metro_topo_metadata/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

county_zip = cache_dir / "tl_2025_us_county.zip"
county_url = "https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip"

if not gpkg.exists():
    raise FileNotFoundError(gpkg)

if not county_zip.exists():
    print("Downloading Census county boundary file...")
    r = requests.get(county_url, timeout=120)
    r.raise_for_status()
    county_zip.write_bytes(r.content)

print("Reading county boundaries...")
counties = gpd.read_file(f"zip://{county_zip}").to_crs("EPSG:4326")

la_county = counties[counties["GEOID"] == "06037"].copy()

if la_county.empty:
    raise RuntimeError("Could not find Los Angeles County GEOID 06037.")

la_county["boundary_type"] = "Los Angeles County"
la_county["source"] = "US Census TIGER/Line 2025 County"
la_county = la_county[
    [
        "STATEFP",
        "COUNTYFP",
        "GEOID",
        "NAME",
        "NAMELSAD",
        "boundary_type",
        "source",
        "geometry",
    ]
]

# Add/replace layer in the existing cleaned GeoPackage.
la_county.to_file(gpkg, layer="la_county_boundary", driver="GPKG")

# Also write a separate small GeoPackage for easy drag-and-drop into QGIS.
single_out = out_dir / "la_county_boundary.gpkg"
if single_out.exists():
    single_out.unlink()
la_county.to_file(single_out, layer="la_county_boundary", driver="GPKG")

print("Done.")
print(f"Added layer to: {gpkg}")
print(f"Also wrote: {single_out}")
print("Layer name: la_county_boundary")
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done. LA County boundary added."
echo "Files:"
echo "  ${GPKG}"
echo "  ${OUT_DIR}/la_county_boundary.gpkg"
