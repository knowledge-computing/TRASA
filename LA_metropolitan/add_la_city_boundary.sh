#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-la_metro_topo_metadata_cleaned_v2}"
GPKG="${OUT_DIR}/la_metro_topo_map_editions_clean.gpkg"
CACHE_DIR="la_metro_topo_metadata/_cache"

mkdir -p "$CACHE_DIR"

TMP_PY="/tmp/add_la_city_boundary_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import sys
import requests
import geopandas as gpd

out_dir = Path(sys.argv[1])
gpkg = out_dir / "la_metro_topo_map_editions_clean.gpkg"
cache_dir = Path("la_metro_topo_metadata/_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

place_zip = cache_dir / "tl_2025_06_place.zip"
place_url = "https://www2.census.gov/geo/tiger/TIGER2025/PLACE/tl_2025_06_place.zip"

if not gpkg.exists():
    raise FileNotFoundError(gpkg)

if not place_zip.exists():
    print("Downloading California place boundaries...")
    r = requests.get(place_url, timeout=120)
    r.raise_for_status()
    place_zip.write_bytes(r.content)

places = gpd.read_file(f"zip://{place_zip}").to_crs("EPSG:4326")

# City of Los Angeles, CA.
# STATEFP 06, PLACEFP 44000, GEOID 0644000.
la_city = places[places["GEOID"] == "0644000"].copy()

if la_city.empty:
    raise RuntimeError("Could not find City of Los Angeles GEOID 0644000.")

la_city["boundary_type"] = "City of Los Angeles"
la_city["source"] = "US Census TIGER/Line 2025 Place"

keep_cols = [
    "STATEFP",
    "PLACEFP",
    "GEOID",
    "NAME",
    "NAMELSAD",
    "boundary_type",
    "source",
    "geometry",
]
la_city = la_city[keep_cols]

# Add to main GeoPackage.
la_city.to_file(gpkg, layer="city_of_los_angeles_boundary", driver="GPKG")

# Also write separate file for easy drag-and-drop.
single_out = out_dir / "city_of_los_angeles_boundary.gpkg"
if single_out.exists():
    single_out.unlink()
la_city.to_file(single_out, layer="city_of_los_angeles_boundary", driver="GPKG")

print("Done.")
print(f"Added layer to: {gpkg}")
print(f"Also wrote: {single_out}")
print("Layer name: city_of_los_angeles_boundary")
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done. City of Los Angeles boundary added."
echo "Files:"
echo "  ${GPKG}"
echo "  ${OUT_DIR}/city_of_los_angeles_boundary.gpkg"
