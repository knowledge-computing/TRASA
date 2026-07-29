#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-la_metro_topo_metadata_cleaned_v2}"
IN_GPKG="${OUT_DIR}/la_metro_topo_map_editions_clean.gpkg"
OUT_GPKG="${OUT_DIR}/la_metro_topo_qgis_visual_layers.gpkg"

if [ ! -f "$IN_GPKG" ]; then
  echo "ERROR: missing input GeoPackage: $IN_GPKG"
  exit 1
fi

TMP_PY="/tmp/make_qgis_ready_layers_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import re
import sys
import geopandas as gpd
import pandas as pd

out_dir = Path(sys.argv[1])
in_gpkg = out_dir / "la_metro_topo_map_editions_clean.gpkg"
out_gpkg = out_dir / "la_metro_topo_qgis_visual_layers.gpkg"

if out_gpkg.exists():
    out_gpkg.unlink()

maps = gpd.read_file(in_gpkg, layer="map_editions")
boundary = gpd.read_file(in_gpkg, layer="la_metro_cbsa_boundary")

for c in ["year_clean", "scale_clean", "decade", "overlap_pct_of_map"]:
    if c in maps.columns:
        maps[c] = pd.to_numeric(maps[c], errors="coerce")

def safe_layer_name(x):
    x = str(x)
    x = x.replace(":", "_")
    x = x.replace(",", "")
    x = x.replace(".", "_")
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x.lower()

def write_layer(gdf, layer_name):
    if len(gdf) == 0:
        return
    gdf.to_file(out_gpkg, layer=layer_name, driver="GPKG")

# Main layers
write_layer(boundary, "la_metro_boundary")
write_layer(maps, "all_map_editions")

# Scale layers
for scale in sorted(maps["scale_label"].dropna().unique(),
                    key=lambda s: int(str(s).replace("1:", "").replace(",", ""))):
    subset = maps[maps["scale_label"] == scale].copy()
    layer = "scale_" + safe_layer_name(scale)
    write_layer(subset, layer)

# Decade layers
for decade in sorted(maps["decade"].dropna().unique()):
    decade_int = int(decade)
    subset = maps[maps["decade"] == decade].copy()
    layer = f"decade_{decade_int}s"
    write_layer(subset, layer)

# Product group layers
for group in sorted(maps["product_group"].dropna().unique()):
    subset = maps[maps["product_group"] == group].copy()
    layer = "product_" + safe_layer_name(group)
    write_layer(subset, layer)

# Overlap-bin layers
def overlap_bin(x):
    if pd.isna(x):
        return "missing"
    if x < 1:
        return "lt_1pct"
    if x < 5:
        return "1_5pct"
    if x < 10:
        return "5_10pct"
    if x < 25:
        return "10_25pct"
    if x < 50:
        return "25_50pct"
    if x < 75:
        return "50_75pct"
    if x < 100:
        return "75_100pct"
    return "100pct"

maps["overlap_bin_qgis"] = maps["overlap_pct_of_map"].apply(overlap_bin)

for b in ["lt_1pct", "1_5pct", "5_10pct", "10_25pct", "25_50pct", "50_75pct", "75_100pct", "100pct", "missing"]:
    subset = maps[maps["overlap_bin_qgis"] == b].copy()
    write_layer(subset, "overlap_" + b)

# Practical high-level subsets
write_layer(maps[maps["overlap_pct_of_map"] >= 25].copy(), "maps_overlap_ge_25pct")
write_layer(maps[maps["overlap_pct_of_map"] >= 50].copy(), "maps_overlap_ge_50pct")
write_layer(maps[maps["year_clean"] < 1950].copy(), "historical_before_1950")
write_layer(maps[(maps["year_clean"] >= 1950) & (maps["year_clean"] < 1980)].copy(), "historical_1950_1979")
write_layer(maps[maps["year_clean"] >= 2000].copy(), "modern_2000_plus")

readme = out_dir / "QGIS_VISUAL_LAYERS_README.txt"
readme.write_text(f"""QGIS visual layers created.

Open this file in QGIS:
  {out_gpkg}

Important layers:
  la_metro_boundary
  all_map_editions

Scale layers:
  scale_1_20000
  scale_1_24000
  scale_1_31680
  scale_1_48000
  scale_1_50000
  scale_1_62500
  scale_1_96000
  scale_1_100000
  scale_1_125000
  scale_1_250000

Decade layers:
  decade_1890s, decade_1900s, ...

Product layers:
  product_historical_topographic_maps
  product_us_topo

Overlap layers:
  overlap_lt_1pct
  overlap_1_5pct
  overlap_5_10pct
  overlap_10_25pct
  overlap_25_50pct
  overlap_50_75pct
  overlap_75_100pct
  overlap_100pct

Recommended QGIS styling:
  - boundary: no fill, thick outline
  - footprint layers: transparent fill, colored outline
  - do not use opaque fills for all layers together because footprints overlap.
""")

print("Wrote:", out_gpkg)
print("Wrote:", readme)
print("Layers created:")
print("  all_map_editions")
print("  by scale")
print("  by decade")
print("  by product group")
print("  by overlap bin")
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done."
echo "Open this in QGIS:"
echo "  ${OUT_GPKG}"
