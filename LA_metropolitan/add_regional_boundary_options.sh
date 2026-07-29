#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-la_metro_topo_metadata_cleaned_v2}"
CACHE_DIR="la_metro_topo_metadata/_cache"

mkdir -p "$CACHE_DIR"

TMP_PY="/tmp/add_regional_boundary_options_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import sys
import requests
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(sys.argv[1])
CACHE_DIR = Path("la_metro_topo_metadata/_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAP_GPKG = OUT_DIR / "la_metro_topo_map_editions_clean.gpkg"
BOUNDARY_GPKG = OUT_DIR / "regional_boundary_options.gpkg"
SUMMARY_DIR = OUT_DIR / "summary_tables"
PLOTS_DIR = OUT_DIR / "plots"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TIGER_YEAR = 2025

URLS = {
    "csa": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/CSA/tl_{TIGER_YEAR}_us_csa.zip",
    "cbsa": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/CBSA/tl_{TIGER_YEAR}_us_cbsa.zip",
    "county": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/COUNTY/tl_{TIGER_YEAR}_us_county.zip",
}

FILES = {
    "csa": CACHE_DIR / f"tl_{TIGER_YEAR}_us_csa.zip",
    "cbsa": CACHE_DIR / f"tl_{TIGER_YEAR}_us_cbsa.zip",
    "county": CACHE_DIR / f"tl_{TIGER_YEAR}_us_county.zip",
}

def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)

for key in URLS:
    download(URLS[key], FILES[key])

print("Reading TIGER boundaries...")
csa = gpd.read_file(f"zip://{FILES['csa']}").to_crs("EPSG:4326")
cbsa = gpd.read_file(f"zip://{FILES['cbsa']}").to_crs("EPSG:4326")
county = gpd.read_file(f"zip://{FILES['county']}").to_crs("EPSG:4326")

# Codes:
# CSA 348 = Los Angeles-Long Beach, CA CSA
# CBSA 31080 = Los Angeles-Long Beach-Anaheim, CA MSA
# CBSA 40140 = Riverside-San Bernardino-Ontario, CA MSA
# CBSA 37100 = Oxnard-Thousand Oaks-Ventura, CA MSA
# County 06037 = Los Angeles County
CSA_CODE = "348"
COMPONENT_CBSA_CODES = ["31080", "40140", "37100"]
LA_COUNTY_GEOID = "06037"

la_csa = csa[csa["GEOID"].astype(str) == CSA_CODE].copy()
component_msas = cbsa[cbsa["GEOID"].astype(str).isin(COMPONENT_CBSA_CODES)].copy()
la_msa = cbsa[cbsa["GEOID"].astype(str) == "31080"].copy()
la_county = county[county["GEOID"].astype(str) == LA_COUNTY_GEOID].copy()

if la_csa.empty:
    raise RuntimeError("Could not find CSA GEOID 348.")
if len(component_msas) != 3:
    print("WARNING: Expected 3 component MSAs, found:", len(component_msas))
if la_msa.empty:
    raise RuntimeError("Could not find LA MSA GEOID 31080.")
if la_county.empty:
    raise RuntimeError("Could not find LA County GEOID 06037.")

# Add friendly names.
la_csa["boundary_option"] = "Option C: Los Angeles–Long Beach, CA CSA"
la_csa["boundary_level"] = "CSA"
la_csa["source"] = f"US Census TIGER/Line {TIGER_YEAR}; OMB-defined CSA"

component_msas["boundary_option"] = component_msas["NAME"].apply(
    lambda x: f"Option D component MSA: {x}"
)
component_msas["boundary_level"] = "Component MSA"
component_msas["source"] = f"US Census TIGER/Line {TIGER_YEAR}; OMB-defined CBSA"

la_msa["boundary_option"] = "Option B: Los Angeles–Long Beach–Anaheim, CA MSA"
la_msa["boundary_level"] = "MSA"
la_msa["source"] = f"US Census TIGER/Line {TIGER_YEAR}; OMB-defined CBSA"

la_county["boundary_option"] = "Option A: Los Angeles County"
la_county["boundary_level"] = "County"
la_county["source"] = f"US Census TIGER/Line {TIGER_YEAR} County"

# Counties useful for explaining CSA composition.
CSA_COUNTIES = {
    "06037": "Los Angeles County",
    "06059": "Orange County",
    "06111": "Ventura County",
    "06065": "Riverside County",
    "06071": "San Bernardino County",
}
csa_counties = county[county["GEOID"].astype(str).isin(CSA_COUNTIES.keys())].copy()
csa_counties["boundary_option"] = csa_counties["NAMELSAD"]
csa_counties["boundary_level"] = "CSA component county"
csa_counties["source"] = f"US Census TIGER/Line {TIGER_YEAR} County"

# Clean columns.
def keep_boundary_cols(gdf):
    cols = [
        c for c in [
            "GEOID", "CSAFP", "CBSAFP", "STATEFP", "COUNTYFP",
            "NAME", "NAMELSAD", "boundary_option", "boundary_level", "source",
            "geometry"
        ]
        if c in gdf.columns
    ]
    return gdf[cols].copy()

la_csa = keep_boundary_cols(la_csa)
component_msas = keep_boundary_cols(component_msas)
la_msa = keep_boundary_cols(la_msa)
la_county = keep_boundary_cols(la_county)
csa_counties = keep_boundary_cols(csa_counties)

all_options = pd.concat(
    [
        la_county,
        la_msa,
        la_csa,
        component_msas,
        csa_counties,
    ],
    ignore_index=True,
)
all_options = gpd.GeoDataFrame(all_options, geometry="geometry", crs="EPSG:4326")

# Write fresh boundary options file.
if BOUNDARY_GPKG.exists():
    BOUNDARY_GPKG.unlink()

la_county.to_file(BOUNDARY_GPKG, layer="option_a_la_county", driver="GPKG")
la_msa.to_file(BOUNDARY_GPKG, layer="option_b_la_msa_current", driver="GPKG")
la_csa.to_file(BOUNDARY_GPKG, layer="option_c_la_csa", driver="GPKG")
component_msas.to_file(BOUNDARY_GPKG, layer="option_d_csa_component_msas", driver="GPKG")
csa_counties.to_file(BOUNDARY_GPKG, layer="csa_component_counties", driver="GPKG")
all_options.to_file(BOUNDARY_GPKG, layer="all_boundary_options", driver="GPKG")

# Also add layers to the main map GeoPackage for convenience.
if MAP_GPKG.exists():
    la_county.to_file(MAP_GPKG, layer="option_a_la_county", driver="GPKG")
    la_msa.to_file(MAP_GPKG, layer="option_b_la_msa_current", driver="GPKG")
    la_csa.to_file(MAP_GPKG, layer="option_c_la_csa", driver="GPKG")
    component_msas.to_file(MAP_GPKG, layer="option_d_csa_component_msas", driver="GPKG")
    csa_counties.to_file(MAP_GPKG, layer="csa_component_counties", driver="GPKG")

# Area comparison.
AREA_CRS = "EPSG:3310"

area_rows = []
for label, gdf in [
    ("Option A: LA County", la_county),
    ("Option B: LA MSA/current", la_msa),
    ("Option C: LA CSA", la_csa),
]:
    g = gdf.to_crs(AREA_CRS)
    area_km2 = g.geometry.union_all().area / 1_000_000
    area_rows.append({
        "boundary_option": label,
        "area_km2": area_km2,
    })

for _, row in component_msas.iterrows():
    g = gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").to_crs(AREA_CRS)
    area_rows.append({
        "boundary_option": f"Component MSA: {row['NAME']}",
        "area_km2": g.geometry.union_all().area / 1_000_000,
    })

area_df = pd.DataFrame(area_rows).sort_values("area_km2")
area_df.to_csv(SUMMARY_DIR / "regional_boundary_options_area.csv", index=False)

# Current inventory overlap with each option.
# Important: current map inventory was collected for LA MSA, so this is an audit,
# not a final CSA dataset coverage estimate.
if MAP_GPKG.exists():
    try:
        maps = gpd.read_file(MAP_GPKG, layer="map_editions").to_crs(AREA_CRS)
        maps_union = maps.geometry.union_all()

        overlap_rows = []
        for label, gdf in [
            ("Option A: LA County", la_county),
            ("Option B: LA MSA/current", la_msa),
            ("Option C: LA CSA", la_csa),
        ]:
            b = gdf.to_crs(AREA_CRS).geometry.union_all()
            b_area = b.area / 1_000_000
            covered = maps_union.intersection(b).area / 1_000_000
            overlap_rows.append({
                "boundary_option": label,
                "boundary_area_km2": b_area,
                "covered_by_current_inventory_union_km2": covered,
                "covered_by_current_inventory_pct": 100 * covered / b_area if b_area else None,
                "note": "Current map inventory was collected for LA MSA; CSA coverage requires re-querying if CSA is selected.",
            })

        pd.DataFrame(overlap_rows).to_csv(
            SUMMARY_DIR / "current_inventory_overlap_with_boundary_options.csv",
            index=False,
        )
    except Exception as e:
        print("WARNING: Could not compute inventory overlap:", e)

# Overview plot.
fig, ax = plt.subplots(figsize=(10, 8))

la_csa.boundary.plot(ax=ax, linewidth=2.0, color="#6a3d9a", label="LA–Long Beach CSA")
component_msas.boundary.plot(ax=ax, linewidth=1.4, color="#1b9e77", label="Component MSAs")
la_msa.boundary.plot(ax=ax, linewidth=2.0, color="#ff7f00", label="Current LA MSA")
la_county.boundary.plot(ax=ax, linewidth=2.0, color="#e31a1c", label="LA County")
csa_counties.boundary.plot(ax=ax, linewidth=0.6, color="#777777", linestyle="--", label="CSA component counties")

ax.set_title("Boundary options for LA topo-map dataset expansion")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "regional_boundary_options_overview.png", dpi=250)
plt.close()

readme = OUT_DIR / "REGIONAL_BOUNDARY_OPTIONS_README.txt"
readme.write_text(f"""Regional boundary options added.

Main file:
  {BOUNDARY_GPKG}

Layers:
  option_a_la_county
  option_b_la_msa_current
  option_c_la_csa
  option_d_csa_component_msas
  csa_component_counties
  all_boundary_options

Also added these layers to:
  {MAP_GPKG}

Useful tables:
  summary_tables/regional_boundary_options_area.csv
  summary_tables/current_inventory_overlap_with_boundary_options.csv

Useful plot:
  plots/regional_boundary_options_overview.png

Interpretation:
  Option B is the boundary used for the current collected inventory.
  Option C, the CSA, is much larger. If the team chooses the CSA as the expansion boundary,
  the TNM metadata collection should be rerun using the CSA geometry so that Riverside,
  San Bernardino, and Ventura-area map sheets are not missed.
""")

print("Done.")
print("Boundary options GeoPackage:", BOUNDARY_GPKG)
print("Overview plot:", PLOTS_DIR / "regional_boundary_options_overview.png")
print("Area table:", SUMMARY_DIR / "regional_boundary_options_area.csv")
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done."
echo "Created:"
echo "  ${OUT_DIR}/regional_boundary_options.gpkg"
echo "  ${OUT_DIR}/REGIONAL_BOUNDARY_OPTIONS_README.txt"
echo "  ${OUT_DIR}/plots/regional_boundary_options_overview.png"
echo "  ${OUT_DIR}/summary_tables/regional_boundary_options_area.csv"
echo "  ${OUT_DIR}/summary_tables/current_inventory_overlap_with_boundary_options.csv"
