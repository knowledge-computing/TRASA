#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-regional_scope_comparison}"
CACHE_DIR="la_metro_topo_metadata/_cache"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

TMP_PY="/tmp/compare_la_boundary_options_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import sys
import time
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Reuse final working code, but write outputs separately.
sys.path.insert(0, str(Path(".").resolve()))

from collect_la_topo_metadata import (
    make_session,
    query_tnm_products_for_dataset,
    normalize_tnm_items,
)
from repair_la_topo_analysis import clean_metadata

OUT_DIR = Path(sys.argv[1])
CACHE_DIR = Path("la_metro_topo_metadata/_cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_DIR = OUT_DIR / "summary_tables"
PLOTS_DIR = OUT_DIR / "plots"
SUMMARY_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TIGER_YEAR = 2025

URLS = {
    "csa": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/CSA/tl_{TIGER_YEAR}_us_csa.zip",
    "cbsa": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/CBSA/tl_{TIGER_YEAR}_us_cbsa.zip",
    "county": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/COUNTY/tl_{TIGER_YEAR}_us_county.zip",
    "place": f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/PLACE/tl_{TIGER_YEAR}_06_place.zip",
}

FILES = {
    "csa": CACHE_DIR / f"tl_{TIGER_YEAR}_us_csa.zip",
    "cbsa": CACHE_DIR / f"tl_{TIGER_YEAR}_us_cbsa.zip",
    "county": CACHE_DIR / f"tl_{TIGER_YEAR}_us_county.zip",
    "place": CACHE_DIR / f"tl_{TIGER_YEAR}_06_place.zip",
}

def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    print("Downloading:", url)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)

for key in URLS:
    download(URLS[key], FILES[key])

print("Reading official boundaries...")
csa = gpd.read_file(f"zip://{FILES['csa']}").to_crs("EPSG:4326")
cbsa = gpd.read_file(f"zip://{FILES['cbsa']}").to_crs("EPSG:4326")
county = gpd.read_file(f"zip://{FILES['county']}").to_crs("EPSG:4326")
place = gpd.read_file(f"zip://{FILES['place']}").to_crs("EPSG:4326")

# Boundary codes.
# City of Los Angeles: GEOID 0644000
# Los Angeles County: GEOID 06037
# LA MSA: CBSA GEOID 31080
# Greater LA CSA: CSA GEOID 348
boundaries = {
    "city_of_los_angeles": place[place["GEOID"].astype(str) == "0644000"].copy(),
    "los_angeles_county": county[county["GEOID"].astype(str) == "06037"].copy(),
    "la_msa_current_inventory": cbsa[cbsa["GEOID"].astype(str) == "31080"].copy(),
    "greater_la_csa": csa[csa["GEOID"].astype(str) == "348"].copy(),
}

for name, gdf in boundaries.items():
    if gdf.empty:
        raise RuntimeError(f"Boundary not found: {name}")

# Use Greater LA CSA for the exploratory query.
greater_la = boundaries["greater_la_csa"]
greater_la_geom = greater_la.geometry.iloc[0]
bbox = tuple(float(v) for v in greater_la_geom.bounds)

print("Greater LA CSA bbox:", bbox)

session = make_session()

datasets = [
    "Historical Topographic Maps",
    "US Topo",
]

print("Querying TNMAccess for Greater LA CSA extent...")
items = []

for dataset in datasets:
    print("Dataset:", dataset)
    got = query_tnm_products_for_dataset(
        session=session,
        bbox_wgs84=bbox,
        dataset_name=dataset,
    )
    items.extend(got)
    time.sleep(0.5)

print("Raw TNM product/file records returned:", len(items))

raw_gdf = normalize_tnm_items(items, greater_la)
print("Raw records intersecting Greater LA CSA:", len(raw_gdf))

raw_csv = OUT_DIR / "raw_topo_product_records_greater_la_csa_query.csv"
raw_gdf.drop(columns="geometry").to_csv(raw_csv, index=False)

# Clean/parser stage, same logic as final LA metro inventory.
cleaned = clean_metadata(pd.DataFrame(raw_gdf.drop(columns="geometry")))
cleaned = cleaned.reset_index(drop=True)

# Attach geometry back by row order after filtering.
raw_with_idx = raw_gdf.reset_index(drop=True).copy()
raw_with_idx["_raw_row"] = range(len(raw_with_idx))

cleaned_with_idx = clean_metadata(
    pd.concat(
        [
            pd.DataFrame(raw_gdf.drop(columns="geometry")).reset_index(drop=True),
            pd.DataFrame({"_raw_row": range(len(raw_gdf))}),
        ],
        axis=1,
    )
)

cleaned_with_geom = cleaned_with_idx.merge(
    raw_with_idx[["_raw_row", "geometry"]],
    on="_raw_row",
    how="left",
)

gdf = gpd.GeoDataFrame(cleaned_with_geom, geometry="geometry", crs="EPSG:4326")

if gdf.empty:
    raise RuntimeError("No cleaned topo records found for Greater LA CSA.")

# Boundary flags per topo product/file record.
for bname, bgdf in boundaries.items():
    geom = bgdf.geometry.iloc[0]
    gdf[f"in_{bname}"] = gdf.geometry.intersects(geom)

gdf["in_greater_la_but_not_current_msa"] = (
    gdf["in_greater_la_csa"] & ~gdf["in_la_msa_current_inventory"]
)

# Unique edition table for the whole Greater LA CSA query.
# Use one row per edition_key, but preserve any boundary flag if any file record intersects.
flag_cols = [
    "in_city_of_los_angeles",
    "in_los_angeles_county",
    "in_la_msa_current_inventory",
    "in_greater_la_csa",
    "in_greater_la_but_not_current_msa",
]

first_cols = [
    "edition_key",
    "product_group",
    "dataset",
    "map_name_clean",
    "state_clean",
    "scale_clean",
    "scale_label",
    "scale_bucket_clean",
    "year_clean",
    "decade",
    "series_clean",
    "available_formats",
    "preferred_format",
    "preferred_download_url",
    "overlap_pct_of_map",
    "bbox_west",
    "bbox_south",
    "bbox_east",
    "bbox_north",
    "title",
    "download_url",
]

existing_first_cols = [c for c in first_cols if c in gdf.columns]

editions = (
    gdf.groupby("edition_key", dropna=False)
    .agg(
        **{c: (c, "first") for c in existing_first_cols if c != "edition_key"},
        n_file_records=("edition_key", "count"),
        **{c: (c, "max") for c in flag_cols},
    )
    .reset_index()
)

# Save full comparison inventory separately.
editions.to_csv(OUT_DIR / "greater_la_csa_map_editions_with_boundary_flags.csv", index=False)
gdf.drop(columns="geometry").to_csv(OUT_DIR / "greater_la_csa_topo_product_records_clean.csv", index=False)

# Save spatial version separately.
gpkg_out = OUT_DIR / "greater_la_csa_map_editions_comparison.gpkg"
if gpkg_out.exists():
    gpkg_out.unlink()

# Attach one geometry per edition.
geom_by_key = gdf.drop_duplicates("edition_key")[["edition_key", "geometry"]]
editions_gdf = editions.merge(geom_by_key, on="edition_key", how="left")
editions_gdf = gpd.GeoDataFrame(editions_gdf, geometry="geometry", crs="EPSG:4326")

editions_gdf.to_file(gpkg_out, layer="greater_la_csa_map_editions", driver="GPKG")

for bname, bgdf in boundaries.items():
    bgdf.to_file(gpkg_out, layer=bname, driver="GPKG")

# ------------------------------------------------------------------
# Count maps for each boundary option.
# ------------------------------------------------------------------

option_labels = {
    "city_of_los_angeles": "City of Los Angeles",
    "los_angeles_county": "Los Angeles County",
    "la_msa_current_inventory": "LA–Long Beach–Anaheim MSA/current",
    "greater_la_csa": "Greater LA CSA",
    "greater_la_but_not_current_msa": "Greater LA CSA outside current LA MSA",
}

rows = []

for key in [
    "city_of_los_angeles",
    "los_angeles_county",
    "la_msa_current_inventory",
    "greater_la_csa",
    "greater_la_but_not_current_msa",
]:
    flag = f"in_{key}" if key != "greater_la_but_not_current_msa" else "in_greater_la_but_not_current_msa"

    sub_records = gdf[gdf[flag]].copy()
    sub_editions = editions[editions[flag]].copy()

    rows.append({
        "boundary_option": option_labels[key],
        "topo_product_file_records": int(len(sub_records)),
        "unique_map_editions": int(len(sub_editions)),
        "historical_topographic_map_editions": int((sub_editions["product_group"] == "Historical Topographic Maps").sum()),
        "us_topo_editions": int((sub_editions["product_group"] == "US Topo").sum()),
        "unique_quadrangle_names": int(sub_editions["map_name_clean"].nunique(dropna=True)),
        "min_year": int(sub_editions["year_clean"].min()) if len(sub_editions) else None,
        "max_year": int(sub_editions["year_clean"].max()) if len(sub_editions) else None,
    })

counts = pd.DataFrame(rows)
counts.to_csv(SUMMARY_DIR / "boundary_option_map_counts.csv", index=False)

# By product group.
pg_rows = []

for key, label in option_labels.items():
    flag = f"in_{key}" if key != "greater_la_but_not_current_msa" else "in_greater_la_but_not_current_msa"
    sub_editions = editions[editions[flag]].copy()

    for pg, n in sub_editions["product_group"].value_counts(dropna=False).items():
        pg_rows.append({
            "boundary_option": label,
            "product_group": pg,
            "unique_map_editions": int(n),
        })

pd.DataFrame(pg_rows).to_csv(
    SUMMARY_DIR / "boundary_option_counts_by_product_group.csv",
    index=False,
)

# By scale.
scale_rows = []

for key, label in option_labels.items():
    flag = f"in_{key}" if key != "greater_la_but_not_current_msa" else "in_greater_la_but_not_current_msa"
    sub_editions = editions[editions[flag]].copy()

    for scale, n in sub_editions["scale_label"].value_counts(dropna=False).items():
        scale_rows.append({
            "boundary_option": label,
            "scale_label": scale,
            "unique_map_editions": int(n),
        })

pd.DataFrame(scale_rows).to_csv(
    SUMMARY_DIR / "boundary_option_counts_by_scale.csv",
    index=False,
)

# Additional maps not in current LA MSA.
additional = editions[editions["in_greater_la_but_not_current_msa"]].copy()
additional.to_csv(
    SUMMARY_DIR / "additional_maps_greater_la_csa_outside_current_msa.csv",
    index=False,
)

# ------------------------------------------------------------------
# Plots.
# ------------------------------------------------------------------

plot_counts = counts.copy()
plot_counts["boundary_option_wrapped"] = plot_counts["boundary_option"].replace({
    "LA–Long Beach–Anaheim MSA/current": "LA MSA\ncurrent inventory",
    "Greater LA CSA outside current LA MSA": "Greater LA CSA\noutside current LA MSA",
    "City of Los Angeles": "City of\nLos Angeles",
    "Los Angeles County": "Los Angeles\nCounty",
    "Greater LA CSA": "Greater\nLA CSA",
})

ax = plot_counts.plot(
    kind="bar",
    x="boundary_option_wrapped",
    y="unique_map_editions",
    legend=False,
    figsize=(9, 5),
)
ax.set_title("Unique topo map editions by candidate boundary")
ax.set_xlabel("Boundary option")
ax.set_ylabel("Unique map editions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "boundary_option_unique_map_editions.png", dpi=240)
plt.close()

# Stacked product group plot.
pg = pd.DataFrame(pg_rows)
if len(pg):
    pivot = pg.pivot_table(
        index="boundary_option",
        columns="product_group",
        values="unique_map_editions",
        aggfunc="sum",
        fill_value=0,
    )

    order = counts["boundary_option"].tolist()
    pivot = pivot.reindex(order)

    pivot.index = [
        {
            "City of Los Angeles": "City of\nLos Angeles",
            "Los Angeles County": "Los Angeles\nCounty",
            "LA–Long Beach–Anaheim MSA/current": "LA MSA\ncurrent inventory",
            "Greater LA CSA": "Greater\nLA CSA",
            "Greater LA CSA outside current LA MSA": "Greater LA CSA\noutside current LA MSA",
        }.get(x, x)
        for x in pivot.index
    ]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_title("Topo map editions by boundary and product group")
    ax.set_xlabel("Boundary option")
    ax.set_ylabel("Unique map editions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boundary_option_by_product_group.png", dpi=240)
    plt.close()

# Scale heatmap-ish table plot.
scale_df = pd.DataFrame(scale_rows)
if len(scale_df):
    scale_pivot = scale_df.pivot_table(
        index="scale_label",
        columns="boundary_option",
        values="unique_map_editions",
        aggfunc="sum",
        fill_value=0,
    )

    def scale_sort_key(s):
        return int(str(s).replace("1:", "").replace(",", ""))

    scale_pivot = scale_pivot.reindex(
        sorted(scale_pivot.index, key=scale_sort_key)
    )
    scale_pivot = scale_pivot[counts["boundary_option"].tolist()]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(scale_pivot))))
    im = ax.imshow(scale_pivot.values, aspect="auto")

    ax.set_title("Topo map editions by scale and boundary option")
    ax.set_xlabel("Boundary option")
    ax.set_ylabel("Scale")
    ax.set_xticks(range(len(scale_pivot.columns)))
    ax.set_xticklabels(
        [
            {
                "City of Los Angeles": "City",
                "Los Angeles County": "County",
                "LA–Long Beach–Anaheim MSA/current": "LA MSA",
                "Greater LA CSA": "CSA",
                "Greater LA CSA outside current LA MSA": "CSA outside\ncurrent MSA",
            }.get(c, c)
            for c in scale_pivot.columns
        ],
        rotation=0,
    )
    ax.set_yticks(range(len(scale_pivot.index)))
    ax.set_yticklabels(scale_pivot.index)

    for i in range(scale_pivot.shape[0]):
        for j in range(scale_pivot.shape[1]):
            val = int(scale_pivot.values[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Unique map editions")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boundary_option_by_scale_heatmap.png", dpi=240)
    plt.close()

# README.
readme = OUT_DIR / "README.md"
readme.write_text(f"""# Regional Scope Comparison

This folder is exploratory and separate from the final LA metro statistics.

It compares the number of topo map editions intersecting several possible study boundaries:

1. City of Los Angeles
2. Los Angeles County
3. LA–Long Beach–Anaheim MSA/current inventory boundary
4. Greater LA CSA
5. Greater LA CSA outside the current LA MSA

Important:
- This does not modify `la_metro_topo_metadata_cleaned_v2/`.
- This reruns TNM metadata collection over the Greater LA CSA extent.
- Use this only for meeting discussion about possible future expansion scope.
- The advisor-requested LA metro statistics remain in `la_metro_topo_metadata_cleaned_v2/`.

Main outputs:
- `summary_tables/boundary_option_map_counts.csv`
- `summary_tables/boundary_option_counts_by_product_group.csv`
- `summary_tables/boundary_option_counts_by_scale.csv`
- `summary_tables/additional_maps_greater_la_csa_outside_current_msa.csv`
- `plots/boundary_option_unique_map_editions.png`
- `plots/boundary_option_by_product_group.png`
- `plots/boundary_option_by_scale_heatmap.png`
- `greater_la_csa_map_editions_with_boundary_flags.csv`
- `greater_la_csa_map_editions_comparison.gpkg`

""")

print()
print("Done.")
print("Raw Greater LA CSA topo product/file records:", len(gdf))
print("Unique Greater LA CSA map editions:", len(editions))
print()
print(counts.to_string(index=False))
print()
print("Outputs written to:", OUT_DIR)
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done."
echo "Created separate exploratory folder:"
echo "  ${OUT_DIR}"
echo ""
echo "Important outputs:"
echo "  ${OUT_DIR}/summary_tables/boundary_option_map_counts.csv"
echo "  ${OUT_DIR}/summary_tables/additional_maps_greater_la_csa_outside_current_msa.csv"
echo "  ${OUT_DIR}/plots/boundary_option_unique_map_editions.png"
echo "  ${OUT_DIR}/plots/boundary_option_by_product_group.png"
echo "  ${OUT_DIR}/plots/boundary_option_by_scale_heatmap.png"
