#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-la_metro_topo_metadata_cleaned_v2}"
GPKG="${OUT_DIR}/la_metro_topo_map_editions_clean.gpkg"
CSV="${OUT_DIR}/la_metro_topo_map_editions_clean.csv"
SUMMARY_DIR="${OUT_DIR}/summary_tables"
PLOTS_DIR="${OUT_DIR}/plots"

if [ ! -f "$GPKG" ]; then
  echo "ERROR: Missing GeoPackage: $GPKG"
  exit 1
fi

if [ ! -f "$CSV" ]; then
  echo "ERROR: Missing CSV: $CSV"
  exit 1
fi

mkdir -p "$SUMMARY_DIR" "$PLOTS_DIR"

TMP_PY="/tmp/la_topo_extra_stats_$$.py"

cat > "$TMP_PY" <<'PY'
from pathlib import Path
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit(f"geopandas is required for spatial coverage statistics: {e}")

out_dir = Path(sys.argv[1])
gpkg = out_dir / "la_metro_topo_map_editions_clean.gpkg"
csv_path = out_dir / "la_metro_topo_map_editions_clean.csv"
summary_dir = out_dir / "summary_tables"
plots_dir = out_dir / "plots"

summary_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

AREA_CRS = "EPSG:3310"  # California Albers, good for area estimates.

gdf = gpd.read_file(gpkg, layer="map_editions")
boundary = gpd.read_file(gpkg, layer="la_metro_cbsa_boundary")

# Defensive cleanup.
gdf = gdf[gdf.geometry.notna()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

for c in ["year_clean", "scale_clean", "decade", "overlap_pct_of_map", "overlap_area_km2"]:
    if c in gdf.columns:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

# Project for area calculations.
gdf_area = gdf.to_crs(AREA_CRS)
boundary_area = boundary.to_crs(AREA_CRS)
boundary_geom = boundary_area.geometry.union_all()
boundary_area_km2 = boundary_geom.area / 1_000_000

# Clip/intersect map footprints with LA metro boundary.
gdf_area["geometry_intersection"] = gdf_area.geometry.intersection(boundary_geom)
gdf_area = gdf_area[gdf_area["geometry_intersection"].notna()].copy()
gdf_area = gdf_area[~gdf_area["geometry_intersection"].is_empty].copy()
gdf_area["intersection_area_km2"] = gdf_area["geometry_intersection"].area / 1_000_000

def union_area_km2(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return 0.0
    return gpd.GeoSeries(geoms, crs=AREA_CRS).union_all().area / 1_000_000

def coverage_table(group_col, out_name):
    rows = []
    for value, group in gdf_area.groupby(group_col, dropna=False):
        covered = union_area_km2(group["geometry_intersection"])
        rows.append({
            group_col: value if pd.notna(value) else "missing",
            "n_map_editions": int(len(group)),
            "sum_sheet_overlap_area_km2": float(group["intersection_area_km2"].sum()),
            "union_covered_area_km2": float(covered),
            "la_metro_area_km2": float(boundary_area_km2),
            "coverage_pct_of_la_metro": float(100 * covered / boundary_area_km2) if boundary_area_km2 else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / out_name, index=False)
    return df

def sort_decade_df(df):
    if "decade" not in df.columns:
        return df
    tmp = df.copy()
    tmp["_sort"] = pd.to_numeric(tmp["decade"], errors="coerce")
    tmp = tmp.sort_values("_sort", na_position="last").drop(columns="_sort")
    return tmp

def sort_scale_df(df):
    if "scale_label" not in df.columns:
        return df
    tmp = df.copy()
    # Extract numeric denominator from labels like 1:24,000.
    tmp["_sort"] = (
        tmp["scale_label"].astype(str)
        .str.replace("1:", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    tmp["_sort"] = pd.to_numeric(tmp["_sort"], errors="coerce")
    tmp = tmp.sort_values("_sort", na_position="last").drop(columns="_sort")
    return tmp

def save_bar(df, x, y, title, xlabel, ylabel, out_png, rotate=True):
    if df.empty:
        return
    ax = df.plot(kind="bar", x=x, y=y, figsize=(11, 5), legend=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / out_png, dpi=220)
    plt.close()

# ---------------------------------------------------------------------
# 1. Coverage statistics
# ---------------------------------------------------------------------

coverage_by_decade = coverage_table("decade", "coverage_by_decade.csv")
coverage_by_decade = sort_decade_df(coverage_by_decade)
coverage_by_decade.to_csv(summary_dir / "coverage_by_decade.csv", index=False)

coverage_by_scale = coverage_table("scale_label", "coverage_by_scale.csv")
coverage_by_scale = sort_scale_df(coverage_by_scale)
coverage_by_scale.to_csv(summary_dir / "coverage_by_scale.csv", index=False)

coverage_by_product_group = coverage_table("product_group", "coverage_by_product_group.csv")
coverage_by_product_group.to_csv(summary_dir / "coverage_by_product_group.csv", index=False)

coverage_by_scale_decade = (
    gdf_area.groupby(["scale_label", "decade"], dropna=False)
    .size()
    .reset_index(name="n_map_editions")
)
coverage_by_scale_decade.to_csv(summary_dir / "counts_by_scale_decade_extra.csv", index=False)

save_bar(
    coverage_by_decade,
    "decade",
    "coverage_pct_of_la_metro",
    "LA metro topo-map footprint coverage by decade",
    "Decade",
    "Coverage of LA metro area (%)",
    "coverage_by_decade.png",
)

save_bar(
    coverage_by_scale,
    "scale_label",
    "coverage_pct_of_la_metro",
    "LA metro topo-map footprint coverage by scale",
    "Scale",
    "Coverage of LA metro area (%)",
    "coverage_by_scale.png",
)

save_bar(
    coverage_by_product_group,
    "product_group",
    "coverage_pct_of_la_metro",
    "LA metro topo-map footprint coverage by product group",
    "Product group",
    "Coverage of LA metro area (%)",
    "coverage_by_product_group.png",
)

# ---------------------------------------------------------------------
# 2. Low-overlap statistics
# ---------------------------------------------------------------------

def overlap_bin(x):
    if pd.isna(x):
        return "missing"
    if x < 1:
        return "<1%"
    if x < 5:
        return "1–5%"
    if x < 10:
        return "5–10%"
    if x < 25:
        return "10–25%"
    if x < 50:
        return "25–50%"
    if x < 75:
        return "50–75%"
    if x < 100:
        return "75–100%"
    return "100%"

gdf["overlap_bin"] = gdf["overlap_pct_of_map"].apply(overlap_bin)

overlap_order = ["<1%", "1–5%", "5–10%", "10–25%", "25–50%", "50–75%", "75–100%", "100%", "missing"]

low_overlap_summary = (
    gdf.groupby("overlap_bin", dropna=False)
    .size()
    .reset_index(name="n_map_editions")
)
low_overlap_summary["overlap_bin"] = pd.Categorical(
    low_overlap_summary["overlap_bin"], categories=overlap_order, ordered=True
)
low_overlap_summary = low_overlap_summary.sort_values("overlap_bin")
low_overlap_summary.to_csv(summary_dir / "low_overlap_summary.csv", index=False)

low_overlap_maps = gdf[
    gdf["overlap_pct_of_map"].notna() & (gdf["overlap_pct_of_map"] < 25)
].copy()
low_overlap_cols = [
    "map_name_clean", "product_group", "scale_label", "year_clean",
    "overlap_pct_of_map", "preferred_format", "preferred_download_url",
]
low_overlap_cols = [c for c in low_overlap_cols if c in low_overlap_maps.columns]
low_overlap_maps[low_overlap_cols].sort_values("overlap_pct_of_map").to_csv(
    summary_dir / "low_overlap_maps_under_25pct.csv", index=False
)

save_bar(
    low_overlap_summary,
    "overlap_bin",
    "n_map_editions",
    "Map editions by overlap with LA metro boundary",
    "Overlap bin",
    "Number of map editions",
    "low_overlap_distribution.png",
    rotate=False,
)

# ---------------------------------------------------------------------
# 3. Temporal depth by quadrangle
# ---------------------------------------------------------------------

quad = (
    gdf.groupby(["map_name_clean", "scale_label"], dropna=False)
    .agg(
        n_editions=("edition_key", "count"),
        earliest_year=("year_clean", "min"),
        latest_year=("year_clean", "max"),
        n_decades=("decade", lambda x: x.dropna().nunique()),
        product_groups=("product_group", lambda x: ", ".join(sorted(set(x.dropna().astype(str))))),
        formats=("available_formats", lambda x: ", ".join(sorted(set(", ".join(x.dropna().astype(str)).split(", "))))),
        max_overlap_pct=("overlap_pct_of_map", "max"),
    )
    .reset_index()
)
quad["year_span"] = quad["latest_year"] - quad["earliest_year"]
quad = quad.sort_values(["n_editions", "year_span", "map_name_clean"], ascending=[False, False, True])
quad.to_csv(summary_dir / "quadrangle_temporal_depth.csv", index=False)

top_quad = quad.head(30).sort_values("n_editions")
if not top_quad.empty:
    ax = top_quad.plot(kind="barh", x="map_name_clean", y="n_editions", figsize=(9, 8), legend=False)
    ax.set_title("Top quadrangles by number of topo map editions")
    ax.set_xlabel("Number of map editions")
    ax.set_ylabel("Quadrangle")
    plt.tight_layout()
    plt.savefig(plots_dir / "quadrangle_temporal_depth_top30.png", dpi=220)
    plt.close()

# ---------------------------------------------------------------------
# 4. Format availability
# ---------------------------------------------------------------------

def has_format(s, token):
    return token.lower() in str(s).lower()

format_rows = []
for token in ["GeoTIFF", "GeoPDF", "PDF", "JPEG", "ZIP", "unknown"]:
    if token == "PDF":
        mask = gdf["available_formats"].apply(lambda x: "pdf" in str(x).lower())
        label = "Any PDF/GeoPDF"
    elif token == "ZIP":
        mask = gdf["available_formats"].apply(lambda x: "zip" in str(x).lower())
        label = "ZIP/unknown"
    else:
        mask = gdf["available_formats"].apply(lambda x: has_format(x, token))
        label = token
    format_rows.append({"format_category": label, "n_map_editions": int(mask.sum())})

format_availability = pd.DataFrame(format_rows).drop_duplicates("format_category")
format_availability.to_csv(summary_dir / "format_availability.csv", index=False)

save_bar(
    format_availability,
    "format_category",
    "n_map_editions",
    "Download format availability",
    "Format",
    "Number of map editions",
    "format_availability.png",
)

# ---------------------------------------------------------------------
# 5. Candidate processing priority
# ---------------------------------------------------------------------

priority = gdf.copy()

def priority_score(row):
    score = 0

    # Prefer detailed maps.
    scale = row.get("scale_clean")
    if pd.notna(scale):
        scale = int(scale)
        if scale <= 24000:
            score += 4
        elif scale <= 62500:
            score += 3
        elif scale <= 100000:
            score += 2
        else:
            score += 1

    # Prefer maps substantially intersecting the metro area.
    ov = row.get("overlap_pct_of_map")
    if pd.notna(ov):
        if ov >= 75:
            score += 4
        elif ov >= 50:
            score += 3
        elif ov >= 25:
            score += 2
        elif ov >= 10:
            score += 1

    # Prefer historically useful maps.
    year = row.get("year_clean")
    if pd.notna(year):
        if year < 1950:
            score += 3
        elif year < 1980:
            score += 2
        elif year < 2000:
            score += 1

    # Prefer easier raster formats.
    fmt = str(row.get("available_formats", "")).lower()
    if "geotiff" in fmt:
        score += 3
    elif "pdf" in fmt:
        score += 2
    elif "jpeg" in fmt:
        score += 1

    return score

priority["processing_priority_score"] = priority.apply(priority_score, axis=1)

def priority_bin(score):
    if score >= 12:
        return "very high"
    if score >= 9:
        return "high"
    if score >= 6:
        return "medium"
    return "low"

priority["processing_priority_bin"] = priority["processing_priority_score"].apply(priority_bin)

priority_cols = [
    "processing_priority_bin", "processing_priority_score",
    "map_name_clean", "product_group", "scale_label", "year_clean",
    "overlap_pct_of_map", "available_formats", "preferred_format",
    "preferred_download_url",
]
priority_cols = [c for c in priority_cols if c in priority.columns]

priority[priority_cols].sort_values(
    ["processing_priority_score", "overlap_pct_of_map", "year_clean"],
    ascending=[False, False, True],
).to_csv(summary_dir / "candidate_processing_priority.csv", index=False)

priority_summary = (
    priority.groupby("processing_priority_bin")
    .size()
    .reset_index(name="n_map_editions")
)
priority_summary["processing_priority_bin"] = pd.Categorical(
    priority_summary["processing_priority_bin"],
    categories=["very high", "high", "medium", "low"],
    ordered=True,
)
priority_summary = priority_summary.sort_values("processing_priority_bin")
priority_summary.to_csv(summary_dir / "candidate_processing_priority_summary.csv", index=False)

save_bar(
    priority_summary,
    "processing_priority_bin",
    "n_map_editions",
    "Candidate map editions by processing priority",
    "Priority bin",
    "Number of map editions",
    "candidate_processing_priority.png",
    rotate=False,
)

# ---------------------------------------------------------------------
# 6. Spatial overview maps: earliest/latest/density
# ---------------------------------------------------------------------

# Use map footprints; these are not exact union coverage rasters but useful visual summaries.
map_gdf = gdf.to_crs(AREA_CRS)
boundary_plot = boundary.to_crs(AREA_CRS)

for col, title, out_png in [
    ("year_clean", "Topo map footprints by edition year", "map_footprints_by_year.png"),
    ("scale_clean", "Topo map footprints by scale denominator", "map_footprints_by_scale.png"),
    ("overlap_pct_of_map", "Topo map footprints by overlap with LA metro", "map_footprints_by_overlap.png"),
]:
    fig, ax = plt.subplots(figsize=(8, 8))
    boundary_plot.boundary.plot(ax=ax, linewidth=1.5)
    map_gdf.plot(ax=ax, column=col, legend=True, alpha=0.35, linewidth=0.2)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(plots_dir / out_png, dpi=220)
    plt.close()

# ---------------------------------------------------------------------
# 7. One-page text summary
# ---------------------------------------------------------------------

overall = {
    "n_map_editions": int(len(gdf)),
    "n_unique_quadrangle_names": int(gdf["map_name_clean"].nunique(dropna=True)),
    "min_year": int(gdf["year_clean"].min()),
    "max_year": int(gdf["year_clean"].max()),
    "la_metro_area_km2": float(boundary_area_km2),
    "all_maps_union_coverage_km2": float(union_area_km2(gdf_area["geometry_intersection"])),
}
overall["all_maps_union_coverage_pct"] = 100 * overall["all_maps_union_coverage_km2"] / boundary_area_km2
pd.DataFrame([overall]).to_csv(summary_dir / "expanded_overall_spatial_summary.csv", index=False)

readme = []
readme.append("# Expanded LA Metro Topo Statistics")
readme.append("")
readme.append(f"- Map editions: **{overall['n_map_editions']:,}**")
readme.append(f"- Unique quadrangle names: **{overall['n_unique_quadrangle_names']:,}**")
readme.append(f"- Year range: **{overall['min_year']}–{overall['max_year']}**")
readme.append(f"- LA metro area: **{overall['la_metro_area_km2']:.1f} km²**")
readme.append(f"- Union footprint coverage by all retained maps: **{overall['all_maps_union_coverage_pct']:.1f}%**")
readme.append("")
readme.append("New tables added to `summary_tables/`:")
readme.append("")
for name in [
    "coverage_by_decade.csv",
    "coverage_by_scale.csv",
    "coverage_by_product_group.csv",
    "low_overlap_summary.csv",
    "low_overlap_maps_under_25pct.csv",
    "quadrangle_temporal_depth.csv",
    "format_availability.csv",
    "candidate_processing_priority.csv",
    "candidate_processing_priority_summary.csv",
    "expanded_overall_spatial_summary.csv",
]:
    readme.append(f"- `{name}`")
readme.append("")
readme.append("New plots added to `plots/`:")
readme.append("")
for name in [
    "coverage_by_decade.png",
    "coverage_by_scale.png",
    "coverage_by_product_group.png",
    "low_overlap_distribution.png",
    "quadrangle_temporal_depth_top30.png",
    "format_availability.png",
    "candidate_processing_priority.png",
    "map_footprints_by_year.png",
    "map_footprints_by_scale.png",
    "map_footprints_by_overlap.png",
]:
    readme.append(f"- `{name}`")

(out_dir / "EXPANDED_STATS_README.md").write_text("\n".join(readme))

print("Expanded statistics complete.")
print(f"Map editions: {overall['n_map_editions']:,}")
print(f"Unique quadrangle names: {overall['n_unique_quadrangle_names']:,}")
print(f"Year range: {overall['min_year']}–{overall['max_year']}")
print(f"All-map union coverage: {overall['all_maps_union_coverage_pct']:.1f}%")
print(f"Tables written to: {summary_dir}")
print(f"Plots written to: {plots_dir}")
PY

python "$TMP_PY" "$OUT_DIR"
rm -f "$TMP_PY"

echo ""
echo "Done. Updated:"
echo "  ${SUMMARY_DIR}"
echo "  ${PLOTS_DIR}"
echo "  ${OUT_DIR}/EXPANDED_STATS_README.md"
