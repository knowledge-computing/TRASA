#!/usr/bin/env python3
"""
Repair and re-analyze LA metropolitan topo metadata.

Input:
  la_metro_topo_metadata/
    la_metro_topo_map_metadata.csv
    la_metro_topo_map_footprints.gpkg   optional, but recommended

Output:
  la_metro_topo_metadata_cleaned/
    la_metro_topo_map_editions_clean.csv
    la_metro_topo_map_editions_clean.xlsx
    la_metro_topo_map_editions_clean.gpkg
    summary_tables/*.csv
    plots/*.png
    README.md

This script does NOT download maps.
It cleans the already-collected metadata table.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None


VALID_MAP_DATASETS = {
    "Historical Topographic Maps",
    "US Topo",
}

BAD_DATASET_KEYWORDS = [
    "National Elevation Dataset",
    "3D Hydrography",
    "Combined Vector",
    "Map Indices",
    "Contours",
    "NED",
]


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_name(x: str) -> str:
    x = safe_str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def parse_year_from_title(title: str | None) -> int | None:
    if not title:
        return None
    years = re.findall(r"\b(18[8-9]\d|19\d{2}|20\d{2})\b", safe_str(title))
    if not years:
        return None
    return int(years[-1])


def parse_year_from_url(url: str | None) -> int | None:
    """
    US Topo titles sometimes omit the year, but URLs contain dates like:
      CA_Sitton_Peak_20220103_TM_geo.pdf
    """
    text = safe_str(url)

    m = re.search(r"[_/-](20\d{6})(?=[_./-])", text)
    if m:
        return int(m.group(1)[:4])

    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return int(m.group(1))

    return None


def infer_format(title: str, url: str) -> str:
    text = f"{safe_str(title)} {safe_str(url)}".lower()

    if "geotiff" in text or ".tif" in text or ".tiff" in text:
        return "GeoTIFF"
    if "geopdf" in text or ".pdf" in text:
        return "GeoPDF/PDF"
    if "jpeg" in text or ".jpg" in text:
        return "JPEG"
    if "shapefile" in text or "shape" in text or ".shp" in text:
        return "Shapefile"
    if "filegdb" in text or "gdb" in text:
        return "FileGDB"
    if "gpkg" in text or "geopackage" in text:
        return "GeoPackage"
    if ".zip" in text:
        return "ZIP/unknown"
    return "unknown"


def parse_topo_title(title: str, dataset: str, url: str = "") -> dict:
    """
    Extract useful metadata from TNM product titles and URLs.

    Important cases:
      USGS 1:24000-scale Quadrangle for Frazier Mtn, CA 1958
      USGS US Topo 7.5-minute map for Anaheim, CA 2018
      US Topo 7.5-minute map for Sitton Peak, CA
        with URL CA_Sitton_Peak_20220103_TM_geo.pdf
    """
    title = normalize_name(title)
    url = safe_str(url)
    year_from_url = parse_year_from_url(url)

    out = {
        "map_name_parsed": None,
        "scale_parsed": None,
        "year_parsed": parse_year_from_title(title) or year_from_url,
        "state_parsed": None,
        "series_parsed": None,
        "product_group": None,
    }

    # Historical Topographic Map Collection.
    m = re.search(
        r"(?:USGS\s+)?1:(?P<scale>[\d,]+)-scale\s+Quadrangle\s+for\s+"
        r"(?P<name>.*?),\s*(?P<state>[A-Z]{2}|California)\s+"
        r"(?P<year>18[8-9]\d|19\d{2}|20\d{2})",
        title,
        flags=re.I,
    )
    if m:
        out["map_name_parsed"] = normalize_name(m.group("name"))
        out["scale_parsed"] = int(m.group("scale").replace(",", ""))
        out["year_parsed"] = int(m.group("year"))
        out["state_parsed"] = m.group("state")
        out["series_parsed"] = f"1:{out['scale_parsed']:,}"
        out["product_group"] = "Historical Topographic Maps"
        return out

    # US Topo title, with optional leading "USGS" and optional title year.
    m = re.search(
        r"^(?:USGS\s+)?US\s+Topo\s+(?P<minute>[\d.]+)-minute\s+map\s+for\s+"
        r"(?P<name>.*?),\s*(?P<state>[A-Z]{2}|California)"
        r"(?:\s+(?P<year>20\d{2}))?$",
        title,
        flags=re.I,
    )
    if m:
        minute = float(m.group("minute"))
        out["map_name_parsed"] = normalize_name(m.group("name"))
        out["scale_parsed"] = 24000 if abs(minute - 7.5) < 0.01 else None
        out["year_parsed"] = int(m.group("year")) if m.group("year") else year_from_url
        out["state_parsed"] = m.group("state")
        out["series_parsed"] = f"{minute:g}-minute US Topo"
        out["product_group"] = "US Topo"
        return out

    # US Topo URL fallback.
    # Example:
    # https://.../Maps/USTopo/PDF/CA/CA_Sitton_Peak_20220103_TM_geo.pdf
    m = re.search(
        r"/(?:PDF|GeoPDF|USTopo)/(?P<state1>[A-Z]{2})/"
        r"(?P<state2>[A-Z]{2})_(?P<name>.+?)_(?P<date>20\d{6})_TM_geo",
        url,
        flags=re.I,
    )
    if not m:
        m = re.search(
            r"/(?P<state2>[A-Z]{2})_(?P<name>.+?)_(?P<date>20\d{6})_TM_geo",
            url,
            flags=re.I,
        )

    if m and ("US Topo" in safe_str(dataset) or "USTopo" in url or "_TM_geo" in url):
        out["map_name_parsed"] = normalize_name(m.group("name").replace("_", " "))
        out["scale_parsed"] = 24000
        out["year_parsed"] = int(m.group("date")[:4])
        out["state_parsed"] = m.group("state2")
        out["series_parsed"] = "7.5-minute US Topo"
        out["product_group"] = "US Topo"
        return out

    # Fallback: keep the row, but mark what product group it belongs to.
    if dataset == "Historical Topographic Maps":
        out["product_group"] = "Historical Topographic Maps"
    elif dataset == "US Topo":
        out["product_group"] = "US Topo"

    return out


def scale_bucket(scale: float | int | None) -> str:
    if scale is None or pd.isna(scale):
        return "unknown"

    scale = int(scale)

    if scale <= 24000:
        return "<= 1:24k"
    if scale <= 62500:
        return "1:24k–1:62.5k"
    if scale <= 100000:
        return "1:62.5k–1:100k"
    if scale <= 250000:
        return "1:100k–1:250k"
    return "> 1:250k"


def scale_label(scale) -> str:
    if pd.isna(scale):
        return "unknown"
    return f"1:{int(scale):,}"


def choose_preferred_url(group: pd.DataFrame) -> str:
    """
    Prefer formats most useful for later raster processing.
    """
    priority = {
        "GeoTIFF": 0,
        "GeoPDF/PDF": 1,
        "JPEG": 2,
        "ZIP/unknown": 3,
        "GeoPackage": 4,
        "Shapefile": 5,
        "FileGDB": 6,
        "unknown": 9,
    }

    g = group.copy()
    g["_rank"] = g["format_clean"].map(priority).fillna(9)
    g = g.sort_values(["_rank", "download_url"])
    return safe_str(g["download_url"].iloc[0])


def choose_preferred_format(group: pd.DataFrame) -> str:
    priority = {
        "GeoTIFF": 0,
        "GeoPDF/PDF": 1,
        "JPEG": 2,
        "ZIP/unknown": 3,
        "GeoPackage": 4,
        "Shapefile": 5,
        "FileGDB": 6,
        "unknown": 9,
    }

    g = group.copy()
    g["_rank"] = g["format_clean"].map(priority).fillna(9)
    g = g.sort_values(["_rank", "download_url"])
    return safe_str(g["format_clean"].iloc[0])


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace({"": pd.NA})

    # Keep only actual topo map datasets.
    df["dataset"] = df["dataset"].astype("string")
    df = df[df["dataset"].isin(VALID_MAP_DATASETS)].copy()

    # Extra safety filter against accidentally included vector/elevation products.
    bad_re = "|".join(re.escape(x) for x in BAD_DATASET_KEYWORDS)
    bad_mask = (
        df["dataset"].fillna("").str.contains(bad_re, case=False, regex=True)
        | df["title"].fillna("").str.contains(
            r"Vector Data|Contours|NED|3D Hydrography|Map Index|Arc Second",
            case=False,
            regex=True,
        )
    )
    df = df[~bad_mask].copy()

    parsed = df.apply(
        lambda r: parse_topo_title(r.get("title"), r.get("dataset"), r.get("download_url")),
        axis=1,
        result_type="expand",
    )

    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    df["map_name_clean"] = df["map_name_parsed"].combine_first(df.get("map_name"))
    df["scale_clean"] = pd.to_numeric(df["scale_parsed"], errors="coerce")
    df["year_clean"] = pd.to_numeric(df["year_parsed"], errors="coerce")
    df["state_clean"] = df["state_parsed"].fillna(df.get("primary_state"))
    df["series_clean"] = df["series_parsed"]

    df["format_clean"] = df.apply(
        lambda r: infer_format(r.get("title"), r.get("download_url")),
        axis=1,
    )

    df["decade"] = df["year_clean"].apply(
        lambda y: int(y // 10 * 10) if pd.notna(y) else pd.NA
    )
    df["scale_bucket_clean"] = df["scale_clean"].apply(scale_bucket)
    df["scale_label"] = df["scale_clean"].apply(scale_label)

    # Product ID for one map edition. This intentionally ignores file format,
    # so GeoPDF/GeoTIFF duplicates collapse into one map-edition row.
    df["edition_key"] = (
        df["product_group"].fillna("unknown").astype(str)
        + "|"
        + df["map_name_clean"].fillna("unknown").astype(str).str.lower()
        + "|"
        + df["state_clean"].fillna("unknown").astype(str)
        + "|"
        + df["scale_clean"].fillna(-1).astype(int).astype(str)
        + "|"
        + df["year_clean"].fillna(-1).astype(int).astype(str)
        + "|"
        + df["bbox_west"].round(5).astype(str)
        + "|"
        + df["bbox_south"].round(5).astype(str)
        + "|"
        + df["bbox_east"].round(5).astype(str)
        + "|"
        + df["bbox_north"].round(5).astype(str)
    )

    return df


def aggregate_map_editions(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
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
        "bbox_west",
        "bbox_south",
        "bbox_east",
        "bbox_north",
        "footprint_area_km2",
        "overlap_area_km2",
        "overlap_pct_of_map",
    ]

    for c in group_cols:
        if c not in df.columns:
            df[c] = pd.NA

    rows = []

    for key, g in df.groupby("edition_key", dropna=False):
        first = g.iloc[0].to_dict()

        available_formats = sorted(set(g["format_clean"].dropna().astype(str)))
        urls_by_format = (
            g.dropna(subset=["download_url"])
            .groupby("format_clean")["download_url"]
            .apply(lambda x: " ; ".join(sorted(set(x.astype(str)))))
            .to_dict()
        )

        row = {c: first.get(c) for c in group_cols}
        row.update(
            {
                "n_file_records": len(g),
                "available_formats": ", ".join(available_formats),
                "preferred_format": choose_preferred_format(g),
                "preferred_download_url": choose_preferred_url(g),
                "all_download_urls": " ; ".join(
                    sorted(set(g["download_url"].dropna().astype(str)))
                ),
                "geotiff_urls": urls_by_format.get("GeoTIFF"),
                "geopdf_urls": urls_by_format.get("GeoPDF/PDF"),
                "source_records": " ; ".join(
                    sorted(set(g["record_uid"].dropna().astype(str)))
                ),
                "title_examples": " ; ".join(
                    sorted(set(g["title"].dropna().astype(str)))[:3]
                ),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["product_group", "scale_clean", "map_name_clean", "year_clean"],
        na_position="last",
    ).reset_index(drop=True)

    return out


def write_summary_tables(editions: pd.DataFrame, out_dir: Path) -> None:
    summary_dir = out_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame(
        [
            {
                "n_unique_map_editions": len(editions),
                "n_unique_quadrangle_names": editions["map_name_clean"].nunique(dropna=True),
                "min_year": editions["year_clean"].min(),
                "max_year": editions["year_clean"].max(),
                "n_missing_year": editions["year_clean"].isna().sum(),
                "n_missing_scale": editions["scale_clean"].isna().sum(),
            }
        ]
    )
    overall.to_csv(summary_dir / "overall.csv", index=False)

    editions.groupby("product_group", dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_product_group.csv", index=False)

    editions.groupby("scale_label", dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_scale.csv", index=False)

    editions.groupby("scale_bucket_clean", dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_scale_bucket.csv", index=False)

    editions.groupby("year_clean", dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_year.csv", index=False)

    editions.groupby("decade", dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_decade.csv", index=False)

    editions.groupby(["decade", "scale_label"], dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_decade_scale.csv", index=False)

    editions.groupby(["product_group", "decade"], dropna=False).size().reset_index(
        name="n_map_editions"
    ).to_csv(summary_dir / "by_product_group_decade.csv", index=False)

    editions.groupby(["map_name_clean", "scale_label"], dropna=False).agg(
        n_editions=("edition_key", "count"),
        min_year=("year_clean", "min"),
        max_year=("year_clean", "max"),
        product_groups=("product_group", lambda x: ", ".join(sorted(set(x.dropna())))),
    ).reset_index().sort_values(
        ["n_editions", "map_name_clean"], ascending=[False, True]
    ).to_csv(summary_dir / "by_quadrangle_time_span.csv", index=False)


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path):
    ax = series.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_plots(editions: pd.DataFrame, out_dir: Path, gdf=None, boundary=None) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_decade = editions.dropna(subset=["decade"]).groupby("decade").size().sort_index()
    if len(by_decade):
        plot_bar(
            by_decade,
            "LA metro topo map editions by decade",
            "Decade",
            "Number of map editions",
            plots_dir / "map_editions_by_decade.png",
        )

    by_scale = editions.dropna(subset=["scale_clean"]).groupby("scale_label").size()
    if len(by_scale):
        # sort by numeric scale, not alphabetically
        scale_order = (
            editions.dropna(subset=["scale_clean"])
            .drop_duplicates("scale_label")
            .sort_values("scale_clean")["scale_label"]
            .tolist()
        )
        by_scale = by_scale.reindex(scale_order)
        plot_bar(
            by_scale,
            "LA metro topo map editions by scale",
            "Scale",
            "Number of map editions",
            plots_dir / "map_editions_by_scale.png",
        )

    by_group_decade = (
        editions.dropna(subset=["decade"])
        .groupby(["decade", "product_group"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if len(by_group_decade):
        ax = by_group_decade.plot(kind="bar", stacked=True, figsize=(12, 5))
        ax.set_title("LA metro topo map editions by decade and product group")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Number of map editions")
        plt.tight_layout()
        plt.savefig(plots_dir / "map_editions_by_decade_product_group.png", dpi=220)
        plt.close()

    pivot = (
        editions.dropna(subset=["decade", "scale_clean"])
        .pivot_table(
            index="scale_label",
            columns="decade",
            values="edition_key",
            aggfunc="count",
            fill_value=0,
        )
    )
    if not pivot.empty:
        scale_order = (
            editions.dropna(subset=["scale_clean"])
            .drop_duplicates("scale_label")
            .sort_values("scale_clean")["scale_label"]
            .tolist()
        )
        pivot = pivot.reindex([x for x in scale_order if x in pivot.index])

        fig, ax = plt.subplots(figsize=(13, max(4, 0.42 * len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title("LA metro topo map editions: scale × decade")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Scale")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = int(pivot.values[i, j])
                if val:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, label="Number of map editions")
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_scale_by_decade_clean.png", dpi=220)
        plt.close()

    span = editions.groupby(["map_name_clean", "scale_label"], dropna=False).size()
    if len(span):
        edition_counts = span.value_counts().sort_index()
        plot_bar(
            edition_counts,
            "How many editions exist per quadrangle/scale?",
            "Number of editions for one quadrangle/scale",
            "Number of quadrangle-scale groups",
            plots_dir / "editions_per_quadrangle_scale.png",
        )

    top_quads = (
        editions.groupby("map_name_clean", dropna=True)
        .size()
        .sort_values(ascending=False)
        .head(30)
        .sort_values()
    )
    if len(top_quads):
        ax = top_quads.plot(kind="barh", figsize=(9, 8))
        ax.set_title("Top quadrangle names by number of map editions")
        ax.set_xlabel("Number of map editions")
        ax.set_ylabel("Quadrangle name")
        plt.tight_layout()
        plt.savefig(plots_dir / "top_quadrangles_by_editions.png", dpi=220)
        plt.close()

    if gdf is not None and boundary is not None and len(gdf):
        fig, ax = plt.subplots(figsize=(8, 8))
        boundary.boundary.plot(ax=ax, linewidth=2)
        gdf.boundary.plot(ax=ax, linewidth=0.35, alpha=0.35)
        ax.set_title("Clean topo map footprints intersecting LA metro CBSA")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(plots_dir / "clean_footprints_over_la_metro.png", dpi=220)
        plt.close()


def write_readme(editions: pd.DataFrame, out_dir: Path, original_n: int, cleaned_file_records_n: int):
    by_group = editions.groupby("product_group").size().sort_values(ascending=False)
    by_scale = editions.groupby("scale_label").size().sort_values(ascending=False)

    lines = []
    lines.append("# Cleaned LA Metro Topo Metadata")
    lines.append("")
    lines.append("This folder is produced by `repair_la_topo_analysis.py`.")
    lines.append("")
    lines.append("## What was fixed")
    lines.append("")
    lines.append(f"- Original collected rows: **{original_n:,}**")
    lines.append(f"- File/product records kept after removing non-map datasets: **{cleaned_file_records_n:,}**")
    lines.append(f"- Unique map editions after collapsing multiple file formats: **{len(editions):,}**")
    lines.append("- Removed NED contours, 3DHP, Combined Vector, and Map Indices.")
    lines.append("- Reconstructed `map_name`, `scale`, `year`, and `product_group` from TNM titles.")
    lines.append("- Collapsed duplicate file-format rows into one row per map edition.")
    lines.append("")
    lines.append("## Counts by product group")
    lines.append("")
    for k, v in by_group.items():
        lines.append(f"- {k}: **{v:,}**")
    lines.append("")
    lines.append("## Counts by scale")
    lines.append("")
    for k, v in by_scale.items():
        lines.append(f"- {k}: **{v:,}**")
    lines.append("")
    lines.append("## Main files")
    lines.append("")
    lines.append("- `la_metro_topo_map_editions_clean.csv`")
    lines.append("- `la_metro_topo_map_editions_clean.xlsx`")
    lines.append("- `la_metro_topo_map_editions_clean.gpkg`, if GeoPandas can read the original GeoPackage")
    lines.append("- `summary_tables/`")
    lines.append("- `plots/`")
    lines.append("")

    (out_dir / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default="la_metro_topo_metadata",
        help="Original metadata output directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="la_metro_topo_metadata_cleaned",
        help="Cleaned output directory.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = in_dir / "la_metro_topo_map_metadata.csv"
    gpkg_path = in_dir / "la_metro_topo_map_footprints.gpkg"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw = pd.read_csv(csv_path)
    original_n = len(raw)

    cleaned_records = clean_metadata(raw)
    cleaned_file_records_n = len(cleaned_records)

    editions = aggregate_map_editions(cleaned_records)

    csv_out = out_dir / "la_metro_topo_map_editions_clean.csv"
    xlsx_out = out_dir / "la_metro_topo_map_editions_clean.xlsx"

    editions.to_csv(csv_out, index=False)
    editions.to_excel(xlsx_out, index=False)

    # Optional geospatial output.
    gdf_editions = None
    boundary = None

    if gpd is not None and gpkg_path.exists():
        try:
            gdf_raw = gpd.read_file(gpkg_path, layer="map_footprints")
            boundary = gpd.read_file(gpkg_path, layer="la_metro_cbsa_boundary")

            # Same row order as CSV from original script.
            gdf_raw = gdf_raw.reset_index(drop=True)
            cleaned_with_geom = clean_metadata(
                pd.concat(
                    [
                        raw.reset_index(drop=True),
                        pd.DataFrame({"_orig_row": range(len(raw))}),
                    ],
                    axis=1,
                )
            )

            gdf_raw["_orig_row"] = range(len(gdf_raw))
            geom_lookup = gdf_raw[["_orig_row", "geometry"]]

            tmp = cleaned_with_geom.merge(geom_lookup, on="_orig_row", how="left")
            tmp_gdf = gpd.GeoDataFrame(tmp, geometry="geometry", crs=gdf_raw.crs)

            geom_by_key = (
                tmp_gdf.dropna(subset=["geometry"])
                .drop_duplicates("edition_key")[["edition_key", "geometry"]]
            )

            gdf_editions = editions.merge(geom_by_key, on="edition_key", how="left")
            gdf_editions = gpd.GeoDataFrame(
                gdf_editions, geometry="geometry", crs=gdf_raw.crs
            )

            gpkg_out = out_dir / "la_metro_topo_map_editions_clean.gpkg"
            gdf_editions.to_file(gpkg_out, layer="map_editions", driver="GPKG")
            boundary.to_file(gpkg_out, layer="la_metro_cbsa_boundary", driver="GPKG")

        except Exception as e:
            print(f"WARNING: Could not write cleaned GeoPackage: {e}")

    write_summary_tables(editions, out_dir)
    make_plots(editions, out_dir, gdf=gdf_editions, boundary=boundary)
    write_readme(editions, out_dir, original_n, cleaned_file_records_n)

    print()
    print("Done.")
    print(f"Original rows: {original_n:,}")
    print(f"Kept topo file/product records: {cleaned_file_records_n:,}")
    print(f"Unique map editions: {len(editions):,}")
    print(f"Saved cleaned outputs to: {out_dir}")


if __name__ == "__main__":
    main()