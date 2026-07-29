#!/usr/bin/env python3
"""
Collect metadata for all USGS topo map products intersecting the
Los Angeles-Long Beach-Anaheim, CA metropolitan area.

Main source:
  TNMAccess API:
  https://tnmaccess.nationalmap.gov/api/v1/products

Backup source:
  topoView ArcGIS overlay service:
  https://ngmdb.usgs.gov/topoview/
  https://ngmdb.usgs.gov/arcgis/rest/services/topoview/ustOverlay/MapServer/0/query

Outputs:
  - CSV metadata spreadsheet
  - XLSX metadata spreadsheet
  - GeoPackage with map footprints
  - summary CSV tables
  - plots for scale/year/decade/source coverage
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.ops import unary_union


TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASETS_URL = "https://tnmaccess.nationalmap.gov/api/v1/datasets"

TOPOVIEW_QUERY_URL = (
    "https://ngmdb.usgs.gov/arcgis/rest/services/topoview/"
    "ustOverlay/MapServer/0/query"
)

CENSUS_CBSA_ZIP_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/CBSA/tl_{year}_us_cbsa.zip"
)

DEFAULT_CBSA_GEOID = "31080"  # Los Angeles-Long Beach-Anaheim, CA
DEFAULT_OUT_DIR = "la_metro_topo_metadata"

# Useful metadata only: enough for dataset statistics and later download planning.
OUTPUT_COLUMNS = [
    "source_system",
    "record_uid",
    "title",
    "map_name",
    "dataset",
    "series",
    "scale",
    "year",
    "date_on_map",
    "imprint_year",
    "primary_state",
    "map_state",
    "datum",
    "format",
    "agency",
    "publisher",
    "download_url",
    "metadata_url",
    "thumbnail_url",
    "scan_id",
    "gda_item_id",
    "tnm_id",
    "byte_count",
    "md5",
    "bbox_west",
    "bbox_south",
    "bbox_east",
    "bbox_north",
    "footprint_area_km2",
    "overlap_area_km2",
    "overlap_pct_of_map",
]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "UMN-research-map-metadata-collector/1.0 "
                "(metadata only; contact: research script)"
            )
        }
    )
    return s


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
    sleep_seconds: float = 1.5,
) -> dict[str, Any]:
    last_error = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Request failed after {max_retries} tries: {url}") from last_error


def download_file(session: requests.Session, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    with session.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_cbsa_boundary(
    session: requests.Session,
    cache_dir: Path,
    cbsa_year: int = 2025,
    cbsa_geoid: str = DEFAULT_CBSA_GEOID,
) -> gpd.GeoDataFrame:
    """
    Load official CBSA boundary from Census TIGER/Line.
    Returns one-row GeoDataFrame in EPSG:4326.
    """
    zip_url = CENSUS_CBSA_ZIP_URL_TEMPLATE.format(year=cbsa_year)
    zip_path = cache_dir / f"tl_{cbsa_year}_us_cbsa.zip"

    print(f"Downloading/loading CBSA boundary: {zip_url}")
    download_file(session, zip_url, zip_path)

    # GeoPandas can read a local zipped shapefile directly.
    gdf = gpd.read_file(f"zip://{zip_path}").to_crs("EPSG:4326")

    geoid_col = "GEOID" if "GEOID" in gdf.columns else "CBSAFP"
    la = gdf[gdf[geoid_col].astype(str) == str(cbsa_geoid)].copy()

    if la.empty:
        available_names = gdf[gdf.astype(str).apply(
            lambda row: row.str.contains("Los Angeles", case=False, na=False).any(),
            axis=1,
        )]
        raise ValueError(
            f"Could not find CBSA GEOID={cbsa_geoid}. "
            f"Los Angeles-like rows found:\n{available_names.head(10)}"
        )

    la["boundary_source"] = f"Census TIGER/Line CBSA {cbsa_year}"
    return la


def polygon_to_esri_geometry(geom: Polygon | MultiPolygon) -> dict[str, Any]:
    """
    Convert Shapely Polygon/MultiPolygon to ESRI geometry JSON.
    ArcGIS accepts rings in lon/lat when inSR=4326.
    """
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        polygons = list(unary_union(geom).geoms)

    rings = []
    for poly in polygons:
        exterior = list(poly.exterior.coords)
        rings.append([[float(x), float(y)] for x, y in exterior])

        for interior in poly.interiors:
            ring = list(interior.coords)
            rings.append([[float(x), float(y)] for x, y in ring])

    return {"rings": rings, "spatialReference": {"wkid": 4326}}


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_nonempty(*values: Any) -> Any:
    for v in values:
        if v not in [None, "", [], {}]:
            return v
    return None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return None


def parse_year_from_text(text: str | None) -> int | None:
    if not text:
        return None
    # Prefer plausible topo map years.
    years = re.findall(r"\b(18[8-9]\d|19\d{2}|20\d{2})\b", str(text))
    if not years:
        return None
    return int(years[0])


def normalize_scale(value: Any) -> int | None:
    """
    Convert values such as 24000, '1:24,000', '24000' to integer denominator.
    """
    if value is None or value == "":
        return None
    text = str(value)
    if ":" in text:
        text = text.split(":")[-1]
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    return int(digits)


def extract_url(item: dict[str, Any], kind: str) -> str | None:
    """
    TNM/ScienceBase style records often store links in webLinks.
    """
    direct_keys = {
        "download": ["downloadURL", "downloadUrl", "url"],
        "metadata": ["metadataUrl", "metadataURL", "metaUrl"],
        "thumbnail": ["thumbnailURL", "thumbnailUrl"],
    }
    for key in direct_keys.get(kind, []):
        if item.get(key):
            return item[key]

    links = item.get("webLinks") or item.get("links") or []
    if not isinstance(links, list):
        return None

    kind_terms = {
        "download": ["download", "geotiff", "geopdf", "file"],
        "metadata": ["metadata", "xml"],
        "thumbnail": ["thumbnail", "preview", "browse"],
    }

    for link in links:
        if not isinstance(link, dict):
            continue
        uri = first_nonempty(link.get("uri"), link.get("url"), link.get("href"))
        label = " ".join(
            str(link.get(k, "")) for k in ["type", "title", "rel", "name"]
        ).lower()
        if uri and any(term in label for term in kind_terms[kind]):
            return uri

    return None


def geometry_from_tnm_item(item: dict[str, Any]) -> Polygon | MultiPolygon | None:
    """
    Try several common TNM/ScienceBase geometry forms.
    """
    # 1) GeoJSON-like geometry.
    for key in ["geometry", "spatial"]:
        candidate = item.get(key)
        if isinstance(candidate, dict) and candidate.get("type") in {
            "Polygon",
            "MultiPolygon",
            "GeometryCollection",
        }:
            try:
                return shape(candidate)
            except Exception:
                pass

    # 2) ScienceBase spatial.boundingBox.
    bbox_obj = safe_get(item, "spatial", "boundingBox")
    if isinstance(bbox_obj, dict):
        minx = first_nonempty(bbox_obj.get("minX"), bbox_obj.get("westBoundLongitude"))
        miny = first_nonempty(bbox_obj.get("minY"), bbox_obj.get("southBoundLatitude"))
        maxx = first_nonempty(bbox_obj.get("maxX"), bbox_obj.get("eastBoundLongitude"))
        maxy = first_nonempty(bbox_obj.get("maxY"), bbox_obj.get("northBoundLatitude"))
        try:
            return box(float(minx), float(miny), float(maxx), float(maxy))
        except Exception:
            pass

    # 3) Top-level bbox/boundingBox.
    for key in ["bbox", "boundingBox", "extent"]:
        b = item.get(key)
        if isinstance(b, dict):
            try:
                minx = first_nonempty(b.get("xmin"), b.get("minX"), b.get("west"))
                miny = first_nonempty(b.get("ymin"), b.get("minY"), b.get("south"))
                maxx = first_nonempty(b.get("xmax"), b.get("maxX"), b.get("east"))
                maxy = first_nonempty(b.get("ymax"), b.get("maxY"), b.get("north"))
                return box(float(minx), float(miny), float(maxx), float(maxy))
            except Exception:
                pass
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                return box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            except Exception:
                pass

    return None


def discover_tnm_topo_datasets(session: requests.Session) -> list[str]:
    """
    Discover TNM topo/topographic dataset tags.

    TNM /datasets currently returns a top-level list, not always a dict.
    We collect both parent dataset tags and nested tag names.
    """
    fallback = [
        "US Topo",
        "US Topo Current",
        "US Topo Historical",
        "Historical Topographic Maps",
    ]

    try:
        data = request_json(
            session,
            TNM_DATASETS_URL,
            params={"outputFormat": "JSON"},
            max_retries=3,
        )
    except Exception:
        print("Could not query TNM /datasets. Using fallback dataset names.")
        return fallback

    if isinstance(data, list):
        datasets = data
    elif isinstance(data, dict):
        datasets = data.get("items") or data.get("datasets") or []
    else:
        datasets = []

    names = []

    for d in datasets:
        if not isinstance(d, dict):
            continue

        title = first_nonempty(d.get("title"), d.get("name"), d.get("dataset"))
        tag = first_nonempty(d.get("sbDatasetTag"), title)

        text = " ".join(
            str(x)
            for x in [
                title,
                tag,
                d.get("description"),
                d.get("id"),
                d.get("parentCategory"),
            ]
            if x
        )

        if re.search(r"\btopo\b|topographic|HTMC|US Topo", text, flags=re.I):
            if tag:
                names.append(str(tag))
            elif title:
                names.append(str(title))

        # US Topo has nested tags like US Topo Current / US Topo Historical.
        for t in d.get("tags", []) or []:
            if not isinstance(t, dict):
                continue

            tag_title = first_nonempty(t.get("sbDatasetTag"), t.get("title"))
            tag_text = " ".join(
                str(x)
                for x in [
                    t.get("title"),
                    t.get("sbDatasetTag"),
                    t.get("id"),
                    t.get("description"),
                ]
                if x
            )

            if tag_title and re.search(
                r"\btopo\b|topographic|HTMC|US Topo", tag_text, flags=re.I
            ):
                names.append(str(tag_title))

    names = list(dict.fromkeys(names))
    return names or fallback


def query_tnm_products_for_dataset(
    session: requests.Session,
    bbox_wgs84: tuple[float, float, float, float],
    dataset_name: str,
    max_per_page: int = 100,
) -> list[dict[str, Any]]:
    """
    Query TNMAccess by bounding box and one dataset name.
    Later we spatially filter against the real CBSA polygon.
    """
    all_items = []
    offset = 0
    bbox_str = ",".join(f"{v:.8f}" for v in bbox_wgs84)

    while True:
        params = {
            "bbox": bbox_str,
            "extentType": "bbox",
            "datasets": dataset_name,
            "outputFormat": "JSON",
            "max": max_per_page,
            "offset": offset,
        }

        data = request_json(session, TNM_PRODUCTS_URL, params=params)
        items = data.get("items", [])
        total = data.get("total", None)

        if not items:
            break

        for item in items:
            item["_queried_dataset"] = dataset_name

        all_items.extend(items)
        print(f"TNM {dataset_name}: collected {len(all_items)} / {total}")

        offset += len(items)
        if total is not None and offset >= int(total):
            break
        if len(items) < max_per_page:
            break

        time.sleep(0.25)

    return all_items


def normalize_tnm_items(
    items: list[dict[str, Any]],
    la_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    la_geom = la_boundary.geometry.iloc[0]
    rows = []

    for item in items:
        geom = geometry_from_tnm_item(item)

        # If no precise geometry exists, keep bbox-derived records only if available.
        if geom is None:
            continue

        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            pass

        if geom is None or geom.is_empty or not geom.intersects(la_geom):
            continue

        overlap = geom.intersection(la_geom)
        item_area_km2 = (
            gpd.GeoSeries([geom], crs="EPSG:4326")
            .to_crs("EPSG:3310")
            .area.iloc[0]
            / 1_000_000
        )
        overlap_km2 = (
            gpd.GeoSeries([overlap], crs="EPSG:4326")
            .to_crs("EPSG:3310")
            .area.iloc[0]
            / 1_000_000
        )

        minx, miny, maxx, maxy = geom.bounds

        title = first_nonempty(item.get("title"), item.get("name"))
        scale = normalize_scale(
            first_nonempty(
                item.get("scale"),
                item.get("mapScale"),
                item.get("map_scale"),
                item.get("scaleDenominator"),
                title,
            )
        )
        year = first_nonempty(
            parse_int(item.get("year")),
            parse_int(item.get("publicationYear")),
            parse_year_from_text(title),
            parse_year_from_text(item.get("summary")),
        )

        record_uid = first_nonempty(
            item.get("sourceId"),
            item.get("id"),
            item.get("identifier"),
            extract_url(item, "download"),
            title,
        )

        rows.append(
            {
                "source_system": "TNMAccess",
                "record_uid": f"tnm:{record_uid}",
                "title": title,
                "map_name": first_nonempty(item.get("mapName"), item.get("map_name")),
                "dataset": item.get("_queried_dataset"),
                "series": first_nonempty(item.get("series"), item.get("productType")),
                "scale": scale,
                "year": year,
                "date_on_map": None,
                "imprint_year": None,
                "primary_state": first_nonempty(item.get("state"), item.get("primaryState")),
                "map_state": None,
                "datum": item.get("datum"),
                "format": first_nonempty(item.get("format"), item.get("prodFormat")),
                "agency": first_nonempty(item.get("agency"), item.get("sourceAgency")),
                "publisher": first_nonempty(item.get("publisher"), item.get("sourceInfo")),
                "download_url": extract_url(item, "download"),
                "metadata_url": extract_url(item, "metadata"),
                "thumbnail_url": extract_url(item, "thumbnail"),
                "scan_id": None,
                "gda_item_id": None,
                "tnm_id": first_nonempty(item.get("id"), item.get("sourceId")),
                "byte_count": parse_int(first_nonempty(item.get("sizeInBytes"), item.get("bytes"))),
                "md5": item.get("md5"),
                "bbox_west": minx,
                "bbox_south": miny,
                "bbox_east": maxx,
                "bbox_north": maxy,
                "footprint_area_km2": item_area_km2,
                "overlap_area_km2": overlap_km2,
                "overlap_pct_of_map": (
                    100 * overlap_km2 / item_area_km2 if item_area_km2 > 0 else None
                ),
                "geometry": geom,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=OUTPUT_COLUMNS + ["geometry"], crs="EPSG:4326")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def query_topoview_backup(
    session: requests.Session,
    la_boundary: gpd.GeoDataFrame,
    page_size: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Query topoView ArcGIS overlay directly with the CBSA polygon.
    This is especially useful for historical USGS topo maps.
    """
    la_geom = la_boundary.geometry.iloc[0]
    esri_geom = polygon_to_esri_geometry(la_geom)

    rows = []
    offset = 0

    out_fields = ",".join(
        [
            "OBJECTID",
            "map_scale",
            "map_name",
            "primary_state",
            "date_on_map",
            "imprint_year",
            "scan_id",
            "drg_name",
            "datum",
            "series",
            "byte_count",
            "md5",
            "gda_item_id",
            "map_state",
        ]
    )

    while True:
        params = {
            "f": "geojson",
            "where": "1=1",
            "geometry": json.dumps(esri_geom),
            "geometryType": "esriGeometryPolygon",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
            "outSR": 4326,
            "outFields": out_fields,
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": "date_on_map ASC, map_scale ASC",
        }

        data = request_json(session, TOPOVIEW_QUERY_URL, params=params)
        features = data.get("features", [])

        if not features:
            break

        print(f"topoView: collected {offset + len(features)} records")

        for feat in features:
            props = feat.get("properties", {})
            geom = shape(feat["geometry"])

            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty or not geom.intersects(la_geom):
                continue

            overlap = geom.intersection(la_geom)
            item_area_km2 = (
                gpd.GeoSeries([geom], crs="EPSG:4326")
                .to_crs("EPSG:3310")
                .area.iloc[0]
                / 1_000_000
            )
            overlap_km2 = (
                gpd.GeoSeries([overlap], crs="EPSG:4326")
                .to_crs("EPSG:3310")
                .area.iloc[0]
                / 1_000_000
            )

            minx, miny, maxx, maxy = geom.bounds
            date_on_map = parse_int(props.get("date_on_map"))
            imprint_year = parse_int(props.get("imprint_year"))
            year = first_nonempty(date_on_map, imprint_year)

            scan_id = props.get("scan_id")
            gda_item_id = props.get("gda_item_id")
            title = first_nonempty(props.get("drg_name"), props.get("map_name"))

            rows.append(
                {
                    "source_system": "topoView",
                    "record_uid": f"topoview:{scan_id or gda_item_id or props.get('OBJECTID')}",
                    "title": title,
                    "map_name": props.get("map_name"),
                    "dataset": "Historical Topographic Map Collection",
                    "series": props.get("series"),
                    "scale": normalize_scale(props.get("map_scale")),
                    "year": year,
                    "date_on_map": date_on_map,
                    "imprint_year": imprint_year,
                    "primary_state": props.get("primary_state"),
                    "map_state": props.get("map_state"),
                    "datum": props.get("datum"),
                    "format": None,
                    "agency": "USGS",
                    "publisher": "USGS",
                    "download_url": None,
                    "metadata_url": None,
                    "thumbnail_url": None,
                    "scan_id": scan_id,
                    "gda_item_id": gda_item_id,
                    "tnm_id": None,
                    "byte_count": parse_int(props.get("byte_count")),
                    "md5": props.get("md5"),
                    "bbox_west": minx,
                    "bbox_south": miny,
                    "bbox_east": maxx,
                    "bbox_north": maxy,
                    "footprint_area_km2": item_area_km2,
                    "overlap_area_km2": overlap_km2,
                    "overlap_pct_of_map": (
                        100 * overlap_km2 / item_area_km2 if item_area_km2 > 0 else None
                    ),
                    "geometry": geom,
                }
            )

        offset += len(features)
        if len(features) < page_size:
            break

        time.sleep(0.25)

    if not rows:
        return gpd.GeoDataFrame(columns=OUTPUT_COLUMNS + ["geometry"], crs="EPSG:4326")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def clean_and_deduplicate(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    for col in OUTPUT_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None

    gdf = gdf[OUTPUT_COLUMNS + ["geometry"]].copy()

    # Stable string key for de-duplication across source systems.
    gdf["dedupe_key"] = (
        gdf["download_url"].fillna("").astype(str)
        + "|"
        + gdf["md5"].fillna("").astype(str)
        + "|"
        + gdf["map_name"].fillna("").astype(str).str.lower()
        + "|"
        + gdf["year"].fillna("").astype(str)
        + "|"
        + gdf["scale"].fillna("").astype(str)
    )

    # Prefer TNM when same download URL/md5 exists, otherwise keep all unique records.
    gdf["source_rank"] = gdf["source_system"].map({"TNMAccess": 0, "topoView": 1}).fillna(9)
    gdf = gdf.sort_values(["dedupe_key", "source_rank", "record_uid"])
    gdf = gdf.drop_duplicates(subset=["dedupe_key"], keep="first")

    return gdf.drop(columns=["dedupe_key", "source_rank"])


def add_summary_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf["decade"] = gdf["year"].apply(
        lambda y: int(math.floor(float(y) / 10) * 10)
        if pd.notna(y)
        else None
    )

    def scale_bucket(s):
        if pd.isna(s):
            return "unknown"
        s = int(s)
        if s <= 24000:
            return "large / detailed (<=1:24k)"
        if s <= 62500:
            return "medium (1:24k-1:62.5k)"
        if s <= 100000:
            return "small-medium (1:63k-1:100k)"
        if s <= 250000:
            return "small (1:100k-1:250k)"
        return "very small (>1:250k)"

    gdf["scale_bucket"] = gdf["scale"].apply(scale_bucket)
    return gdf


def write_outputs(gdf: gpd.GeoDataFrame, la_boundary: gpd.GeoDataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "summary_tables"
    plots_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    gdf = add_summary_fields(gdf)

    # Main metadata spreadsheet.
    csv_path = out_dir / "la_metro_topo_map_metadata.csv"
    xlsx_path = out_dir / "la_metro_topo_map_metadata.xlsx"
    gpkg_path = out_dir / "la_metro_topo_map_footprints.gpkg"

    table_df = pd.DataFrame(gdf.drop(columns="geometry"))
    table_df.to_csv(csv_path, index=False)
    table_df.to_excel(xlsx_path, index=False)

    gdf.to_file(gpkg_path, layer="map_footprints", driver="GPKG")
    la_boundary.to_file(gpkg_path, layer="la_metro_cbsa_boundary", driver="GPKG")

    # Summary tables.
    summaries = {
        "by_source": gdf.groupby("source_system").size().reset_index(name="n_maps"),
        "by_scale": gdf.groupby("scale", dropna=False).size().reset_index(name="n_maps"),
        "by_scale_bucket": gdf.groupby("scale_bucket", dropna=False).size().reset_index(name="n_maps"),
        "by_year": gdf.groupby("year", dropna=False).size().reset_index(name="n_maps"),
        "by_decade": gdf.groupby("decade", dropna=False).size().reset_index(name="n_maps"),
        "by_series": gdf.groupby("series", dropna=False).size().reset_index(name="n_maps"),
        "by_dataset": gdf.groupby("dataset", dropna=False).size().reset_index(name="n_maps"),
        "by_scale_decade": (
            gdf.groupby(["decade", "scale"], dropna=False)
            .size()
            .reset_index(name="n_maps")
        ),
    }

    for name, df in summaries.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)

    # Plots.
    make_plots(gdf, la_boundary, plots_dir)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {xlsx_path}")
    print(f"  {gpkg_path}")
    print(f"  {tables_dir}")
    print(f"  {plots_dir}")


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    ax = series.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_plots(gdf: gpd.GeoDataFrame, la_boundary: gpd.GeoDataFrame, plots_dir: Path) -> None:
    if gdf.empty:
        return

    # 1. Count by decade.
    decade_counts = gdf.dropna(subset=["decade"]).groupby("decade").size().sort_index()
    if not decade_counts.empty:
        plot_bar(
            decade_counts,
            "Topo map records by decade",
            "Decade",
            "Number of records",
            plots_dir / "records_by_decade.png",
        )

    # 2. Count by scale.
    scale_counts = gdf.dropna(subset=["scale"]).groupby("scale").size().sort_index()
    if not scale_counts.empty:
        scale_counts.index = [f"1:{int(s):,}" for s in scale_counts.index]
        plot_bar(
            scale_counts,
            "Topo map records by scale",
            "Scale",
            "Number of records",
            plots_dir / "records_by_scale.png",
        )

    # 3. Count by source system.
    source_counts = gdf.groupby("source_system").size().sort_values(ascending=False)
    plot_bar(
        source_counts,
        "Records by source system",
        "Source",
        "Number of records",
        plots_dir / "records_by_source.png",
    )

    # 4. Scale x decade heatmap.
    pivot = (
        gdf.dropna(subset=["decade", "scale"])
        .assign(scale_label=lambda x: x["scale"].apply(lambda s: f"1:{int(s):,}"))
        .pivot_table(index="scale_label", columns="decade", values="record_uid", aggfunc="count", fill_value=0)
    )

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title("Topo map record counts: scale × decade")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Scale")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if val:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, label="Number of records")
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_scale_by_decade.png", dpi=200)
        plt.close()

    # 5. Footprint map.
    fig, ax = plt.subplots(figsize=(8, 8))
    la_boundary.boundary.plot(ax=ax, linewidth=2)
    gdf.boundary.plot(ax=ax, linewidth=0.4, alpha=0.35)
    ax.set_title("Topo map footprints intersecting LA metro CBSA")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(plots_dir / "footprints_over_la_metro.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--cache", default=None, help="Cache directory")
    parser.add_argument("--cbsa-year", type=int, default=2025)
    parser.add_argument("--cbsa-geoid", default=DEFAULT_CBSA_GEOID)
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated TNM dataset names. "
            "If omitted, script tries to discover topo/topographic datasets."
        ),
    )
    parser.add_argument(
        "--skip-tnm",
        action="store_true",
        help="Skip TNMAccess and only use topoView backup.",
    )
    parser.add_argument(
        "--skip-topoview",
        action="store_true",
        help="Skip topoView backup.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    la_boundary = load_cbsa_boundary(
        session=session,
        cache_dir=cache_dir,
        cbsa_year=args.cbsa_year,
        cbsa_geoid=args.cbsa_geoid,
    )

    la_geom = la_boundary.geometry.iloc[0]
    bbox_wgs84 = tuple(float(v) for v in la_geom.bounds)
    print(f"LA metro bbox: {bbox_wgs84}")

    all_gdfs = []

    if not args.skip_tnm:
        if args.datasets:
            datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        else:
            datasets = discover_tnm_topo_datasets(session)

        print("\nTNM datasets to query:")
        for d in datasets:
            print(f"  - {d}")

        tnm_items = []
        for dataset in datasets:
            try:
                tnm_items.extend(
                    query_tnm_products_for_dataset(
                        session=session,
                        bbox_wgs84=bbox_wgs84,
                        dataset_name=dataset,
                    )
                )
            except Exception as e:
                print(f"WARNING: TNM query failed for dataset={dataset!r}: {e}")

        tnm_gdf = normalize_tnm_items(tnm_items, la_boundary)
        print(f"TNM records intersecting LA metro polygon: {len(tnm_gdf)}")
        all_gdfs.append(tnm_gdf)

    if not args.skip_topoview:
        try:
            topoview_gdf = query_topoview_backup(session, la_boundary)
            print(f"topoView records intersecting LA metro polygon: {len(topoview_gdf)}")
            all_gdfs.append(topoview_gdf)
        except Exception as e:
            print(f"WARNING: topoView backup query failed: {e}")

    if not all_gdfs:
        raise RuntimeError("No sources were queried.")

    combined = pd.concat(all_gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    combined = clean_and_deduplicate(combined)

    print(f"\nFinal de-duplicated record count: {len(combined)}")
    write_outputs(combined, la_boundary, out_dir)


if __name__ == "__main__":
    main()