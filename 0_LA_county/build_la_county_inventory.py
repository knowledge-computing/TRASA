#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from shapely.geometry import box, mapping
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_METADATA_CSV = PROJECT_ROOT / "metadata" / "historicaltopo" / "historicaltopo.csv"
DEFAULT_INVENTORY_CSV = SCRIPT_DIR / "la_county_historical_inventory.csv"
DEFAULT_SUMMARY_TXT = SCRIPT_DIR / "la_county_historical_inventory_summary.txt"
DEFAULT_FOOTPRINTS_GEOJSON = SCRIPT_DIR / "la_county_historical_footprints.geojson"
DEFAULT_BOUNDARY_GEOJSON = SCRIPT_DIR / "la_county_historical_coverage_boundary.geojson"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "CA_area" / "maps" / "scanned" / "raw"
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "CA_area" / "outputs" / "scanned"

SCAN_STEM_RE = re.compile(r"^(?P<prefix>.+)_(?P<scan_id>\d+)_(?P<year>\d{4})_(?P<scale>\d+)_geo(?:_tif)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Los Angeles County historical topo inventory from "
            "metadata/historicaltopo/historicaltopo.csv and optionally "
            "download missing GeoTIFFs."
        )
    )
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--summary-txt", type=Path, default=DEFAULT_SUMMARY_TXT)
    parser.add_argument("--footprints-geojson", type=Path, default=DEFAULT_FOOTPRINTS_GEOJSON)
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_BOUNDARY_GEOJSON)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--county-name", default="Los Angeles")
    parser.add_argument("--state-name", default="California")
    parser.add_argument(
        "--filter-mode",
        choices=("county_list", "footprint_intersects"),
        default="county_list",
        help=(
            "county_list: keep rows whose county_list includes Los Angeles. "
            "footprint_intersects: first derive an approximate coverage boundary "
            "from county_list matches, then keep all California rows whose map "
            "footprint intersects that derived boundary."
        ),
    )
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--download-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser.parse_args()


def normalize_county_token(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def county_tokens(value: str) -> List[str]:
    if not value:
        return []
    return [normalize_county_token(part) for part in value.split(",") if part.strip()]


def scan_id_from_existing_name(name: str) -> Optional[str]:
    match = SCAN_STEM_RE.match(name)
    if not match:
        return None
    return match.group("scan_id")


def build_existing_scan_id_index(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not root.exists():
        return index

    for path in root.iterdir():
        if not path.is_dir():
            continue
        scan_id = scan_id_from_existing_name(path.name)
        if scan_id:
            index[scan_id] = path.name
    return index


def make_expected_tif_name(row: Dict[str, str]) -> str:
    url_name = unquote(Path(urlsplit(row["geotiff_url"]).path).name.strip())
    if url_name:
        return url_name
    product_filename = row["product_filename"].strip()
    if product_filename.lower().endswith(".pdf"):
        return product_filename[:-4] + ".tif"
    return product_filename + ".tif"


def make_expected_stem(row: Dict[str, str]) -> str:
    return Path(make_expected_tif_name(row)).stem


def make_expected_raw_subdir(row: Dict[str, str]) -> str:
    return f"{make_expected_stem(row)}_tif"


def geometry_from_row(row: Dict[str, str]):
    return box(
        float(row["westbc"]),
        float(row["southbc"]),
        float(row["eastbc"]),
        float(row["northbc"]),
    )


def load_matching_rows(metadata_csv: Path, county_name: str, state_name: str) -> List[Dict[str, str]]:
    county_name_norm = normalize_county_token(county_name)
    rows: List[Dict[str, str]] = []

    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("primary_state", "").strip() != state_name:
                continue
            tokens = county_tokens(row.get("county_list", ""))
            if county_name_norm in tokens:
                rows.append(row)

    return rows


def choose_boundary_rows(rows: Sequence[Dict[str, str]], county_name: str) -> List[Dict[str, str]]:
    if not rows:
        return []

    def area(row: Dict[str, str]) -> float:
        return geometry_from_row(row).area

    county_name_norm = normalize_county_token(county_name)
    exact_county_rows = [
        row for row in rows
        if county_tokens(row.get("county_list", "")) == [county_name_norm]
    ]

    fine_scale_rows = [
        row for row in (exact_county_rows or rows)
        if row.get("map_scale", "").isdigit() and int(row["map_scale"]) <= 24000
    ]
    candidate_rows = fine_scale_rows or list(exact_county_rows or rows)
    min_area = min(area(row) for row in candidate_rows)
    area_limit = min_area * 4.0 + 1e-12
    return [row for row in candidate_rows if area(row) <= area_limit]


def select_rows_by_coverage(
    metadata_csv: Path,
    state_name: str,
    coverage_geom,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("primary_state", "").strip() != state_name:
                continue
            if geometry_from_row(row).intersects(coverage_geom):
                rows.append(row)

    return rows


def annotate_rows(
    rows: Sequence[Dict[str, str]],
    existing_raw: Dict[str, str],
    existing_outputs: Dict[str, str],
) -> List[Dict[str, str]]:
    annotated: List[Dict[str, str]] = []

    for row in rows:
        scan_id = row["scan_id"].strip()
        expected_tif_name = make_expected_tif_name(row)
        expected_stem = make_expected_stem(row)
        expected_raw_subdir = make_expected_raw_subdir(row)
        raw_name = existing_raw.get(scan_id, "")
        output_name = existing_outputs.get(scan_id, "")

        annotated_row = dict(row)
        annotated_row.update(
            {
                "expected_tif_name": expected_tif_name,
                "expected_stem": expected_stem,
                "expected_raw_subdir": expected_raw_subdir,
                "raw_downloaded": "yes" if scan_id in existing_raw else "no",
                "processed_output_exists": "yes" if scan_id in existing_outputs else "no",
                "existing_raw_dirname": raw_name,
                "existing_output_dirname": output_name,
            }
        )
        annotated.append(annotated_row)

    annotated.sort(
        key=lambda row: (
            int(row["date_on_map"] or 0),
            int(row["map_scale"] or 0),
            row["map_name"],
            int(row["scan_id"]),
        )
    )
    return annotated


def write_inventory_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows selected; refusing to write an empty inventory.")

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(rows: Sequence[Dict[str, str]], footprints_path: Path, boundary_path: Path) -> Tuple[Tuple[float, float, float, float], float]:
    features = []
    shapes = []

    for row in rows:
        geom = geometry_from_row(row)
        shapes.append(geom)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "scan_id": row["scan_id"],
                    "map_name": row["map_name"],
                    "date_on_map": row["date_on_map"],
                    "map_scale": row["map_scale"],
                    "county_list": row["county_list"],
                    "expected_stem": row["expected_stem"],
                    "raw_downloaded": row["raw_downloaded"],
                    "processed_output_exists": row["processed_output_exists"],
                },
            }
        )

    footprints_fc = {"type": "FeatureCollection", "features": features}
    merged = unary_union(shapes)
    boundary_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "note": (
                        "Approximate LA County coverage derived from selected "
                        "historical map footprints. This is not an official "
                        "county boundary polygon."
                    )
                },
            }
        ],
    }

    footprints_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    footprints_path.write_text(json.dumps(footprints_fc, indent=2), encoding="utf-8")
    boundary_path.write_text(json.dumps(boundary_fc, indent=2), encoding="utf-8")

    return merged.bounds, merged.area


def counter_lines(counter: Counter, numeric: bool = False) -> List[str]:
    if numeric:
        items = sorted(counter.items(), key=lambda item: int(item[0]))
    else:
        items = sorted(counter.items(), key=lambda item: item[0])
    return [f"{key}: {value}" for key, value in items]


def write_summary(
    rows: Sequence[Dict[str, str]],
    path: Path,
    bounds: Tuple[float, float, float, float],
    area_sq_degrees: float,
    filter_mode: str,
    county_name: str,
) -> None:
    years = Counter(row["date_on_map"] for row in rows)
    scales = Counter(row["map_scale"] for row in rows)
    county_lists = Counter(row["county_list"] for row in rows)
    downloaded = sum(row["raw_downloaded"] == "yes" for row in rows)
    processed = sum(row["processed_output_exists"] == "yes" for row in rows)

    west, south, east, north = bounds
    lines = [
        "LA County historical topo inventory summary",
        "",
        f"Filter mode: {filter_mode}",
        f"County selector: {county_name}",
        "Boundary note: derived from selected map-sheet footprints; not an official county polygon.",
        "",
        f"Selected maps: {len(rows)}",
        f"Already downloaded in CA_area/maps/scanned/raw: {downloaded}",
        f"Already processed in CA_area/outputs/scanned: {processed}",
        f"Missing downloads: {len(rows) - downloaded}",
        f"Missing processed outputs: {len(rows) - processed}",
        "",
        "Approximate coverage bounds (WGS84):",
        f"west={west:.6f}, south={south:.6f}, east={east:.6f}, north={north:.6f}",
        f"Approximate dissolved coverage area (square degrees): {area_sq_degrees:.6f}",
        "",
        "Maps by year:",
    ]
    lines.extend(counter_lines(years, numeric=True))
    lines.extend(["", "Maps by scale:"])
    lines.extend(counter_lines(scales, numeric=True))
    lines.extend(["", "Distinct county_list values among selected rows:"])
    lines.extend(counter_lines(county_lists))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_file(url: str, destination: Path, overwrite: bool) -> str:
    if destination.exists() and not overwrite:
        return "skipped_existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())
    return "downloaded"


def sidecar_urls(row: Dict[str, str]) -> List[Tuple[str, str]]:
    base_url = row["geotiff_url"].strip()
    if not base_url.lower().endswith(".tif"):
        return []
    stem_url = base_url[:-4]
    return [
        (stem_url + suffix, suffix)
        for suffix in (".tfw", ".prj", ".xml")
    ]


def download_missing_rows(
    rows: Sequence[Dict[str, str]],
    download_root: Path,
    overwrite: bool,
    limit: Optional[int],
    sleep_seconds: float,
) -> None:
    attempted = 0

    for row in rows:
        if row["raw_downloaded"] == "yes" and not overwrite:
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        target_dir = download_root / row["expected_raw_subdir"]
        target_tif = target_dir / row["expected_tif_name"]

        print(f"[download] {row['scan_id']} -> {target_tif}")
        status = download_file(row["geotiff_url"], target_tif, overwrite=overwrite)
        print(f"[download] main tif: {status}")

        for sidecar_url, suffix in sidecar_urls(row):
            target_sidecar = target_dir / f"{target_tif.stem}{suffix}"
            try:
                sidecar_status = download_file(sidecar_url, target_sidecar, overwrite=overwrite)
                print(f"[download] {suffix}: {sidecar_status}")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    print(f"[download] {suffix}: missing on server (404), skipping")
                else:
                    raise

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> int:
    args = parse_args()

    county_rows = load_matching_rows(args.metadata_csv, args.county_name, args.state_name)
    if not county_rows:
        print("No LA County rows found in metadata.", file=sys.stderr)
        return 1

    boundary_rows = choose_boundary_rows(county_rows, args.county_name)
    coverage_geom = unary_union([geometry_from_row(row) for row in boundary_rows])
    if args.filter_mode == "county_list":
        selected_rows = county_rows
    else:
        selected_rows = select_rows_by_coverage(args.metadata_csv, args.state_name, coverage_geom)

    existing_raw = build_existing_scan_id_index(args.raw_root)
    existing_outputs = build_existing_scan_id_index(args.outputs_root)
    annotated_rows = annotate_rows(selected_rows, existing_raw, existing_outputs)

    write_inventory_csv(annotated_rows, args.inventory_csv)
    bounds, area_sq_degrees = write_geojson(
        annotated_rows,
        args.footprints_geojson,
        args.boundary_geojson,
    )
    write_summary(
        annotated_rows,
        args.summary_txt,
        bounds=bounds,
        area_sq_degrees=area_sq_degrees,
        filter_mode=args.filter_mode,
        county_name=args.county_name,
    )

    print(f"Inventory rows written: {len(annotated_rows)}")
    print(f"Inventory CSV: {args.inventory_csv}")
    print(f"Summary TXT: {args.summary_txt}")
    print(f"Footprints GeoJSON: {args.footprints_geojson}")
    print(f"Coverage boundary GeoJSON: {args.boundary_geojson}")

    if args.download_missing:
        download_root = args.download_dir or args.raw_root
        download_missing_rows(
            annotated_rows,
            download_root=download_root,
            overwrite=args.overwrite,
            limit=args.download_limit,
            sleep_seconds=args.sleep_seconds,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
