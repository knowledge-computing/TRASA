# save as: add_processing_info_to_inventory.py

import re
from pathlib import Path
import pandas as pd


INVENTORY_CSV = Path("/users/4/kirsa002/IMOLA/metadata/la_county_ngmdb_inventory_bbox_only.csv")
MAPS_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area/maps")
OUTPUTS_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area/outputs")

OUT_CSV = Path("/users/4/kirsa002/IMOLA/metadata/la_county_ngmdb_inventory_bbox_only_with_processing.csv")
OUT_XLSX = Path("/users/4/kirsa002/IMOLA/metadata/la_county_ngmdb_inventory_bbox_only_with_processing.xlsx")


def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.replace("%20", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\.(pdf|tif|tiff|jpg|jpeg|png|xml|gpkg)$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def basename_no_ext(path_or_name: str) -> str:
    s = str(path_or_name or "")
    s = s.split("/")[-1]
    s = re.sub(r"\.(pdf|tif|tiff|jpg|jpeg|png|xml|gpkg)$", "", s, flags=re.I)
    return s


def yyyymmdd_from_date(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # e.g. 2022-01-03 -> 20220103
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return "".join(m.groups())
    # fallback for year-only values
    m = re.match(r"(\d{4})", s)
    if m:
        return m.group(1)
    return ""


def build_raw_index():
    raw_index = {}
    raw_type_index = {}

    for map_type in ["digital", "scanned"]:
        root = MAPS_ROOT / map_type / "raw"
        if not root.exists():
            continue

        for d in root.iterdir():
            if not d.is_dir():
                continue

            names = {d.name}

            # folder name usually ends with _tif
            if d.name.endswith("_tif"):
                names.add(d.name[:-4])

            # also use actual tif file names inside the folder
            for tif in d.glob("*.tif"):
                names.add(tif.stem)

            for name in names:
                k = norm(name)
                raw_index.setdefault(k, []).append(str(d))
                raw_type_index[k] = map_type

    return raw_index, raw_type_index


def build_output_index():
    out_index = {}
    out_type_index = {}

    for map_type in ["digital", "scanned"]:
        root = OUTPUTS_ROOT / map_type
        if not root.exists():
            continue

        for d in root.iterdir():
            if not d.is_dir():
                continue

            # only count as a real output dir if it has expected content
            has_output_content = (
                (d / "metadata" / "pipeline_info.json").exists()
                or (d / "prediction").exists()
                or (d / "frame").exists()
                or (d / "viz").exists()
                or (d / "gt").exists()
            )
            if not has_output_content:
                continue

            k = norm(d.name)
            out_index.setdefault(k, []).append(str(d))
            out_type_index[k] = map_type

    return out_index, out_type_index


def inventory_candidate_keys(row):
    keys = set()

    product_filename = str(row.get("product_filename", "") or "")
    geotiff_url = str(row.get("geotiff_url", "") or "")
    product_url = str(row.get("product_url", "") or "")
    map_name = str(row.get("map_name", "") or "")
    source_table = str(row.get("source_table", "") or "")
    publication_date = row.get("publication_date", "")
    production_date = row.get("production_date", "")
    date_on_map = row.get("date_on_map", "")

    # direct filename-based keys
    if product_filename:
        keys.add(norm(product_filename))
        keys.add(norm(basename_no_ext(product_filename)))

    if geotiff_url and geotiff_url != "nan":
        keys.add(norm(basename_no_ext(geotiff_url)))

    if product_url and product_url != "nan":
        keys.add(norm(basename_no_ext(product_url)))

    # USTopo current raw folders use map name + exact date + TM_geo
    # example: CA_Beverly_Hills_20220119_TM_geo_tif
    date_candidates = [
        yyyymmdd_from_date(publication_date),
        yyyymmdd_from_date(production_date),
        yyyymmdd_from_date(date_on_map),
    ]
    date_candidates = [d for d in date_candidates if d]

    for d in date_candidates:
        keys.add(norm(f"CA_{map_name}_{d}_TM_geo"))
        keys.add(norm(f"CA_{map_name}_{d}_TM_geo_tif"))

    # helpful fallback for current PDFs like CA_Acton.pdf
    if source_table == "ustopo_current":
        keys.add(norm(f"CA_{map_name}"))

    return [k for k in keys if k]


def first_match(candidate_keys, index_dict):
    matched = []
    for k in candidate_keys:
        if k in index_dict:
            matched.extend(index_dict[k])
    # unique, stable order
    out = []
    seen = set()
    for x in matched:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_type(raw_matches, output_matches, raw_type_index, out_type_index, candidate_keys):
    for k in candidate_keys:
        if k in raw_type_index:
            return raw_type_index[k]
    for k in candidate_keys:
        if k in out_type_index:
            return out_type_index[k]
    if raw_matches or output_matches:
        return "unknown"
    return ""


def main():
    df = pd.read_csv(INVENTORY_CSV, low_memory=False)

    raw_index, raw_type_index = build_raw_index()
    out_index, out_type_index = build_output_index()

    processed_map_found = []
    processed_output_found = []
    processed_any = []
    processed_type = []
    matched_map_paths = []
    matched_output_paths = []
    match_key_used = []

    for _, row in df.iterrows():
        keys = inventory_candidate_keys(row)

        raw_matches = first_match(keys, raw_index)
        out_matches = first_match(keys, out_index)

        p_map = len(raw_matches) > 0
        p_out = len(out_matches) > 0

        processed_map_found.append(p_map)
        processed_output_found.append(p_out)
        processed_any.append(p_map or p_out)
        processed_type.append(infer_type(raw_matches, out_matches, raw_type_index, out_type_index, keys))
        matched_map_paths.append(" | ".join(raw_matches))
        matched_output_paths.append(" | ".join(out_matches))
        match_key_used.append(" | ".join(keys[:10]))

    df["processed_map_found"] = processed_map_found
    df["processed_output_found"] = processed_output_found
    df["processed_any"] = processed_any
    df["processed_type"] = processed_type
    df["matched_map_paths"] = matched_map_paths
    df["matched_output_paths"] = matched_output_paths
    df["match_key_used"] = match_key_used

    # optional convenience flag
    df["has_output"] = df["processed_output_found"]

    # sort to bring processed rows up
    df = df.sort_values(
        by=["processed_any", "processed_output_found", "processed_type", "map_name", "date_on_map"],
        ascending=[False, False, True, True, True]
    ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_excel(OUT_XLSX, index=False)
    except Exception as e:
        print("Excel export failed:", e)

    print(f"Saved CSV:  {OUT_CSV}")
    print(f"Saved XLSX: {OUT_XLSX}")
    print()
    print("Processed type counts:")
    print(df["processed_type"].value_counts(dropna=False))
    print()
    print("processed_map_found:", int(df["processed_map_found"].sum()))
    print("processed_output_found:", int(df["processed_output_found"].sum()))
    print("processed_any:", int(df["processed_any"].sum()))

    print("\nSample processed rows:")
    cols = [
        "source_table", "map_name", "date_on_map", "product_filename",
        "processed_type", "processed_map_found", "processed_output_found",
        "matched_map_paths", "matched_output_paths"
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[df["processed_any"]].head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()