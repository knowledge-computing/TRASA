import re
from pathlib import Path
import pandas as pd

ROOT = Path("/users/4/kirsa002/IMOLA/metadata")
MAPS_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area/maps")
OUTPUTS_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area/outputs")

FILES = [
    ("historicaltopo", ROOT / "historicaltopo" / "historicaltopo.csv"),
    ("ustopo_historical", ROOT / "ustopo_historical" / "ustopo_historical.csv"),
    ("ustopo_current", ROOT / "ustopo_current" / "ustopo_current.csv"),
]

OUT_CSV = ROOT / "master_inventory_la_plus_all_processed.csv"

LA_MIN_LON = -118.944112
LA_MAX_LON = -117.644787
LA_MIN_LAT = 32.806553
LA_MAX_LAT = 34.822766

DESIRED_COLUMNS = [
    "source_table",
    "series",
    "edition",
    "map_name",
    "primary_state",
    "state_list",
    "county_list",
    "date_on_map",
    "publication_date",
    "production_date",
    "imprint_year",
    "survey_year",
    "field_check_year",
    "edit_year",
    "aerial_photo_year",
    "photo_inspection_year",
    "photo_revision_year",
    "map_scale",
    "grid_size",
    "cell_type",
    "westbc",
    "eastbc",
    "northbc",
    "southbc",
    "geom_wkt",
    "product_filename",
    "product_format",
    "product_url",
    "geotiff_url",
    "thumbnail_url",
    "metadata_url",
    "sciencebase_url",
]

def norm(s):
    s = str(s or "").strip().lower()
    s = s.replace("%20", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\.(pdf|tif|tiff|jpg|jpeg|png|xml|gpkg)$", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def basename_no_ext(path_or_name):
    s = str(path_or_name or "")
    s = s.split("/")[-1]
    s = re.sub(r"\.(pdf|tif|tiff|jpg|jpeg|png|xml|gpkg)$", "", s, flags=re.I)
    return s

def yyyymmdd_from_date(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return "".join(m.groups())
    m = re.match(r"(\d{4})", s)
    if m:
        return m.group(1)
    return ""

def overlaps_la_bbox(df):
    return ~(
        (df["eastbc"] < LA_MIN_LON) |
        (df["westbc"] > LA_MAX_LON) |
        (df["northbc"] < LA_MIN_LAT) |
        (df["southbc"] > LA_MAX_LAT)
    )

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
            if d.name.endswith("_tif"):
                names.add(d.name[:-4])
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

    if product_filename:
        keys.add(norm(product_filename))
        keys.add(norm(basename_no_ext(product_filename)))

    if geotiff_url and geotiff_url != "nan":
        keys.add(norm(basename_no_ext(geotiff_url)))

    if product_url and product_url != "nan":
        keys.add(norm(basename_no_ext(product_url)))

    date_candidates = [
        yyyymmdd_from_date(publication_date),
        yyyymmdd_from_date(production_date),
        yyyymmdd_from_date(date_on_map),
    ]
    date_candidates = [d for d in date_candidates if d]

    for d in date_candidates:
        keys.add(norm(f"CA_{map_name}_{d}_TM_geo"))
        keys.add(norm(f"CA_{map_name}_{d}_TM_geo_tif"))

    if source_table == "ustopo_current":
        keys.add(norm(f"CA_{map_name}"))

    return [k for k in keys if k]

def first_match(candidate_keys, index_dict):
    matched = []
    for k in candidate_keys:
        if k in index_dict:
            matched.extend(index_dict[k])
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

def load_full_inventory():
    frames = []
    for source_name, path in FILES:
        print("Reading", path)
        df = pd.read_csv(path, low_memory=False)
        df["source_table"] = source_name

        for col in ["westbc", "eastbc", "northbc", "southbc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in DESIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[DESIRED_COLUMNS].copy()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # dedupe exact same product row
    out = out.drop_duplicates(
        subset=["source_table", "product_url", "product_filename"],
        keep="first"
    ).reset_index(drop=True)

    # mark LA bbox
    bbox_cols = ["westbc", "eastbc", "northbc", "southbc"]
    out["in_la_bbox"] = False
    good = out[bbox_cols].notna().all(axis=1)
    out.loc[good, "in_la_bbox"] = overlaps_la_bbox(out.loc[good, bbox_cols.join if False else out.columns])

    # above line workaround not needed; recompute explicitly
    tmp = out.loc[good, ["westbc", "eastbc", "northbc", "southbc"]].copy()
    out.loc[good, "in_la_bbox"] = ~(
        (tmp["eastbc"] < LA_MIN_LON) |
        (tmp["westbc"] > LA_MAX_LON) |
        (tmp["northbc"] < LA_MIN_LAT) |
        (tmp["southbc"] > LA_MAX_LAT)
    )

    out["is_la_county_tagged"] = out["county_list"].fillna("").astype(str).str.contains("Los Angeles", case=False, regex=False)
    return out

def main():
    df = load_full_inventory()

    raw_index, raw_type_index = build_raw_index()
    out_index, out_type_index = build_output_index()

    processed_map_found = []
    processed_output_found = []
    processed_any = []
    processed_type = []
    matched_map_paths = []
    matched_output_paths = []
    match_key_used = []

    matched_processed_keys = set()

    for _, row in df.iterrows():
        keys = inventory_candidate_keys(row)
        raw_matches = first_match(keys, raw_index)
        out_matches = first_match(keys, out_index)

        for k in keys:
            if k in raw_index or k in out_index:
                matched_processed_keys.add(k)

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

    # keep all LA maps + all processed maps anywhere
    keep = df["in_la_bbox"] | df["processed_any"]
    master = df[keep].copy()

    master = master.sort_values(
        by=["processed_any", "in_la_bbox", "processed_type", "map_name", "date_on_map"],
        ascending=[False, False, True, True, True]
    ).reset_index(drop=True)

    master.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    print("Rows:", len(master))
    print("\nCounts:")
    print("LA bbox rows:", int(master["in_la_bbox"].sum()))
    print("Processed rows:", int(master["processed_any"].sum()))
    print("Processed + outside LA:", int((master["processed_any"] & ~master["in_la_bbox"]).sum()))
    print("\nProcessed type counts:")
    print(master["processed_type"].value_counts(dropna=False))

    print("\nExamples of processed maps outside LA:")
    cols = ["map_name", "date_on_map", "county_list", "processed_type", "product_url"]
    print(master[master["processed_any"] & ~master["in_la_bbox"]][cols].head(30).to_string(index=False))

if __name__ == "__main__":
    main()