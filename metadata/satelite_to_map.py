import re
from pathlib import Path

import pandas as pd


MASTER_CSV = "/users/4/kirsa002/IMOLA/metadata/master_inventory_la_plus_all_processed_with_prediction_links.csv"
OUTPUT_CSV = "/users/4/kirsa002/IMOLA/metadata/master_inventory_la_plus_all_processed_with_correct_public_map_links.csv"

# You already have these unpacked as folders
TOPOVIEW_INVENTORY_SOURCES = [
    "/users/4/kirsa002/IMOLA/metadata/historicaltopo",
    "/users/4/kirsa002/IMOLA/metadata/topomaps_all",
    "/users/4/kirsa002/IMOLA/metadata/ustopo_current",
    "/users/4/kirsa002/IMOLA/metadata/ustopo_historical",
    "/users/4/kirsa002/IMOLA/metadata/ustopo_replacements",
]


def has_value(x):
    return pd.notna(x) and str(x).strip() != ""


def pick_first_path(paths_str):
    if not has_value(paths_str):
        return ""

    text = str(paths_str).strip()
    parts = re.split(r"\s*[;|]\s*", text)
    for part in parts:
        part = part.strip()
        if part:
            return part
    return text


def url_basename_no_ext(url_or_path):
    if not has_value(url_or_path):
        return ""
    s = str(url_or_path).strip().split("?")[0]
    name = Path(s).name
    return Path(name).stem


def normalize_key(s):
    if not has_value(s):
        return ""

    s = str(s).strip()
    s = s.replace("%20", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()

    return s


def make_inventory_key_from_url(url_or_path):
    return normalize_key(url_basename_no_ext(url_or_path))


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_product_source(url):
    s = str(url)
    if "HistoricalTopo" in s:
        return "HistoricalTopo"
    if "USTopo" in s:
        return "USTopo"
    return ""


def derive_processed_map_key(matched_map_path):
    if not has_value(matched_map_path):
        return ""

    p = Path(str(matched_map_path).strip())
    name = p.name

    if p.suffix:
        stem = p.stem
    else:
        stem = name

    # remove trailing folder marker
    if stem.endswith("_tif"):
        stem = stem[:-4]

    # remove duplicate suffix like -2
    stem = re.sub(r"-\d+$", "", stem)

    # force orthophoto names back to map names
    stem = stem.replace("_TMorth_geo", "_TM_geo")

    # fix scanned names like BeverlyHills -> Beverly Hills
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)

    # only now normalize
    stem = normalize_key(stem)

    return stem


def find_csv_files(source_path):
    p = Path(source_path)

    if not p.exists():
        print("WARNING: inventory source not found:", source_path)
        return []

    if p.is_file():
        if p.suffix.lower() == ".csv":
            return [p]
        print("WARNING: file is not CSV:", source_path)
        return []

    # directory
    csvs = sorted(p.rglob("*.csv"))
    if not csvs:
        print("WARNING: no CSV files found under:", source_path)
    return csvs


def load_topoview_inventory(sources):
    csv_files = []
    for src in sources:
        csv_files.extend(find_csv_files(src))

    if not csv_files:
        raise ValueError("No TopoView inventory CSVs were found in the provided folders/files.")

    print("Found inventory CSV files:")
    for f in csv_files:
        print(" -", str(f))

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(str(csv_file))
            df["_inventory_source_file"] = str(csv_file)
            dfs.append(df)
        except Exception as e:
            print("WARNING: failed to read {}: {}".format(csv_file, e))

    if not dfs:
        raise ValueError("CSV files were found, but none could be read successfully.")

    return pd.concat(dfs, ignore_index=True, sort=False)


def build_inventory_index(inv_df):
    product_url_col = first_existing_column(inv_df, ["product_url", "Product URL", "download_url"])
    geotiff_url_col = first_existing_column(inv_df, ["geotiff_url", "GeoTIFF URL"])
    product_filename_col = first_existing_column(inv_df, ["product_filename", "Product Filename"])

    if product_url_col is None and geotiff_url_col is None and product_filename_col is None:
        raise ValueError(
            "Could not find inventory URL/filename columns. "
            "Available columns are:\n{}".format(list(inv_df.columns))
        )

    rows = []

    for _, row in inv_df.iterrows():
        product_url = row[product_url_col] if product_url_col else ""
        geotiff_url = row[geotiff_url_col] if geotiff_url_col else ""
        product_filename = row[product_filename_col] if product_filename_col else ""
        source_file = row.get("_inventory_source_file", "")

        candidate_keys = set()

        if has_value(product_url):
            candidate_keys.add(make_inventory_key_from_url(product_url))
        if has_value(geotiff_url):
            candidate_keys.add(make_inventory_key_from_url(geotiff_url))
        if has_value(product_filename):
            candidate_keys.add(normalize_key(Path(str(product_filename)).stem))

        for key in candidate_keys:
            if not key:
                continue

            rows.append({
                "inventory_key": key,
                "product_url": product_url if has_value(product_url) else "",
                "geotiff_url": geotiff_url if has_value(geotiff_url) else "",
                "product_source": detect_product_source(product_url),
                "inventory_source_file": source_file,
            })

    idx = pd.DataFrame(rows)

    if len(idx) == 0:
        raise ValueError("Inventory index is empty after parsing the inventory files.")

    idx["is_tmorth"] = idx["inventory_key"].astype(str).str.contains("TMorth", na=False)
    idx["has_product_url"] = idx["product_url"].apply(has_value)
    idx["has_geotiff_url"] = idx["geotiff_url"].apply(has_value)

    idx = idx.sort_values(
        by=["inventory_key", "is_tmorth", "has_product_url", "has_geotiff_url"],
        ascending=[True, True, False, False]
    )

    idx = idx.drop_duplicates(subset=["inventory_key"], keep="first").reset_index(drop=True)
    return idx


def main():
    master = pd.read_csv(MASTER_CSV)
    inventory = load_topoview_inventory(TOPOVIEW_INVENTORY_SOURCES)
    inv_index = build_inventory_index(inventory)

    processed_mask = master["prediction_download_url"].apply(has_value)

    if "product_url_original" not in master.columns:
        master["product_url_original"] = master["product_url"]
    if "geotiff_url_original" not in master.columns:
        master["geotiff_url_original"] = master["geotiff_url"]

    master["matched_map_path_primary"] = master["matched_map_paths"].apply(pick_first_path)
    master["processed_map_key"] = master["matched_map_path_primary"].apply(derive_processed_map_key)

    master = master.merge(
        inv_index[["inventory_key", "product_url", "geotiff_url", "product_source", "inventory_source_file"]].rename(
            columns={
                "inventory_key": "processed_map_key",
                "product_url": "product_url_official",
                "geotiff_url": "geotiff_url_official",
                "product_source": "product_source_official",
                "inventory_source_file": "inventory_source_file_official",
            }
        ),
        on="processed_map_key",
        how="left",
    )

    use_product = processed_mask & master["product_url_official"].apply(has_value)
    use_geotiff = processed_mask & master["geotiff_url_official"].apply(has_value)

    master.loc[use_product, "product_url"] = master.loc[use_product, "product_url_official"]
    master.loc[use_geotiff, "geotiff_url"] = master.loc[use_geotiff, "geotiff_url_official"]

    master["official_link_found"] = processed_mask & master["product_url_official"].apply(has_value)
    master["official_geotiff_found"] = processed_mask & master["geotiff_url_official"].apply(has_value)
    master["resolved_to_tmorth"] = master["product_url"].astype(str).str.contains("TMorth", case=False, na=False)

    master.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)
    print("Total rows:", len(master))
    print("Processed rows:", int(processed_mask.sum()))
    print("Processed rows with official map PDF link:", int(master.loc[processed_mask, "official_link_found"].sum()))
    print("Processed rows with official GeoTIFF link:", int(master.loc[processed_mask, "official_geotiff_found"].sum()))
    print("Processed rows resolved to TMorth (should be 0):", int(master.loc[processed_mask, "resolved_to_tmorth"].sum()))

    missing = master.loc[processed_mask & ~master["official_link_found"], ["map_name", "matched_map_path_primary", "processed_map_key"]]
    if len(missing) > 0:
        print("\nProcessed rows with no official match (first 20):")
        print(missing.head(20).to_string(index=False))

    print("\nExample fixed rows:")
    cols = [
        "map_name",
        "processed_type",
        "matched_map_path_primary",
        "processed_map_key",
        "product_source_official",
        "inventory_source_file_official",
        "product_url",
    ]
    print(master.loc[processed_mask, cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()