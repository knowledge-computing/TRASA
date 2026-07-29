import pandas as pd
from pathlib import Path

ROOT = Path.home() / "IMOLA" / "metadata"

FILES = [
    ("historicaltopo", ROOT / "historicaltopo" / "historicaltopo.csv"),
    ("ustopo_historical", ROOT / "ustopo_historical" / "ustopo_historical.csv"),
    ("ustopo_current", ROOT / "ustopo_current" / "ustopo_current.csv"),
]

# Optional LA County bbox sanity check
LA_MIN_LON = -118.944112
LA_MAX_LON = -117.644787
LA_MIN_LAT = 32.806553
LA_MAX_LAT = 34.822766

OUTPUT_CSV = ROOT / "la_county_ngmdb_inventory_bbox_only.csv"

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

def overlaps_bbox(df: pd.DataFrame) -> pd.Series:
    # Keep rows whose map footprint intersects the LA County bbox.
    return ~(
        (df["eastbc"] < LA_MIN_LON) |
        (df["westbc"] > LA_MAX_LON) |
        (df["northbc"] < LA_MIN_LAT) |
        (df["southbc"] > LA_MAX_LAT)
    )

frames = []

for source_name, path in FILES:
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df["source_table"] = source_name

    # Normalize county/state columns to strings
    for col in ["state_list", "county_list", "primary_state"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # First-pass filter: California + Los Angeles county mention
    mask = pd.Series(True, index=df.index)

    if "state_list" in df.columns:
        mask &= df["state_list"].str.contains(r"\bCA\b", case=False, regex=True)
    elif "primary_state" in df.columns:
        mask &= df["primary_state"].str.contains("California|CA", case=False, regex=True)

    df = df[mask].copy()

    # Convert bbox cols if present and apply optional sanity check
    bbox_cols = ["westbc", "eastbc", "northbc", "southbc"]
    if all(c in df.columns for c in bbox_cols):
        for c in bbox_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[overlaps_bbox(df)].copy()

    # Add missing desired columns as blank
    for col in DESIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[DESIRED_COLUMNS]
    frames.append(df)

out = pd.concat(frames, ignore_index=True)

# Optional de-duplication: exact duplicate product URLs
if "product_url" in out.columns:
    out = out.drop_duplicates(subset=["source_table", "product_url", "product_filename"], keep="first")

# Better sort
sort_cols = [c for c in ["map_scale", "map_name", "date_on_map", "publication_date", "product_filename"] if c in out.columns]
out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Rows: {len(out)}")
print("\nCounts by source_table:")
print(out["source_table"].value_counts(dropna=False))

print("\nSample rows:")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(out.head(10).to_string(index=False))