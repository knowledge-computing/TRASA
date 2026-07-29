import re
from pathlib import Path

import pandas as pd


INPUT_CSV = "/users/4/kirsa002/IMOLA/metadata/master_inventory_la_plus_all_processed_with_prediction_links.csv"
OUTPUT_CSV = "/users/4/kirsa002/IMOLA/metadata/master_inventory_la_public_clean.csv"


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def to_yes_no(value):
    if pd.isna(value):
        return "No"
    if isinstance(value, bool):
        return "Yes" if value else "No"

    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return "Yes"
    if text in ("false", "0", "no", "n", ""):
        return "No"

    return "Yes"


def link_exists(value):
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def processing_status_from_link(value):
    return "Processed" if link_exists(value) else "Not Processed"


def source_type_public(value, has_prediction):
    if pd.notna(value) and str(value).strip():
        text = str(value).strip().lower()
        if text == "digital":
            return "Digital"
        if text == "scanned":
            return "Scanned"
        return str(value).strip()

    if has_prediction == "No":
        return "Not processed"

    return ""


def to_int_year(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_int_scale(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def make_region_note(row):
    in_bbox = str(row.get("in_la_bbox", "")).strip().lower() == "yes"
    has_prediction = row.get("has_prediction", "No") == "Yes"

    if in_bbox:
        return "Inside LA bbox"
    if has_prediction:
        return "Outside LA bbox, included because processed"
    return "Outside LA bbox"


def make_data_available(row):
    has_prediction = row.get("has_prediction", "No") == "Yes"
    has_geotiff = link_exists(row.get("geotiff_link", ""))
    has_map_pdf = link_exists(row.get("map_download_link", ""))

    parts = []

    if has_prediction:
        parts.append("Prediction")
    if has_geotiff:
        parts.append("GeoTIFF")
    if has_map_pdf:
        parts.append("Map file")

    if not parts:
        return "No linked data"

    return " + ".join(parts)


def pick_first_existing(df, candidates, default_value=""):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default_value] * len(df), index=df.index)


def count_nonempty(series):
    return series.astype(str).str.strip().ne("").sum()


def main():
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    df = pd.read_csv(str(input_path))

    public = pd.DataFrame(index=df.index)

    public["map_name"] = pick_first_existing(
        df,
        ["map_name", "Map Name"],
        ""
    ).apply(clean_text)

    public["county"] = pick_first_existing(
        df,
        ["county_list", "County List", "county"],
        ""
    ).apply(clean_text)

    public["state"] = pick_first_existing(
        df,
        ["primary_state", "Primary State", "state"],
        ""
    ).apply(clean_text)

    public["year"] = to_int_year(
        pick_first_existing(
            df,
            ["date_on_map", "Date On Map", "year"],
            pd.NA
        )
    )

    public["scale"] = to_int_scale(
        pick_first_existing(
            df,
            ["map_scale", "Scale", "scale"],
            pd.NA
        )
    )

    public["series"] = pick_first_existing(
        df,
        ["series", "Series"],
        ""
    ).apply(clean_text)

    public["product_format"] = pick_first_existing(
        df,
        ["product_format", "Product Format"],
        ""
    ).apply(clean_text)

    public["prediction_link"] = pick_first_existing(
        df,
        ["prediction_download_url"],
        ""
    ).fillna("").astype(str).str.strip()

    public["map_download_link"] = pick_first_existing(
        df,
        ["product_url", "Download Product S3", "download_url"],
        ""
    ).fillna("").astype(str).str.strip()

    public["geotiff_link"] = pick_first_existing(
        df,
        ["geotiff_url"],
        ""
    ).fillna("").astype(str).str.strip()

    public["has_prediction"] = public["prediction_link"].apply(
        lambda x: "Yes" if link_exists(x) else "No"
    )

    public["processing_status"] = public["prediction_link"].apply(
        processing_status_from_link
    )

    processed_type_series = pick_first_existing(
        df,
        ["processed_type"],
        ""
    )

    public["source_type"] = [
        source_type_public(value, has_prediction)
        for value, has_prediction in zip(processed_type_series, public["has_prediction"])
    ]

    in_la_bbox_series = pick_first_existing(
        df,
        ["in_la_bbox"],
        False
    )
    public["in_la_bbox"] = in_la_bbox_series.apply(to_yes_no)

    is_la_county_series = pick_first_existing(
        df,
        ["is_la_county_tagged"],
        False
    )
    public["is_la_county_tagged"] = is_la_county_series.apply(to_yes_no)

    public["region_note"] = public.apply(make_region_note, axis=1)
    public["data_available"] = public.apply(make_data_available, axis=1)

    preferred_order = [
        "map_name",
        "year",
        "source_type",
        "processing_status",
        "prediction_link",
        "county",
        "state",
        "scale",
        "series",
        "map_download_link",
        "geotiff_link",
        "product_format",
        "data_available",
        "in_la_bbox",
        "is_la_county_tagged",
        "region_note",
        "has_prediction",
    ]

    existing_preferred = [col for col in preferred_order if col in public.columns]
    remaining = [col for col in public.columns if col not in existing_preferred]
    public = public[existing_preferred + remaining]

    public["processing_status"] = pd.Categorical(
        public["processing_status"],
        categories=["Processed", "Not Processed"],
        ordered=True
    )

    public["source_type"] = pd.Categorical(
        public["source_type"],
        categories=["Digital", "Scanned", "Not processed", ""],
        ordered=True
    )

    sort_columns = [col for col in ["processing_status", "source_type", "year", "map_name"] if col in public.columns]
    public = public.sort_values(
        by=sort_columns,
        ascending=[True, True, False, True]
    ).reset_index(drop=True)

    public.to_csv(str(output_path), index=False)

    print("Saved:", str(output_path))
    print("Rows:", len(public))
    print("Columns and non-empty counts:")
    for col in public.columns:
        nonempty = count_nonempty(public[col])
        print(" - {}: {}/{}".format(col, nonempty, len(public)))


if __name__ == "__main__":
    main()