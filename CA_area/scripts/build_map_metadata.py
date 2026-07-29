#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "maps"
OUTPUTS = ROOT / "outputs"
OUT = ROOT / "map_metadata_clean.csv"


def clean_location_name(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_digital_filename(stem: str):
    """
    Example:
    CA_Beverly_Hills_20220119_TM_geo
    """
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected digital stem: {stem}")

    state = parts[0]
    date_str = parts[-3]
    location_raw = "_".join(parts[1:-3])

    year = int(date_str[:4])
    location_name = clean_location_name(location_raw)

    return {
        "state": state,
        "location_name": location_name,
        "map_year": year,
        "scale": "1:62500",
        "source_type": "digital",
    }


def parse_scanned_filename(stem: str):
    """
    Example:
    CA_Arroyo Grande NE_288055_1995_24000_geo

    Expected structure:
    state + location + sheet_id + year + scale + geo
    """
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected scanned stem: {stem}")

    state = parts[0]
    year_str = parts[-3]
    scale_str = parts[-2]
    location_raw = "_".join(parts[1:-4])

    year = int(year_str)
    scale = f"1:{scale_str}"
    location_name = clean_location_name(location_raw)

    return {
        "state": state,
        "location_name": location_name,
        "map_year": year,
        "scale": scale,
        "source_type": "scanned",
    }


def raster_meta(fp: Path):
    import rasterio

    with rasterio.open(fp) as ds:
        b = ds.bounds
        return {
            "width": ds.width,
            "height": ds.height,
            "res_x": ds.res[0],
            "res_y": ds.res[1],
            "bounds_left": b.left,
            "bounds_bottom": b.bottom,
            "bounds_right": b.right,
            "bounds_top": b.top,
        }


def build_output_index(outputs_root: Path):
    """
    Build:
    map_stem -> list of output files

    Assumes:
    outputs/digital/<map_stem>/...
    outputs/scanned/<map_stem>/...
    """
    index = {}

    for p in outputs_root.rglob("*"):
        if not p.is_file():
            continue

        parts = p.relative_to(outputs_root).parts
        if len(parts) >= 2:
            map_key = parts[1]
            index.setdefault(map_key, []).append(p)

    return index


def find_map_files():
    """
    Return candidate map tif files from both digital and scanned trees.
    """
    tif_files = []

    # digital maps
    digital_root = MAPS / "digital"
    if digital_root.exists():
        tif_files.extend(digital_root.rglob("*.tif"))

    # scanned maps
    scanned_root = MAPS / "scanned"
    if scanned_root.exists():
        tif_files.extend(scanned_root.rglob("*.tif"))

    return tif_files


def detect_source_type_from_path(fp: Path):
    s = str(fp)
    if "/digital/" in s:
        return "digital"
    if "/scanned/" in s:
        return "scanned"
    return "unknown"


def main():
    print(f"ROOT:    {ROOT}")
    print(f"MAPS:    {MAPS} exists={MAPS.exists()}")
    print(f"OUTPUTS: {OUTPUTS} exists={OUTPUTS.exists()}")

    print("Building output index...")
    output_index = build_output_index(OUTPUTS)
    print(f"Found output groups: {len(output_index)}")

    all_tifs = find_map_files()
    print(f"All candidate .tif maps found: {len(all_tifs)}")

    rows = []
    matched = 0

    print("Scanning maps...")
    for fp in all_tifs:
        stem = fp.stem

        # exclude orthophotos / satellite-like rasters
        if "TMorth_geo" in stem:
            continue

        if stem not in output_index:
            continue

        matched += 1

        source_type = detect_source_type_from_path(fp)

        try:
            if source_type == "digital":
                parsed = parse_digital_filename(stem)
            elif source_type == "scanned":
                parsed = parse_scanned_filename(stem)
            else:
                print(f"Skipping {stem}: unknown source type from path")
                continue
        except Exception as e:
            print(f"Skipping {stem}: filename parse failed: {e}")
            continue

        outputs = output_index[stem]

        has_vector_data = any(
            ("/gt/" in str(p)) or str(p).endswith(".gpkg") or str(p).endswith(".shp")
            for p in outputs
        )

        try:
            meta = raster_meta(fp)
        except Exception as e:
            print(f"Skipping {stem}: raster read failed: {e}")
            continue

        rows.append({
            "map_id": stem,
            "state": parsed["state"],
            "location_name": parsed["location_name"],
            "map_year": parsed["map_year"],
            "source_type": parsed["source_type"],
            "scale": parsed["scale"],
            "width": meta["width"],
            "height": meta["height"],
            "res_x": meta["res_x"],
            "res_y": meta["res_y"],
            "bounds_left": meta["bounds_left"],
            "bounds_bottom": meta["bounds_bottom"],
            "bounds_right": meta["bounds_right"],
            "bounds_top": meta["bounds_top"],
            "has_outputs": True,
            "n_outputs": len(outputs),
            "has_vector_data": has_vector_data,
        })

    print(f"Matched maps with outputs: {matched}")
    print(f"Rows written: {len(rows)}")

    if not rows:
        print("No rows found. No CSV written.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["source_type", "map_id"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print(f"\nSaved {len(df)} maps -> {OUT}")
    print("\nSource type counts:")
    print(df["source_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()