import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString


def to_lines(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def cut_line_by_distance(line, max_len):
    """
    Split a LineString into chunks of approximately max_len (same units as CRS).
    """
    if line.length <= max_len:
        return [line]

    distances = list(np.arange(max_len, line.length, max_len))
    if not distances:
        return [line]

    pieces = []
    start_d = 0.0
    for d in distances:
        seg = substring(line, start_d, d)
        if seg is not None and not seg.is_empty and seg.length > 0:
            pieces.append(seg)
        start_d = d

    seg = substring(line, start_d, line.length)
    if seg is not None and not seg.is_empty and seg.length > 0:
        pieces.append(seg)

    return pieces


def substring(line, start_dist, end_dist):
    """
    Return a LineString substring between start_dist and end_dist.
    Compatible with old shapely by manual interpolation.
    """
    if start_dist == end_dist:
        return None

    coords = list(line.coords)
    if len(coords) < 2:
        return None

    acc = 0.0
    out = []

    for i in range(len(coords) - 1):
        p0 = np.array(coords[i], dtype=float)
        p1 = np.array(coords[i + 1], dtype=float)
        seg_len = np.linalg.norm(p1 - p0)
        next_acc = acc + seg_len

        # no overlap
        if next_acc <= start_dist:
            acc = next_acc
            continue
        if acc >= end_dist:
            break

        local_start = max(start_dist, acc)
        local_end = min(end_dist, next_acc)

        t0 = 0.0 if seg_len == 0 else (local_start - acc) / seg_len
        t1 = 1.0 if seg_len == 0 else (local_end - acc) / seg_len

        q0 = p0 + t0 * (p1 - p0)
        q1 = p0 + t1 * (p1 - p0)

        if not out:
            out.append(tuple(q0))
        else:
            if np.linalg.norm(np.array(out[-1]) - q0) > 1e-9:
                out.append(tuple(q0))

        if np.linalg.norm(np.array(out[-1]) - q1) > 1e-9:
            out.append(tuple(q1))

        acc = next_acc

    if len(out) < 2:
        return None

    return LineString(out)


def explode_and_split(gdf, chunk_len):
    rows = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        for line in to_lines(geom):
            for piece in cut_line_by_distance(line, chunk_len):
                new_row = row.copy()
                new_row.geometry = piece
                rows.append(new_row)

    if not rows:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    out.reset_index(drop=True, inplace=True)
    return out


def drop_probability(row):
    code = row["mtfcc_code"]
    length = row["chunk_len"]

    # more realistic class-based miss rates
    probs = {
        "S1100": 0.03,  # primary
        "S1200": 0.08,  # secondary
        "S1400": 0.22,  # local roads
        "S1630": 0.20,  # ramps
        "S1740": 0.38,  # private roads
        "S1500": 0.45,  # trail
    }

    p = probs.get(code, 0.20)

    # short chunks more likely to disappear
    if length < 25:
        p += 0.30
    elif length < 50:
        p += 0.18
    elif length < 100:
        p += 0.08

    return min(p, 0.97)


def remove_tiny_segments(gdf, min_len=8):
    if len(gdf) == 0:
        return gdf
    keep = gdf.geometry.length >= min_len
    return gdf[keep].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input clipped GT roads")
    parser.add_argument("--output", required=True, help="Output fake prediction")
    parser.add_argument("--metric-crs", default="EPSG:26911", help="Projected CRS for processing")
    parser.add_argument("--chunk-len", type=float, default=80.0, help="Chunk length in meters")
    parser.add_argument("--simplify", type=float, default=4.0, help="Simplify tolerance in meters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(in_path)

    # Reproject to metric CRS
    gdf = gdf.to_crs(args.metric_crs)

    # Split multipart + cut into chunks
    split_gdf = explode_and_split(gdf, args.chunk_len)
    split_gdf["chunk_len"] = split_gdf.geometry.length

    rng = np.random.RandomState(args.seed)
    split_gdf["drop_prob"] = split_gdf.apply(drop_probability, axis=1)
    split_gdf["keep"] = rng.rand(len(split_gdf)) > split_gdf["drop_prob"]

    pred = split_gdf[split_gdf["keep"]].copy()

    # Slightly simplify remaining geometry
    pred["geometry"] = pred.geometry.simplify(args.simplify, preserve_topology=False)

    # Remove tiny leftovers
    pred = remove_tiny_segments(pred, min_len=8)

    pred = pred.drop(columns=["chunk_len", "drop_prob", "keep"], errors="ignore")
    pred.reset_index(drop=True, inplace=True)

    pred.to_file(out_path, driver="GPKG")

    print("Original features: {}".format(len(gdf)))
    print("Split chunks:       {}".format(len(split_gdf)))
    print("Prediction chunks:  {}".format(len(pred)))
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()