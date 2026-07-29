import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point


def to_lines(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def substring(line, start_dist, end_dist):
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
        elif np.linalg.norm(np.array(out[-1]) - q0) > 1e-9:
            out.append(tuple(q0))

        if np.linalg.norm(np.array(out[-1]) - q1) > 1e-9:
            out.append(tuple(q1))

        acc = next_acc

    if len(out) < 2:
        return None

    return LineString(out)


def cut_line_by_distance(line, max_len):
    if line.length <= max_len:
        return [line]

    distances = list(np.arange(max_len, line.length, max_len))
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


def explode_and_split(gdf, chunk_len):
    rows = []
    for _, row in gdf.iterrows():
        for line in to_lines(row.geometry):
            for piece in cut_line_by_distance(line, chunk_len):
                new_row = row.copy()
                new_row.geometry = piece
                rows.append(new_row)

    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    out.reset_index(drop=True, inplace=True)
    return out


def endpoint_points(line):
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])


def cluster_points(points, tol=12.0):
    """
    Very simple point clustering by snapping to a grid.
    Good enough for approximate intersection regions.
    """
    bins = {}
    for pt in points:
        x, y = pt.x, pt.y
        key = (round(x / tol), round(y / tol))
        bins.setdefault(key, []).append((x, y))

    clustered = []
    counts = []
    for _, coords in bins.items():
        arr = np.array(coords)
        cx, cy = arr.mean(axis=0)
        clustered.append(Point(cx, cy))
        counts.append(len(coords))
    return clustered, counts


def build_intersection_points(split_gdf, tol=12.0, min_degree=3):
    pts = []
    for geom in split_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        p0, p1 = endpoint_points(geom)
        pts.append(p0)
        pts.append(p1)

    clustered, counts = cluster_points(pts, tol=tol)

    keep = [pt for pt, c in zip(clustered, counts) if c >= min_degree]
    return keep


def compute_intersection_distance(split_gdf, intersections):
    if not intersections:
        split_gdf["near_int_m"] = 1e9
        return split_gdf

    int_gdf = gpd.GeoDataFrame(geometry=intersections, crs=split_gdf.crs)
    int_sindex = int_gdf.sindex

    distances = []
    for geom in split_gdf.geometry:
        mid = geom.interpolate(0.5, normalized=True)
        idx = list(int_sindex.nearest(mid.bounds, 1))
        if len(idx) == 0:
            distances.append(1e9)
        else:
            d = mid.distance(int_gdf.geometry.iloc[idx[0]])
            distances.append(float(d))

    split_gdf["near_int_m"] = distances
    return split_gdf


def compute_local_density(split_gdf, radius=120.0):
    sindex = split_gdf.sindex
    densities = []

    for geom in split_gdf.geometry:
        mid = geom.interpolate(0.5, normalized=True)
        bbox = (mid.x - radius, mid.y - radius, mid.x + radius, mid.y + radius)
        cand = list(sindex.intersection(bbox))
        densities.append(len(cand))

    split_gdf["local_density"] = densities
    return split_gdf


def drop_probability(row):
    code = row["mtfcc_code"]
    length = row["chunk_len"]
    d_int = row["near_int_m"]
    dens = row["local_density"]

    base = {
        "S1100": 0.01,  # primary
        "S1200": 0.04,  # secondary
        "S1400": 0.10,  # local
        "S1630": 0.14,  # ramp
        "S1740": 0.22,  # private
        "S1500": 0.30,  # trail
    }.get(code, 0.10)

    # Short chunks are easier to miss
    if length < 20:
        base += 0.28
    elif length < 40:
        base += 0.18
    elif length < 80:
        base += 0.08

    # Near intersections / junctions: much easier to break
    if d_int < 12:
        base += 0.30
    elif d_int < 25:
        base += 0.18
    elif d_int < 45:
        base += 0.08

    # Dense neighborhoods are harder
    if dens > 25:
        base += 0.16
    elif dens > 15:
        base += 0.08

    # Primary roads should still survive more often
    if code == "S1100" and length > 80 and d_int > 25:
        base -= 0.05

    return min(max(base, 0.0), 0.97)


def remove_tiny_segments(gdf, min_len=10):
    keep = gdf.geometry.length >= min_len
    return gdf[keep].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric-crs", default="EPSG:26911")
    parser.add_argument("--chunk-len", type=float, default=70.0)
    parser.add_argument("--simplify", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(in_path).to_crs(args.metric_crs)

    split_gdf = explode_and_split(gdf, args.chunk_len)
    split_gdf["chunk_len"] = split_gdf.geometry.length

    intersections = build_intersection_points(split_gdf, tol=12.0, min_degree=3)
    split_gdf = compute_intersection_distance(split_gdf, intersections)
    split_gdf = compute_local_density(split_gdf, radius=120.0)

    rng = np.random.RandomState(args.seed)
    split_gdf["drop_prob"] = split_gdf.apply(drop_probability, axis=1)
    split_gdf["keep"] = rng.rand(len(split_gdf)) > split_gdf["drop_prob"]

    pred = split_gdf[split_gdf["keep"]].copy()

    pred["geometry"] = pred.geometry.simplify(args.simplify, preserve_topology=False)
    pred = remove_tiny_segments(pred, min_len=10)

    pred = pred.drop(
        columns=["chunk_len", "near_int_m", "local_density", "drop_prob", "keep"],
        errors="ignore"
    )
    pred.reset_index(drop=True, inplace=True)

    pred.to_file(out_path, driver="GPKG")

    print("Original features: {}".format(len(gdf)))
    print("Split chunks:      {}".format(len(split_gdf)))
    print("Intersections:     {}".format(len(intersections)))
    print("Prediction chunks: {}".format(len(pred)))
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()