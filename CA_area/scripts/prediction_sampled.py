#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiLineString


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


def build_difficulty(gray):
    gray = gray.astype(np.float32) / 255.0

    mean = ndi.uniform_filter(gray, size=7)
    mean_sq = ndi.uniform_filter(gray * gray, size=7)
    var = np.maximum(mean_sq - mean * mean, 0.0)

    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    grad = np.sqrt(gx * gx + gy * gy)

    def robust_norm(x):
        lo = np.percentile(x, 5)
        hi = np.percentile(x, 95)
        if hi <= lo:
            return np.zeros_like(x)
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    var_n = robust_norm(var)
    grad_n = robust_norm(grad)
    dark_n = 1.0 - gray

    difficulty = 0.5 * dark_n + 0.3 * var_n + 0.2 * grad_n
    return difficulty.astype(np.float32)


def sample_chunk_difficulty(geom, difficulty, transform, n_samples=5):
    vals = []
    for t in np.linspace(0.1, 0.9, n_samples):
        pt = geom.interpolate(t, normalized=True)
        try:
            r, c = rowcol(transform, pt.x, pt.y)
        except Exception:
            continue

        if 0 <= r < difficulty.shape[0] and 0 <= c < difficulty.shape[1]:
            vals.append(float(difficulty[r, c]))

    if not vals:
        return 0.0
    return float(np.mean(vals))


def drop_probability(row):
    code = row.get("mtfcc_code", None)
    length = row["chunk_len"]
    vis = row["vis_score"]

    base = {
        "S1100": 0.01,
        "S1200": 0.04,
        "S1400": 0.10,
        "S1630": 0.12,
        "S1740": 0.20,
        "S1500": 0.28,
    }.get(code, 0.10)

    if length < 25:
        base += 0.16
    elif length < 50:
        base += 0.08

    if vis > 0.85:
        base += 0.38
    elif vis > 0.70:
        base += 0.24
    elif vis > 0.55:
        base += 0.12
    elif vis > 0.40:
        base += 0.05

    if code == "S1100" and vis < 0.45 and length > 80:
        base -= 0.05

    return min(max(base, 0.0), 0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-len", type=float, default=140.0)
    parser.add_argument("--simplify", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-crs",
        choices=["raster", "roads"],
        default="raster",
        help="CRS of saved prediction output",
    )
    parser.add_argument(
        "--save-roads-raster-crs",
        action="store_true",
        help="Also save GT reprojected to raster CRS next to the output",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.raster) as src:
        arr = src.read()

        if arr.shape[0] >= 3:
            gray = (
                0.2989 * arr[0].astype(np.float32) +
                0.5870 * arr[1].astype(np.float32) +
                0.1140 * arr[2].astype(np.float32)
            )
        else:
            gray = arr[0].astype(np.float32)

        raster_crs = src.crs
        transform = src.transform

    if raster_crs is None:
        raise RuntimeError("Raster has no CRS.")

    difficulty = build_difficulty(gray)

    roads_orig = gpd.read_file(args.roads)
    roads_orig_crs = roads_orig.crs
    if roads_orig_crs is None:
        raise RuntimeError("Roads file has no CRS.")

    # Reproject roads to raster CRS for all raster-based scoring work
    roads_raster = roads_orig.to_crs(raster_crs)

    if args.save_roads_raster_crs:
        gt_raster_path = out_path.parent / "roads_clip_raster_crs.gpkg"
        roads_raster.to_file(str(gt_raster_path), driver="GPKG")
        print(f"Saved raster-CRS GT: {gt_raster_path}")

    # Split/scoring happen in raster CRS so geometry matches raster space
    chunks_raster = explode_and_split(roads_raster, args.chunk_len)
    chunks_raster["chunk_len"] = chunks_raster.geometry.length
    chunks_raster["vis_score"] = chunks_raster.geometry.apply(
        lambda g: sample_chunk_difficulty(g, difficulty, transform, n_samples=5)
    )

    rng = np.random.RandomState(args.seed)
    chunks_raster["drop_prob"] = chunks_raster.apply(drop_probability, axis=1)
    chunks_raster["keep"] = rng.rand(len(chunks_raster)) > chunks_raster["drop_prob"]

    pred_raster = chunks_raster[chunks_raster["keep"]].copy()
    pred_raster["geometry"] = pred_raster.geometry.simplify(
        args.simplify, preserve_topology=False
    )
    pred_raster = pred_raster.drop(
        columns=["chunk_len", "vis_score", "drop_prob", "keep"],
        errors="ignore",
    )
    pred_raster.reset_index(drop=True, inplace=True)

    if args.output_crs == "roads":
        pred_out = pred_raster.to_crs(roads_orig_crs)
    else:
        pred_out = pred_raster

    pred_out.to_file(str(out_path), driver="GPKG")

    print(f"Raster CRS:          {raster_crs}")
    print(f"Roads original CRS:  {roads_orig_crs}")
    print(f"Prediction CRS:      {pred_out.crs}")
    print(f"Original features:   {len(roads_orig)}")
    print(f"Split chunks:        {len(chunks_raster)}")
    print(f"Prediction chunks:   {len(pred_out)}")
    print(f"Saved:               {out_path}")


if __name__ == "__main__":
    main()