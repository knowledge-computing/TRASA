import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from affine import Affine
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


def build_difficulty_mask(rgb):
    gray = (
        0.2989 * rgb[:, :, 0] +
        0.5870 * rgb[:, :, 1] +
        0.1140 * rgb[:, :, 2]
    ).astype(np.float32) / 255.0

    mean = ndi.uniform_filter(gray, size=9)
    mean_sq = ndi.uniform_filter(gray * gray, size=9)
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

    difficulty = 0.45 * dark_n + 0.35 * var_n + 0.20 * grad_n
    thresh = np.percentile(difficulty, 75)

    mask = difficulty >= thresh
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndi.binary_dilation(mask, structure=np.ones((5, 5)))

    return difficulty, mask.astype(np.uint8)


def overlap_score_windowed(geom, mask, transform):
    minx, miny, maxx, maxy = geom.bounds

    try:
        r0, c0 = rowcol(transform, minx, maxy)
        r1, c1 = rowcol(transform, maxx, miny)
    except Exception:
        return 0.0

    row_min = max(0, min(r0, r1) - 2)
    row_max = min(mask.shape[0] - 1, max(r0, r1) + 2)
    col_min = max(0, min(c0, c1) - 2)
    col_max = min(mask.shape[1] - 1, max(c0, c1) + 2)

    if row_max < row_min or col_max < col_min:
        return 0.0

    submask = mask[row_min:row_max + 1, col_min:col_max + 1]
    h, w = submask.shape
    if h == 0 or w == 0:
        return 0.0

    subtransform = transform * Affine.translation(col_min, row_min)

    road_arr = rasterize(
        [(geom, 1)],
        out_shape=(h, w),
        transform=subtransform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    road_pixels = road_arr > 0
    denom = int(road_pixels.sum())
    if denom == 0:
        return 0.0

    overlap = int((road_pixels & (submask > 0)).sum())
    return float(overlap) / float(denom)


def chunk_overlap_score(chunks_gdf, mask, transform):
    scores = []
    for geom in chunks_gdf.geometry:
        scores.append(overlap_score_windowed(geom, mask, transform))
    chunks_gdf["difficulty_overlap"] = scores
    return chunks_gdf


def drop_probability(row):
    code = row.get("mtfcc_code", None)
    length = row["chunk_len"]
    overlap = row["difficulty_overlap"]

    base = {
        "S1100": 0.01,
        "S1200": 0.04,
        "S1400": 0.10,
        "S1630": 0.12,
        "S1740": 0.20,
        "S1500": 0.28,
    }.get(code, 0.10)

    if length < 20:
        base += 0.18
    elif length < 40:
        base += 0.10
    elif length < 80:
        base += 0.04

    if overlap > 0.85:
        base += 0.45
    elif overlap > 0.60:
        base += 0.28
    elif overlap > 0.35:
        base += 0.14
    elif overlap > 0.15:
        base += 0.06

    if code == "S1100" and overlap < 0.20 and length > 60:
        base -= 0.05

    return min(max(base, 0.0), 0.97)


def remove_tiny_segments(gdf, min_len=10):
    return gdf[gdf.geometry.length >= min_len].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--difficulty-mask-out", required=True)
    parser.add_argument("--chunk-len", type=float, default=120.0)
    parser.add_argument("--simplify", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path = Path(args.difficulty_mask_out)
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.raster) as src:
        arr = src.read()
        raster_crs = src.crs
        transform = src.transform
        profile = src.profile.copy()

    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.float32)
    elif arr.shape[0] == 1:
        gray = arr[0].astype(np.float32)
        rgb = np.stack([gray, gray, gray], axis=2)
    else:
        raise ValueError("Unsupported raster band count: {}".format(arr.shape[0]))

    difficulty, diff_mask = build_difficulty_mask(rgb)

    roads = gpd.read_file(args.roads).to_crs(raster_crs)
    chunks = explode_and_split(roads, args.chunk_len)
    chunks["chunk_len"] = chunks.geometry.length

    chunks = chunk_overlap_score(chunks, diff_mask, transform)

    rng = np.random.RandomState(args.seed)
    chunks["drop_prob"] = chunks.apply(drop_probability, axis=1)
    chunks["keep"] = rng.rand(len(chunks)) > chunks["drop_prob"]

    pred = chunks[chunks["keep"]].copy()
    pred["geometry"] = pred.geometry.simplify(args.simplify, preserve_topology=False)
    pred = remove_tiny_segments(pred, min_len=10)
    pred = pred.drop(columns=["chunk_len", "difficulty_overlap", "drop_prob", "keep"], errors="ignore")
    pred.reset_index(drop=True, inplace=True)

    pred.to_file(str(out_path), driver="GPKG")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8")
    with rasterio.open(str(diff_path), "w", **out_profile) as dst:
        dst.write((diff_mask * 255).astype(np.uint8), 1)

    print("Original features: {}".format(len(roads)))
    print("Split chunks:      {}".format(len(chunks)))
    print("Prediction chunks: {}".format(len(pred)))
    print("Saved prediction:  {}".format(out_path))
    print("Saved difficulty:  {}".format(diff_path))


if __name__ == "__main__":
    main()