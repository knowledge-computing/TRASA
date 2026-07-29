import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
import geopandas as gpd
from shapely.geometry import box
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")


def read_rgb_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
        }

    # Expect 3-band RGB, but tolerate more bands
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    elif arr.shape[0] == 1:
        rgb = np.repeat(arr[0][..., None], 3, axis=2)
    else:
        raise ValueError("Unsupported band count: {}".format(arr.shape[0]))

    return rgb.astype(np.float32), meta


def estimate_border_color(rgb, border_width=80):
    h, w, _ = rgb.shape
    bw = min(border_width, h // 4, w // 4)
    bw = max(bw, 1)

    strips = [
        rgb[:bw, :, :],
        rgb[h - bw:, :, :],
        rgb[:, :bw, :],
        rgb[:, w - bw:, :],
    ]
    border_pixels = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)

    # Median is robust to text and frame lines on the collar
    return np.median(border_pixels, axis=0)


def detect_content_mask(
    rgb,
    border_color,
    color_threshold=18.0,
    min_hole_size=5000,
    min_object_size=50000,
):
    # Euclidean distance from border/collar color
    dist = np.linalg.norm(rgb - border_color[None, None, :], axis=2)

    # Candidate content = sufficiently different from collar
    mask = dist > color_threshold

    # Morphological cleanup
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    mask = ndi.binary_fill_holes(mask)

    # Remove tiny objects
    labeled, num = ndi.label(mask)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= min_object_size
        keep[0] = False
        mask = keep[labeled]

    # Fill tiny holes inside content
    inv = ~mask
    labeled_holes, num_holes = ndi.label(inv)
    if num_holes > 0:
        hole_sizes = np.bincount(labeled_holes.ravel())
        small_holes = hole_sizes < min_hole_size
        small_holes[0] = False
        inv[small_holes[labeled_holes]] = False
        mask = ~inv

    return mask


def largest_central_component(mask):
    labeled, num = ndi.label(mask)
    if num == 0:
        raise RuntimeError("No content region detected.")

    h, w = mask.shape
    cy, cx = h / 2.0, w / 2.0

    best_label = None
    best_score = None

    for lab in range(1, num + 1):
        ys, xs = np.where(labeled == lab)
        if len(xs) == 0:
            continue

        area = len(xs)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        comp_cy = (y0 + y1) / 2.0
        comp_cx = (x0 + x1) / 2.0
        center_dist = np.sqrt((comp_cy - cy) ** 2 + (comp_cx - cx) ** 2)

        # Prefer large components near the center
        score = area - 200.0 * center_dist

        if best_score is None or score > best_score:
            best_score = score
            best_label = lab

    return labeled == best_label


def component_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask after component selection.")

    row_min = int(ys.min())
    row_max = int(ys.max())
    col_min = int(xs.min())
    col_max = int(xs.max())

    return row_min, row_max, col_min, col_max


def pixel_bbox_to_world_polygon(row_min, row_max, col_min, col_max, transform):
    # Use outer pixel corners
    x_left, y_top = xy(transform, row_min, col_min, offset="ul")
    x_right, y_bottom = xy(transform, row_max, col_max, offset="lr")
    return box(x_left, y_bottom, x_right, y_top)


def save_bbox_gpkg(geom, crs, out_path):
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)
    gdf.to_file(out_path, driver="GPKG")


def save_mask_png(mask, out_path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect interior map frame (collar-free content area) from a GeoTIFF."
    )
    parser.add_argument("raster", help="Path to input GeoTIFF")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same folder as raster)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=18.0,
        help="Color distance threshold from collar color",
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=80,
        help="Width of outer border strip used to estimate collar color",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raster_path = Path(args.raster)
    if not raster_path.exists():
        raise FileNotFoundError("Raster not found: {}".format(raster_path))

    if args.out_dir is None:
        out_dir = raster_path.parent
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    stem = raster_path.stem
    out_gpkg = out_dir / "{}_map_frame_bbox.gpkg".format(stem)
    out_png = out_dir / "{}_map_frame_mask.png".format(stem)

    print("Reading raster:", raster_path)
    rgb, meta = read_rgb_geotiff(str(raster_path))

    print("Estimating collar color...")
    border_color = estimate_border_color(rgb, border_width=args.border_width)
    print("Estimated border color:", border_color)

    print("Detecting content mask...")
    mask = detect_content_mask(
        rgb,
        border_color,
        color_threshold=args.threshold,
    )

    print("Selecting largest central component...")
    mask_main = largest_central_component(mask)

    row_min, row_max, col_min, col_max = component_bbox(mask_main)
    print("Detected pixel bbox:")
    print("  row_min =", row_min)
    print("  row_max =", row_max)
    print("  col_min =", col_min)
    print("  col_max =", col_max)

    geom = pixel_bbox_to_world_polygon(
        row_min, row_max, col_min, col_max, meta["transform"]
    )

    print("Saving outputs...")
    save_bbox_gpkg(geom, meta["crs"], str(out_gpkg))
    save_mask_png(mask_main, str(out_png))

    print("Done.")
    print("Saved bbox:", out_gpkg)
    print("Saved mask:", out_png)


if __name__ == "__main__":
    main()