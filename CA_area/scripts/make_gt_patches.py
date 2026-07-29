import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_TIF = PROJECT_ROOT / "maps/digital/raw/CA_Alberhill_20211211_TM_geo_tif/CA_Alberhill_20211211_TM_geo.tif"
CROPPED_TIF = PROJECT_ROOT / "outputs/digital/CA_Alberhill_20211211_TM_geo/viz/map_cropped.tif"
FRAME_MASK = PROJECT_ROOT / "outputs/digital/CA_Alberhill_20211211_TM_geo/frame/map_frame_mask.png"
GT_GPKG = PROJECT_ROOT / "outputs/digital/CA_Alberhill_20211211_TM_geo/gt/roads_clip.gpkg"

OUT_DIR = PROJECT_ROOT / "patches_with_gt" / "CA_Alberhill_20211211_TM_geo"
PATCH_SIZE = 256

# Keep only patches with enough valid map area
MIN_MASK_COVERAGE = 0.80

# Thickness of GT overlay in pixels
GT_LINE_WIDTH = 3


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_png_mask(mask_path: Path):
    """Load polygon mask PNG as binary uint8 array."""
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    # Anything nonzero counts as inside polygon
    return (arr > 0).astype(np.uint8)


def normalize_to_uint8(img):
    """
    Convert raster read from rasterio (bands, H, W) into uint8 RGB image.
    Handles:
      - uint8 already
      - other numeric ranges by min-max scaling per band
      - 1-band grayscale replicated to RGB
      - >3 bands takes first 3
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image with shape (bands, H, W), got {img.shape}")

    bands, h, w = img.shape

    if bands == 1:
        img = np.repeat(img, 3, axis=0)
    elif bands >= 3:
        img = img[:3]
    else:
        # 2 bands -> pad to 3
        pad = np.repeat(img[-1:], 3 - bands, axis=0)
        img = np.concatenate([img, pad], axis=0)

    out = np.zeros_like(img, dtype=np.uint8)

    for b in range(3):
        band = img[b].astype(np.float32)
        finite = np.isfinite(band)
        if not finite.any():
            continue

        vals = band[finite]
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)

        if vmax <= vmin:
            vmax = vals.max()
            vmin = vals.min()

        if vmax <= vmin:
            scaled = np.zeros_like(band, dtype=np.uint8)
        else:
            scaled = (band - vmin) / (vmax - vmin)
            scaled = np.clip(scaled, 0, 1)
            scaled = (scaled * 255).astype(np.uint8)

        out[b] = scaled

    return np.transpose(out, (1, 2, 0))  # (H, W, 3)


def read_raster_patch(src, window: Window):
    arr = src.read(window=window, boundless=False)
    return normalize_to_uint8(arr)


def make_overlay_mask_for_patch(gdf_patch, patch_transform, patch_h, patch_w):
    """
    Rasterize GT lines into a binary mask in patch pixel coordinates.
    """
    if gdf_patch.empty:
        return np.zeros((patch_h, patch_w), dtype=np.uint8)

    shapes = [(geom, 1) for geom in gdf_patch.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((patch_h, patch_w), dtype=np.uint8)

    line_mask = rasterize(
        shapes=shapes,
        out_shape=(patch_h, patch_w),
        transform=patch_transform,
        fill=0,
        all_touched=True,
        default_value=1,
        dtype=np.uint8
    )

    # Thicken lines a bit with simple neighborhood max filter, without cv2/scipy
    if GT_LINE_WIDTH > 1:
        k = GT_LINE_WIDTH // 2
        padded = np.pad(line_mask, k, mode="constant")
        thick = np.zeros_like(line_mask)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                thick = np.maximum(
                    thick,
                    padded[k + dy:k + dy + line_mask.shape[0], k + dx:k + dx + line_mask.shape[1]]
                )
        line_mask = thick

    return line_mask


def apply_red_overlay(rgb_patch, line_mask):
    """
    Draw GT lines in bright red on top of raster patch.
    """
    out = rgb_patch.copy()
    red = line_mask.astype(bool)

    # Strong visible red overlay
    out[red, 0] = 255
    out[red, 1] = 0
    out[red, 2] = 0
    return out


def save_patch(img_array, out_path: Path):
    Image.fromarray(img_array).save(out_path)


def process_with_raw_and_mask():
    """
    Preferred path:
    - use full raw tif
    - use frame mask png to keep polygon only
    - save only tiles mostly inside polygon
    """
    if not (RAW_TIF.exists() and FRAME_MASK.exists() and GT_GPKG.exists()):
        return False

    ensure_dir(OUT_DIR)

    with rasterio.open(RAW_TIF) as src:
        mask = load_png_mask(FRAME_MASK)

        # Require same raster size as mask
        if mask.shape != (src.height, src.width):
            print(f"[WARN] Mask size {mask.shape} != raster size {(src.height, src.width)}. Falling back.")
            return False

        gdf = gpd.read_file(GT_GPKG)
        if gdf.crs is None:
            raise ValueError("GT GPKG has no CRS.")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        tile_count = 0

        for row in range(0, src.height - PATCH_SIZE + 1, PATCH_SIZE):
            for col in range(0, src.width - PATCH_SIZE + 1, PATCH_SIZE):
                mask_patch = mask[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
                coverage = mask_patch.mean()

                if coverage < MIN_MASK_COVERAGE:
                    continue

                window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                rgb_patch = read_raster_patch(src, window)

                # Zero-out non-polygon region inside patch
                rgb_patch[mask_patch == 0] = 0

                bounds = rasterio.windows.bounds(window, src.transform)
                patch_box = box(*bounds)

                gdf_patch = gdf[gdf.intersects(patch_box)].copy()
                patch_transform = src.window_transform(window)

                line_mask = make_overlay_mask_for_patch(
                    gdf_patch=gdf_patch,
                    patch_transform=patch_transform,
                    patch_h=PATCH_SIZE,
                    patch_w=PATCH_SIZE
                )

                overlaid = apply_red_overlay(rgb_patch, line_mask)

                out_name = f"patch_r{row:05d}_c{col:05d}.png"
                save_patch(overlaid, OUT_DIR / out_name)
                tile_count += 1

        print(f"[OK] Saved {tile_count} patches to {OUT_DIR}")
        return True


def process_with_cropped_map():
    """
    Fallback:
    - use already prepared cropped map
    - tile it directly
    """
    if not (CROPPED_TIF.exists() and GT_GPKG.exists()):
        raise FileNotFoundError("Fallback files not found.")

    ensure_dir(OUT_DIR)

    with rasterio.open(CROPPED_TIF) as src:
        gdf = gpd.read_file(GT_GPKG)
        if gdf.crs is None:
            raise ValueError("GT GPKG has no CRS.")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        tile_count = 0

        for row in range(0, src.height - PATCH_SIZE + 1, PATCH_SIZE):
            for col in range(0, src.width - PATCH_SIZE + 1, PATCH_SIZE):
                window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                rgb_patch = read_raster_patch(src, window)

                bounds = rasterio.windows.bounds(window, src.transform)
                patch_box = box(*bounds)

                gdf_patch = gdf[gdf.intersects(patch_box)].copy()
                patch_transform = src.window_transform(window)

                line_mask = make_overlay_mask_for_patch(
                    gdf_patch=gdf_patch,
                    patch_transform=patch_transform,
                    patch_h=PATCH_SIZE,
                    patch_w=PATCH_SIZE
                )

                overlaid = apply_red_overlay(rgb_patch, line_mask)

                out_name = f"patch_r{row:05d}_c{col:05d}.png"
                save_patch(overlaid, OUT_DIR / out_name)
                tile_count += 1

        print(f"[OK] Saved {tile_count} patches to {OUT_DIR}")


if __name__ == "__main__":
    success = process_with_raw_and_mask()
    if not success:
        print("[INFO] Using cropped map fallback.")
        process_with_cropped_map()