import os
import json
import argparse
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from PIL import Image, ImageDraw
from rasterio.windows import Window
from rasterio.warp import transform
from shapely.geometry import box, LineString, MultiLineString, GeometryCollection


def read_raster_patch(src, window):
    patch = src.read(window=window)

    if patch.shape[0] >= 3:
        patch = patch[:3, :, :]
    elif patch.shape[0] == 1:
        patch = np.repeat(patch, 3, axis=0)

    patch = np.transpose(patch, (1, 2, 0))

    if patch.dtype != np.uint8:
        patch = patch.astype(np.float32)
        min_val = np.nanmin(patch)
        max_val = np.nanmax(patch)

        if max_val > min_val:
            patch = (255 * (patch - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            patch = np.zeros_like(patch, dtype=np.uint8)

    return patch


def save_png(path, rgb_img):
    Image.fromarray(rgb_img).save(path)


def get_window_polygon(src, window):
    left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
    return box(left, bottom, right, top)


def geometry_to_lines(geom):
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, LineString):
        return [geom]

    if isinstance(geom, MultiLineString):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        lines = []
        for g in geom.geoms:
            lines.extend(geometry_to_lines(g))
        return lines

    return []


def map_line_to_patch_pixels(line, src, window):
    """
    Convert clipped LineString map coordinates to patch-local pixel coordinates.
    Boundary points can become 128 for a 128x128 patch, so we clamp instead of dropping.
    """
    coords = []
    inv_transform = ~src.transform

    for x_map, y_map in line.coords:
        col, row = inv_transform * (x_map, y_map)

        local_x = col - window.col_off
        local_y = row - window.row_off

        local_x = int(round(local_x))
        local_y = int(round(local_y))

        if -1 <= local_x <= window.width and -1 <= local_y <= window.height:
            local_x = max(0, min(int(window.width) - 1, local_x))
            local_y = max(0, min(int(window.height) - 1, local_y))
            coords.append([local_x, local_y])

    clean_coords = []
    for pt in coords:
        if not clean_coords or pt != clean_coords[-1]:
            clean_coords.append(pt)

    return clean_coords


def draw_overlay(patch_img, line_entities):
    img = Image.fromarray(patch_img.copy())
    draw = ImageDraw.Draw(img)

    for ent in line_entities:
        coords = ent["coords"]
        if len(coords) >= 2:
            draw.line([tuple(p) for p in coords], fill=(255, 0, 0), width=2)

    return np.array(img)


def make_patch_json(
    patch_id,
    patch_size,
    line_entities,
    source_map,
    source_vector,
    row_off,
    col_off,
    center_lat,
    center_lon,
):
    return {
        "patch_id": patch_id,
        "image_size": [patch_size, patch_size],
        "source_map": source_map,
        "source_vector": source_vector,
        "row_off": int(row_off),
        "col_off": int(col_off),
        "selection_center_lat": center_lat,
        "selection_center_lon": center_lon,
        "line_entities": line_entities,
    }


def get_target_center_row_col(src, center_lat=None, center_lon=None):
    if center_lat is not None and center_lon is not None:
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            [center_lon],
            [center_lat],
        )

        center_x = xs[0]
        center_y = ys[0]

        center_row, center_col = src.index(center_x, center_y)

        print("Using user-selected QGIS coordinate for reference patches:")
        print(f"  lat: {center_lat}")
        print(f"  lon: {center_lon}")
        print(f"  transformed raster CRS x/y: {center_x}, {center_y}")
        print(f"  raster row/col: {center_row}, {center_col}")

        return center_row, center_col

    center_row = src.height / 2
    center_col = src.width / 2

    print("No QGIS coordinate provided. Using raster middle for reference patches:")
    print(f"  raster row/col: {center_row}, {center_col}")

    return center_row, center_col


def make_reference_patch_positions(src, patch_size, stride, center_lat=None, center_lon=None):
    patch_positions = []

    center_row, center_col = get_target_center_row_col(
        src=src,
        center_lat=center_lat,
        center_lon=center_lon,
    )

    for row_off in range(0, src.height - patch_size + 1, stride):
        for col_off in range(0, src.width - patch_size + 1, stride):
            patch_center_row = row_off + patch_size / 2
            patch_center_col = col_off + patch_size / 2

            dist_to_center = (
                (patch_center_row - center_row) ** 2
                + (patch_center_col - center_col) ** 2
            ) ** 0.5

            patch_positions.append((dist_to_center, row_off, col_off))

    patch_positions.sort(key=lambda x: x[0])
    return patch_positions


def make_diverse_inference_patch_positions(src, patch_size, stride, margin_px):
    """
    Create inference candidate positions across the whole map, but avoid boundary.
    Later we greedily select patches that are far apart from each other.
    """
    patch_positions = []

    min_row = margin_px
    max_row = src.height - patch_size - margin_px
    min_col = margin_px
    max_col = src.width - patch_size - margin_px

    if max_row <= min_row or max_col <= min_col:
        print("WARNING: margin is too large for raster size. Using no margin.")
        min_row = 0
        max_row = src.height - patch_size
        min_col = 0
        max_col = src.width - patch_size

    raster_center_row = src.height / 2
    raster_center_col = src.width / 2

    for row_off in range(int(min_row), int(max_row) + 1, stride):
        for col_off in range(int(min_col), int(max_col) + 1, stride):
            patch_center_row = row_off + patch_size / 2
            patch_center_col = col_off + patch_size / 2

            dist_to_map_center = (
                (patch_center_row - raster_center_row) ** 2
                + (patch_center_col - raster_center_col) ** 2
            ) ** 0.5

            patch_positions.append((dist_to_map_center, row_off, col_off))

    return patch_positions


def get_line_entities_for_patch(roads, roads_sindex, src, window, min_line_points):
    patch_poly = get_window_polygon(src, window)

    candidate_idx = list(roads_sindex.intersection(patch_poly.bounds))
    if not candidate_idx:
        return []

    candidate_roads = roads.iloc[candidate_idx]
    candidate_roads = candidate_roads[candidate_roads.intersects(patch_poly)]

    if len(candidate_roads) == 0:
        return []

    line_entities = []

    for _, road_row in candidate_roads.iterrows():
        geom = road_row.geometry
        if geom is None or geom.is_empty:
            continue

        clipped = geom.intersection(patch_poly)

        for line in geometry_to_lines(clipped):
            coords = map_line_to_patch_pixels(line, src, window)

            if len(coords) >= min_line_points:
                line_entities.append({
                    "type": "road",
                    "coords": coords,
                })

    return line_entities


def patch_center(row_off, col_off, patch_size):
    return (
        row_off + patch_size / 2,
        col_off + patch_size / 2,
    )


def distance_between_patch_centers(a, b, patch_size):
    ar, ac = patch_center(a[0], a[1], patch_size)
    br, bc = patch_center(b[0], b[1], patch_size)

    return ((ar - br) ** 2 + (ac - bc) ** 2) ** 0.5


def is_far_enough(row_off, col_off, selected_positions, patch_size, min_distance):
    for selected_row, selected_col in selected_positions:
        d = distance_between_patch_centers(
            (row_off, col_off),
            (selected_row, selected_col),
            patch_size,
        )

        if d < min_distance:
            return False

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raster", required=True)
    parser.add_argument("--vector", default="/users/4/kirsa002/IMOLA/requests/roads_clip_raster_crs.gpkg")
    parser.add_argument("--layer", default="roads_clip_raster_crs")
    parser.add_argument("--out_dir", default="/users/4/kirsa002/IMOLA/requests/sam3_sample_BeverlyHills_1995")

    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--num_reference", type=int, default=5)
    parser.add_argument("--num_inference", type=int, default=20)

    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--inference_stride", type=int, default=128)

    parser.add_argument("--min_line_points", type=int, default=2)

    parser.add_argument("--center_lat", type=float, default=None)
    parser.add_argument("--center_lon", type=float, default=None)

    parser.add_argument(
        "--inference_margin_px",
        type=int,
        default=512,
        help="Avoid inference patches closer than this many pixels to map boundary",
    )

    parser.add_argument(
        "--min_inference_distance_px",
        type=int,
        default=700,
        help="Minimum distance between selected inference patch centers",
    )

    parser.add_argument(
        "--min_distance_from_reference_px",
        type=int,
        default=400,
        help="Avoid inference patches too close to selected reference patches",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    reference_dir = out_dir / "reference"
    inference_dir = out_dir / "inference_patches"
    debug_dir = out_dir / "debug_overlays"

    reference_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Reading vector roads...")
    roads = gpd.read_file(args.vector, layer=args.layer)

    print(f"Vector CRS: {roads.crs}")
    print(f"Number of vector features: {len(roads)}")

    reference_saved = 0
    inference_saved = 0

    reference_positions = []
    inference_positions = []

    manifest = {
        "description": "Small SAM3 sample dataset for road-line extraction",
        "note": "Reference images are raw map patches. JSON files contain road-line pixel coordinates. Debug overlays are visualization only.",
        "raster": args.raster,
        "vector": args.vector,
        "vector_layer": args.layer,
        "patch_size": args.patch_size,
        "reference_stride": args.stride,
        "inference_stride": args.inference_stride,
        "reference_selection": "reference patches are searched from the selected QGIS coordinate outward",
        "inference_selection": "inference patches are selected from diverse road-containing areas, away from map boundary and away from reference patches",
        "center_lat": args.center_lat,
        "center_lon": args.center_lon,
        "inference_margin_px": args.inference_margin_px,
        "min_inference_distance_px": args.min_inference_distance_px,
        "min_distance_from_reference_px": args.min_distance_from_reference_px,
        "reference_examples": [],
        "inference_patches": [],
    }

    print("Opening raster...")
    with rasterio.open(args.raster) as src:
        print(f"Raster CRS: {src.crs}")
        print(f"Raster size: {src.width} x {src.height}")
        print("Raster bounds:", src.bounds)
        print("Raster transform:", src.transform)
        print("Vector bounds before possible reprojection:", roads.total_bounds)

        if roads.crs != src.crs:
            print("Reprojecting vector roads to raster CRS...")
            roads = roads.to_crs(src.crs)

        print("Vector bounds after possible reprojection:", roads.total_bounds)

        roads_sindex = roads.sindex

        # 1. Reference patches: keep old behavior, near selected coordinate
        reference_candidates = make_reference_patch_positions(
            src=src,
            patch_size=args.patch_size,
            stride=args.stride,
            center_lat=args.center_lat,
            center_lon=args.center_lon,
        )

        print(f"Number of reference candidate positions: {len(reference_candidates)}")
        print("Selecting reference patches from chosen coordinate outward...")

        for dist_to_center, row_off, col_off in reference_candidates:
            if reference_saved >= args.num_reference:
                break

            window = Window(col_off, row_off, args.patch_size, args.patch_size)

            line_entities = get_line_entities_for_patch(
                roads=roads,
                roads_sindex=roads_sindex,
                src=src,
                window=window,
                min_line_points=args.min_line_points,
            )

            if len(line_entities) == 0:
                continue

            patch_img = read_raster_patch(src, window)

            ref_id = f"BeverlyHills_ref_patch_{reference_saved + 1:03d}"

            ref_img_path = reference_dir / f"{ref_id}.png"
            ref_json_path = reference_dir / f"{ref_id}_lines.json"
            ref_overlay_path = debug_dir / f"{ref_id}_debug_overlay.png"

            save_png(ref_img_path, patch_img)

            patch_json = make_patch_json(
                patch_id=ref_id,
                patch_size=args.patch_size,
                line_entities=line_entities,
                source_map=os.path.basename(args.raster),
                source_vector=os.path.basename(args.vector),
                row_off=row_off,
                col_off=col_off,
                center_lat=args.center_lat,
                center_lon=args.center_lon,
            )

            with open(ref_json_path, "w") as f:
                json.dump(patch_json, f, indent=2)

            overlay = draw_overlay(patch_img, line_entities)
            save_png(ref_overlay_path, overlay)

            manifest["reference_examples"].append({
                "image": str(ref_img_path),
                "json": str(ref_json_path),
                "debug_overlay": str(ref_overlay_path),
                "num_line_entities": len(line_entities),
                "row_off": int(row_off),
                "col_off": int(col_off),
                "distance_to_selected_center_px": float(dist_to_center),
            })

            reference_positions.append((row_off, col_off))
            reference_saved += 1

            print(
                f"Saved reference example {reference_saved}: "
                f"row={row_off}, col={col_off}, "
                f"distance={dist_to_center:.2f}px, "
                f"lines={len(line_entities)}"
            )

        # 2. Inference patches: diverse across map, away from boundary/reference
        inference_candidates = make_diverse_inference_patch_positions(
            src=src,
            patch_size=args.patch_size,
            stride=args.inference_stride,
            margin_px=args.inference_margin_px,
        )

        print(f"Number of inference candidate positions: {len(inference_candidates)}")
        print("Selecting diverse inference patches away from boundary/reference...")

        # Shuffle with fixed seed so result is diverse but reproducible
        rng = np.random.default_rng(42)
        rng.shuffle(inference_candidates)

        # Sort shuffled candidates into a weak map-wide order by keeping randomness.
        # Greedy distance check gives spatial diversity.
        for _, row_off, col_off in inference_candidates:
            if inference_saved >= args.num_inference:
                break

            if not is_far_enough(
                row_off=row_off,
                col_off=col_off,
                selected_positions=reference_positions,
                patch_size=args.patch_size,
                min_distance=args.min_distance_from_reference_px,
            ):
                continue

            if not is_far_enough(
                row_off=row_off,
                col_off=col_off,
                selected_positions=inference_positions,
                patch_size=args.patch_size,
                min_distance=args.min_inference_distance_px,
            ):
                continue

            window = Window(col_off, row_off, args.patch_size, args.patch_size)

            line_entities = get_line_entities_for_patch(
                roads=roads,
                roads_sindex=roads_sindex,
                src=src,
                window=window,
                min_line_points=args.min_line_points,
            )

            if len(line_entities) == 0:
                continue

            patch_img = read_raster_patch(src, window)

            inf_id = f"BeverlyHills_inference_patch_{inference_saved + 1:03d}"
            inf_img_path = inference_dir / f"{inf_id}.png"

            save_png(inf_img_path, patch_img)

            manifest["inference_patches"].append({
                "image": str(inf_img_path),
                "row_off": int(row_off),
                "col_off": int(col_off),
                "num_line_entities_hidden_check": len(line_entities),
            })

            inference_positions.append((row_off, col_off))
            inference_saved += 1

            print(
                f"Saved inference patch {inference_saved}: "
                f"row={row_off}, col={col_off}, "
                f"lines={len(line_entities)}"
            )

        if inference_saved < args.num_inference:
            print("\nWARNING: Could not find enough diverse inference patches with current distance settings.")
            print("Trying relaxed distance settings...")

            relaxed_min_distance = max(args.patch_size * 2, args.min_inference_distance_px // 2)

            for _, row_off, col_off in inference_candidates:
                if inference_saved >= args.num_inference:
                    break

                if (row_off, col_off) in inference_positions:
                    continue

                if not is_far_enough(
                    row_off=row_off,
                    col_off=col_off,
                    selected_positions=reference_positions,
                    patch_size=args.patch_size,
                    min_distance=args.min_distance_from_reference_px,
                ):
                    continue

                if not is_far_enough(
                    row_off=row_off,
                    col_off=col_off,
                    selected_positions=inference_positions,
                    patch_size=args.patch_size,
                    min_distance=relaxed_min_distance,
                ):
                    continue

                window = Window(col_off, row_off, args.patch_size, args.patch_size)

                line_entities = get_line_entities_for_patch(
                    roads=roads,
                    roads_sindex=roads_sindex,
                    src=src,
                    window=window,
                    min_line_points=args.min_line_points,
                )

                if len(line_entities) == 0:
                    continue

                patch_img = read_raster_patch(src, window)

                inf_id = f"BeverlyHills_inference_patch_{inference_saved + 1:03d}"
                inf_img_path = inference_dir / f"{inf_id}.png"

                save_png(inf_img_path, patch_img)

                manifest["inference_patches"].append({
                    "image": str(inf_img_path),
                    "row_off": int(row_off),
                    "col_off": int(col_off),
                    "num_line_entities_hidden_check": len(line_entities),
                    "selection_note": "selected after relaxed diversity distance",
                })

                inference_positions.append((row_off, col_off))
                inference_saved += 1

                print(
                    f"Saved inference patch {inference_saved} after relaxation: "
                    f"row={row_off}, col={col_off}, "
                    f"lines={len(line_entities)}"
                )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    readme_path = out_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(
            "SAM3 sample dataset for Leeje\n"
            "================================\n\n"
            "This sample was created from the Beverly Hills 1995 1:24000 map.\n\n"
            "reference/\n"
            "  Raw 128x128 map patches used as SAM3 reference examples.\n"
            "  Each reference patch has a matching *_lines.json file.\n"
            "  The JSON stores road-line coordinates in patch-local pixel coordinates.\n\n"
            "inference_patches/\n"
            "  Raw 128x128 map patches used as SAM3 inference inputs.\n"
            "  These are not overlaid and do not include JSON annotations.\n"
            "  In this version, inference patches are selected from diverse road-containing areas across the map.\n"
            "  They are also kept away from the map boundary and away from the selected reference patches.\n\n"
            "debug_overlays/\n"
            "  Visualization only.\n"
            "  These images show the road-line coordinates drawn on top of the raw reference patch.\n"
            "  These are not model inputs.\n\n"
            "manifest.json\n"
            "  Summary of generated files, including row/column offsets.\n"
        )

    print("\nDone.")
    print(f"Output folder: {out_dir}")
    print(f"Reference examples saved: {reference_saved}")
    print(f"Inference patches saved: {inference_saved}")
    print(f"Manifest: {manifest_path}")
    print(f"README: {readme_path}")


if __name__ == "__main__":
    main()