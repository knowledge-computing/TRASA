#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area")
RAW_ROOT = PROJECT_ROOT / "maps" / "scanned" / "raw"
OUT_ROOT = PROJECT_ROOT / "outputs" / "scanned"
ROADS_MERGED = PROJECT_ROOT / "shared_data" / "california_roads" / "roads_merged.gpkg"

DETECT_FRAME = PROJECT_ROOT / "scripts" / "detect_map_frame.py"
CROP_RASTER = PROJECT_ROOT / "scripts" / "crop_raster_to_frame.py"
DOWNSAMPLE = PROJECT_ROOT / "scripts" / "downsample_raster.py"
MAKE_PRED = PROJECT_ROOT / "scripts" / "prediction_sampled.py"


def run_cmd(cmd, log_prefix, map_id):
    print(f"[{map_id}] [RUN] " + " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if result.stdout:
        print(f"[{map_id}] {result.stdout}", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{log_prefix} failed with code {result.returncode}")
    return result.stdout


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def find_main_tifs(raw_root):
    return sorted(raw_root.glob("*/*_geo.tif"))


def map_id_from_tif(tif_path):
    return tif_path.stem


def build_output_paths(map_id):
    base = OUT_ROOT / map_id
    return {
        "base": base,
        "frame_dir": base / "frame",
        "gt_dir": base / "gt",
        "pred_dir": base / "prediction",
        "viz_dir": base / "viz",
        "meta_dir": base / "metadata",

        "frame_bbox": base / "frame" / "map_frame_bbox.gpkg",
        "frame_mask": base / "frame" / "map_frame_mask.png",
        "frame_bbox_4269": base / "frame" / "map_frame_bbox_4269.gpkg",

        "roads_clip": base / "gt" / "roads_clip.gpkg",
        "roads_clip_raster_crs": base / "prediction" / "roads_clip_raster_crs.gpkg",

        "map_cropped": base / "viz" / "map_cropped.tif",
        "map_cropped_small": base / "viz" / "map_cropped_small.tif",

        "prediction": base / "prediction" / "roads_prediction_sampled.gpkg",
        "metadata": base / "metadata" / "pipeline_info.json",
    }

def prepare_dirs(paths):
    ensure_dir(paths["frame_dir"])
    ensure_dir(paths["gt_dir"])
    ensure_dir(paths["pred_dir"])
    ensure_dir(paths["viz_dir"])
    ensure_dir(paths["meta_dir"])


def detect_frame_for_map(tif_path, paths, map_id, force=False):
    if paths["frame_bbox"].exists() and paths["frame_mask"].exists() and not force:
        return

    raw_dir = tif_path.parent
    raw_stem = tif_path.stem
    produced_bbox = raw_dir / f"{raw_stem}_map_frame_bbox.gpkg"
    produced_mask = raw_dir / f"{raw_stem}_map_frame_mask.png"

    if produced_bbox.exists():
        produced_bbox.unlink()
    if produced_mask.exists():
        produced_mask.unlink()

    run_cmd(
        ["python3", str(DETECT_FRAME), str(tif_path)],
        "detect_frame",
        map_id,
    )

    if not produced_bbox.exists():
        raise RuntimeError(f"Expected frame bbox not found: {produced_bbox}")
    if not produced_mask.exists():
        raise RuntimeError(f"Expected frame mask not found: {produced_mask}")

    if paths["frame_bbox"].exists():
        paths["frame_bbox"].unlink()
    if paths["frame_mask"].exists():
        paths["frame_mask"].unlink()

    produced_bbox.replace(paths["frame_bbox"])
    produced_mask.replace(paths["frame_mask"])


def reproject_frame_to_4269(paths, map_id, force=False):
    if paths["frame_bbox_4269"].exists() and not force:
        return

    if paths["frame_bbox_4269"].exists():
        paths["frame_bbox_4269"].unlink()

    run_cmd(
        [
            "ogr2ogr", "-f", "GPKG",
            str(paths["frame_bbox_4269"]),
            str(paths["frame_bbox"]),
            "-nln", "map_frame_bbox",
            "-t_srs", "EPSG:4269",
        ],
        "reproject_frame",
        map_id,
    )


def clip_roads(paths, map_id, force=False):
    if paths["roads_clip"].exists() and not force:
        return

    if paths["roads_clip"].exists():
        paths["roads_clip"].unlink()

    run_cmd(
        [
            "ogr2ogr", "-f", "GPKG",
            str(paths["roads_clip"]),
            str(ROADS_MERGED),
            "-nln", "roads_clip",
            "-clipsrc", str(paths["frame_bbox_4269"]),
        ],
        "clip_roads",
        map_id,
    )


def crop_raster(tif_path, paths, map_id, force=False):
    if paths["map_cropped"].exists() and not force:
        return

    if paths["map_cropped"].exists():
        paths["map_cropped"].unlink()

    run_cmd(
        [
            "python3", str(CROP_RASTER),
            "--raster", str(tif_path),
            "--frame", str(paths["frame_bbox"]),
            "--output", str(paths["map_cropped"]),
        ],
        "crop_raster",
        map_id,
    )


def downsample_raster(paths, scale, map_id, force=False):
    if paths["map_cropped_small"].exists() and not force:
        return

    if paths["map_cropped_small"].exists():
        paths["map_cropped_small"].unlink()

    run_cmd(
        [
            "python3", str(DOWNSAMPLE),
            "--input", str(paths["map_cropped"]),
            "--output", str(paths["map_cropped_small"]),
            "--scale", str(scale),
        ],
        "downsample_raster",
        map_id,
    )


def make_prediction(paths, map_id, force=False):
    pred_exists = paths["prediction"].exists()
    gt_raster_exists = paths["roads_clip_raster_crs"].exists()

    if pred_exists and gt_raster_exists and not force:
        return

    if paths["prediction"].exists():
        paths["prediction"].unlink()

    if paths["roads_clip_raster_crs"].exists():
        paths["roads_clip_raster_crs"].unlink()

    run_cmd(
        [
            "python3", str(MAKE_PRED),
            "--raster", str(paths["map_cropped_small"]),
            "--roads", str(paths["roads_clip"]),
            "--output", str(paths["prediction"]),
            "--output-crs", "raster",
            "--save-roads-raster-crs",
        ],
        "make_prediction",
        map_id,
    )


def write_metadata(map_id, tif_path, paths, scale, force=False):
    if paths["metadata"].exists() and not force:
        return

    meta = {
    "map_id": map_id,
    "map_type": "scanned",
    "raw_raster": str(tif_path),
    "frame_bbox": str(paths["frame_bbox"]),
    "frame_bbox_4269": str(paths["frame_bbox_4269"]),
    "roads_clip": str(paths["roads_clip"]),
    "roads_clip_raster_crs": str(paths["roads_clip_raster_crs"]),
    "map_cropped": str(paths["map_cropped"]),
    "map_cropped_small": str(paths["map_cropped_small"]),
    "prediction": str(paths["prediction"]),
    "roads_master": str(ROADS_MERGED),
    "prediction_method": "roads_prediction_sampled",
    "downsample_scale": scale,
}

    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)


def already_done(paths):
    return (
        paths["frame_bbox"].exists() and
        paths["frame_mask"].exists() and
        paths["frame_bbox_4269"].exists() and
        paths["roads_clip"].exists() and
        paths["roads_clip_raster_crs"].exists() and
        paths["map_cropped"].exists() and
        paths["map_cropped_small"].exists() and
        paths["prediction"].exists() and
        paths["metadata"].exists()
    )


def process_one_map(tif_path, downsample_scale, force=False):
    map_id = map_id_from_tif(tif_path)
    paths = build_output_paths(map_id)
    prepare_dirs(paths)

    status = "success"
    message = ""

    print(f"\n========== {map_id} ==========", flush=True)

    try:
        if already_done(paths) and not force:
            status = "skipped_existing"
            message = "All outputs already exist"
            print(f"[{map_id}] skipped_existing", flush=True)
        else:
            detect_frame_for_map(tif_path, paths, map_id, force=force)
            reproject_frame_to_4269(paths, map_id, force=force)
            clip_roads(paths, map_id, force=force)
            crop_raster(tif_path, paths, map_id, force=force)
            downsample_raster(paths, downsample_scale, map_id, force=force)
            make_prediction(paths, map_id, force=force)
            write_metadata(map_id, tif_path, paths, downsample_scale, force=force)

    except Exception as e:
        status = "failed"
        message = str(e)
        print(f"[{map_id}] [ERROR] {message}", flush=True)

    return {
        "map_id": map_id,
        "status": status,
        "message": message,
        "raw_raster": str(tif_path),
        "output_dir": str(paths["base"]),
        "prediction": str(paths["prediction"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N maps after filtering")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--downsample-scale", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1, help="Number of maps to process in parallel")
    parser.add_argument("--map-id", type=str, default=None, help="Process only one map_id")
    parser.add_argument(
        "--summary-csv",
        default=str(PROJECT_ROOT / "logs" / "scanned_batch_summary.csv"),
    )
    args = parser.parse_args()

    if shutil.which("ogr2ogr") is None:
        raise RuntimeError("ogr2ogr not found in PATH. Load GDAL first.")

    for script_path in [DETECT_FRAME, CROP_RASTER, DOWNSAMPLE, MAKE_PRED]:
        if not script_path.exists():
            raise RuntimeError(f"Required script not found: {script_path}")

    tif_paths = find_main_tifs(RAW_ROOT)

    if args.map_id is not None:
        tif_paths = [p for p in tif_paths if map_id_from_tif(p) == args.map_id]
        if not tif_paths:
            raise ValueError(f"--map-id not found: {args.map_id}")

    if args.limit is not None:
        tif_paths = tif_paths[:args.limit]

    if not tif_paths:
        raise RuntimeError("No scanned maps found to process.")

    summary_rows = []

    if args.workers == 1:
        for tif_path in tif_paths:
            row = process_one_map(
                tif_path,
                downsample_scale=args.downsample_scale,
                force=args.force,
            )
            summary_rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    process_one_map,
                    tif_path,
                    args.downsample_scale,
                    args.force,
                ): tif_path
                for tif_path in tif_paths
            }
            for fut in as_completed(futures):
                summary_rows.append(fut.result())

    summary_rows.sort(key=lambda x: x["map_id"])

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["map_id", "status", "message", "raw_raster", "output_dir", "prediction"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary to:", summary_csv, flush=True)


if __name__ == "__main__":
    main()