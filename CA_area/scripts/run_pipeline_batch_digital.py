import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/users/4/kirsa002/IMOLA/CA_area")
RAW_ROOT = PROJECT_ROOT / "maps" / "digital" / "raw"
OUT_ROOT = PROJECT_ROOT / "outputs" / "digital"
ROADS_MERGED = PROJECT_ROOT / "shared_data" / "california_roads" / "roads_merged.gpkg"

DETECT_FRAME = PROJECT_ROOT / "scripts" / "detect_map_frame.py"
CROP_RASTER = PROJECT_ROOT / "scripts" / "crop_raster_to_frame.py"
DOWNSAMPLE = PROJECT_ROOT / "scripts" / "downsample_raster.py"
MAKE_PRED = PROJECT_ROOT / "scripts" / "prediction_sampled.py"


def run_cmd(cmd, log_prefix):
    print("[RUN]", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("{} failed with code {}".format(log_prefix, result.returncode))
    return result.stdout


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def find_main_tifs(raw_root):
    tif_paths = sorted(raw_root.glob("*/*_TM_geo.tif"))
    return tif_paths


def map_id_from_tif(tif_path):
    return tif_path.stem  # e.g. CA_Torrance_20230726_TM_geo


def frame_layer_name_from_path(frame_gpkg):
    return frame_gpkg.stem


def build_output_paths(map_id):
    base = OUT_ROOT / map_id
    paths = {
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

        "map_cropped": base / "viz" / "map_cropped.tif",
        "map_cropped_small": base / "viz" / "map_cropped_small.tif",

        "prediction": base / "prediction" / "roads_prediction_sampled.gpkg",
        "metadata": base / "metadata" / "pipeline_info.json",
    }
    return paths


def prepare_dirs(paths):
    ensure_dir(paths["frame_dir"])
    ensure_dir(paths["gt_dir"])
    ensure_dir(paths["pred_dir"])
    ensure_dir(paths["viz_dir"])
    ensure_dir(paths["meta_dir"])


def detect_frame_for_map(tif_path, paths):
    # detect_map_frame.py writes next to raster by default, so use --out-dir if supported.
    # Since your current script writes to raster folder by default, we will move outputs after.
    run_cmd(
        ["python3", str(DETECT_FRAME), str(tif_path)],
        "detect_frame"
    )

    raw_dir = tif_path.parent
    raw_stem = tif_path.stem

    produced_bbox = raw_dir / (raw_stem + "_map_frame_bbox.gpkg")
    produced_mask = raw_dir / (raw_stem + "_map_frame_mask.png")

    if not produced_bbox.exists():
        raise RuntimeError("Expected frame bbox not found: {}".format(produced_bbox))
    if not produced_mask.exists():
        raise RuntimeError("Expected frame mask not found: {}".format(produced_mask))

    produced_bbox.replace(paths["frame_bbox"])
    produced_mask.replace(paths["frame_mask"])


def reproject_frame_to_4269(paths):
    layer_name = "map_frame_bbox"
    run_cmd(
        [
            "ogr2ogr", "-f", "GPKG",
            str(paths["frame_bbox_4269"]),
            str(paths["frame_bbox"]),
            "-nln", layer_name,
            "-t_srs", "EPSG:4269"
        ],
        "reproject_frame"
    )


def clip_roads(paths):
    run_cmd(
        [
            "ogr2ogr", "-f", "GPKG",
            str(paths["roads_clip"]),
            str(ROADS_MERGED),
            "-nln", "roads_clip",
            "-clipsrc", str(paths["frame_bbox_4269"])
        ],
        "clip_roads"
    )


def crop_raster(tif_path, paths):
    run_cmd(
        [
            "python3", str(CROP_RASTER),
            "--raster", str(tif_path),
            "--frame", str(paths["frame_bbox"]),
            "--output", str(paths["map_cropped"])
        ],
        "crop_raster"
    )


def downsample_raster(paths, scale):
    run_cmd(
        [
            "python3", str(DOWNSAMPLE),
            "--input", str(paths["map_cropped"]),
            "--output", str(paths["map_cropped_small"]),
            "--scale", str(scale)
        ],
        "downsample_raster"
    )


def make_prediction(paths):
    run_cmd(
        [
            "python3", str(MAKE_PRED),
            "--raster", str(paths["map_cropped_small"]),
            "--roads", str(paths["roads_clip"]),
            "--output", str(paths["prediction"])
        ],
        "make_prediction"
    )


def write_metadata(map_id, tif_path, paths, scale):
    meta = {
        "map_id": map_id,
        "map_type": "digital",
        "raw_raster": str(tif_path),
        "frame_bbox": str(paths["frame_bbox"]),
        "frame_bbox_4269": str(paths["frame_bbox_4269"]),
        "roads_clip": str(paths["roads_clip"]),
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
        paths["frame_bbox_4269"].exists() and
        paths["roads_clip"].exists() and
        paths["map_cropped"].exists() and
        paths["map_cropped_small"].exists() and
        paths["prediction"].exists() and
        paths["metadata"].exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N maps")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--downsample-scale", type=float, default=0.25)
    parser.add_argument("--summary-csv", default=str(PROJECT_ROOT / "logs" / "digital_batch_summary.csv"))
    args = parser.parse_args()

    tif_paths = find_main_tifs(RAW_ROOT)
    if args.limit is not None:
        tif_paths = tif_paths[:args.limit]

    summary_rows = []

    for i, tif_path in enumerate(tif_paths, 1):
        map_id = map_id_from_tif(tif_path)
        paths = build_output_paths(map_id)
        prepare_dirs(paths)

        status = "success"
        message = ""

        print("\n========== [{}/{}] {} ==========".format(i, len(tif_paths), map_id))

        try:
            if already_done(paths) and not args.force:
                status = "skipped_existing"
                message = "All outputs already exist"
            else:
                detect_frame_for_map(tif_path, paths)
                reproject_frame_to_4269(paths)
                clip_roads(paths)
                crop_raster(tif_path, paths)
                downsample_raster(paths, args.downsample_scale)
                make_prediction(paths)
                write_metadata(map_id, tif_path, paths, args.downsample_scale)

        except Exception as e:
            status = "failed"
            message = str(e)
            print("[ERROR]", message)

        summary_rows.append({
            "map_id": map_id,
            "status": status,
            "message": message,
            "raw_raster": str(tif_path),
            "output_dir": str(paths["base"]),
            "prediction": str(paths["prediction"]),
        })

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["map_id", "status", "message", "raw_raster", "output_dir", "prediction"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary to:", summary_csv)


if __name__ == "__main__":
    main()