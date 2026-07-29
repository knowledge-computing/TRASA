"""
script creates overlay images from preprocessed keypoints + gt lines.

it writes two overlays:
- pred_ldtr_overlay: yellow lines + orange nodes
- pred_ntc_overlay : green lines + orange nodes + red endpoints

usage example:
  python make_overlays_from_pre.py \
    --raw-dir /path/to/raw \
    --lines-thin-dir /path/to/lines_thin \
    --keypoints-pre-dir /path/to/keypoints_r3_pre \
    --endpoints-pre-dir /path/to/endpoints_r3_pre \
    --out-ldtr-dir /path/to/pred_ldtr_overlay \
    --out-ntc-dir /path/to/pred_ntc_overlay \
    --limit 200
"""

import os
import argparse
import numpy as np
from PIL import Image


def list_patch_ids(raw_dir: str, limit: int) -> list[int]:
    names = [n for n in os.listdir(raw_dir) if n.endswith("_data.png") and n.startswith("sample_")]
    names.sort()
    ids = []
    for n in names:
        pid = int(n.split("_")[1])
        ids.append(pid)
        if len(ids) >= limit:
            break
    return ids


def patch_id_to_base(pid: int) -> str:
    return f"sample_{int(pid):06d}"


def patch_id_to_data_filename(pid: int) -> str:
    return f"{patch_id_to_base(pid)}_data.png"


def load_rgb(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_l(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def overlay(base_rgb: np.ndarray, masks_rgb: list[tuple[np.ndarray, tuple[int, int, int]]], alpha: float) -> np.ndarray:
    out = base_rgb.astype(np.float32)
    for mask_l, color in masks_rgb:
        m = (mask_l > 0).astype(np.float32)[..., None]
        col = np.array(color, dtype=np.float32)[None, None, :]
        out = out * (1.0 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--lines-thin-dir", required=True)
    ap.add_argument("--keypoints-pre-dir", required=True)
    ap.add_argument("--endpoints-pre-dir", required=True)
    ap.add_argument("--out-ldtr-dir", required=True)
    ap.add_argument("--out-ntc-dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out_ldtr_dir, exist_ok=True)
    os.makedirs(args.out_ntc_dir, exist_ok=True)

    patch_ids = list_patch_ids(args.raw_dir, args.limit)
    print(f"found {len(patch_ids)} patches (limit={args.limit})")

    for i, pid in enumerate(patch_ids, start=1):
        base = patch_id_to_base(pid)

        raw_path = os.path.join(args.raw_dir, patch_id_to_data_filename(pid))
        raw = load_rgb(raw_path)
        if raw is None:
            print(f"skip {base}: missing raw {raw_path}")
            continue

        # filenames follow your existing convention for thin line masks
        thin_path = os.path.join(args.lines_thin_dir, f"{base}_graph_thin.png")
        kp_path = os.path.join(args.keypoints_pre_dir, f"{base}_graph_keypoints_r3_pre.png")
        ep_path = os.path.join(args.endpoints_pre_dir, f"{base}_graph_endpoints_r3_pre.png")

        thin = load_l(thin_path)
        kp = load_l(kp_path)
        ep = load_l(ep_path)

        if thin is None or kp is None or ep is None:
            print(f"skip {base}: missing masks (thin/kp/ep)")
            continue

        # pred ldtr: yellow line + orange nodes
        ldtr = overlay(
            raw,
            masks_rgb=[
                (thin, (255, 255, 0)),   # yellow
                (kp,   (255, 165, 0)),   # orange
            ],
            alpha=args.alpha,
        )

        # pred ntc: green line + orange nodes + red endpoints
        ntc = overlay(
            raw,
            masks_rgb=[
                (thin, (0, 255, 0)),     # green
                (kp,   (255, 165, 0)),   # orange
                (ep,   (255, 0, 0)),     # red
            ],
            alpha=args.alpha,
        )

        ldtr_out = os.path.join(args.out_ldtr_dir, f"{base}_data_pred_ldtr_overlay.png")
        ntc_out = os.path.join(args.out_ntc_dir, f"{base}_data_pred_ntc_overlay.png")

        Image.fromarray(ldtr).save(ldtr_out)
        Image.fromarray(ntc).save(ntc_out)

        if i % 25 == 0 or i == len(patch_ids):
            print(f"processed {i}/{len(patch_ids)}")

    print("done")


if __name__ == "__main__":
    main()
