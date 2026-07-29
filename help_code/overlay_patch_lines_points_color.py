"""
script overlays raw patch with 2px lines (yellow) and 3px keypoints (orange).

usage:
python overlay_patch_lines_points_color.py raw_dir lines_thin_dir keypoints_dir out_dir
"""

import os
import sys
import numpy as np
from PIL import Image

alpha_line = 0.6
alpha_point = 0.8

line_color = np.array([255, 255, 0], dtype=np.float32)    # yellow
point_color = np.array([255, 165, 0], dtype=np.float32)   # orange


def thicken_mask(mask, k=2):
    """
    make binary mask thicker by k pixels (manhattan distance).
    simple dilation without cv2.
    """
    h, w = mask.shape
    out = mask.copy().astype(np.uint8)

    for dy in range(-k, k+1):
        for dx in range(-k, k+1):
            if abs(dx) + abs(dy) <= k:
                shifted = np.zeros_like(out)
                y_start = max(0, dy)
                y_end   = min(h, h + dy)
                x_start = max(0, dx)
                x_end   = min(w, w + dx)
                src_y0  = max(0, -dy)
                src_x0  = max(0, -dx)
                shifted[y_start:y_end, x_start:x_end] = mask[src_y0:src_y0+(y_end-y_start),
                                                             src_x0:src_x0+(x_end-x_start)]
                out = np.maximum(out, shifted)

    return out


def base_from_graph_name(graph_name):
    if graph_name.endswith("_graph"):
        return graph_name[:-6]
    return graph_name


def raw_name_from_base(base):
    return base + "_data.png"


def overlay_single(raw_path, line_mask_path, kp_mask_path, out_path):
    raw_img = Image.open(raw_path).convert("RGB")
    raw_arr = np.array(raw_img).astype(np.float32)

    h, w = raw_arr.shape[:2]

    if os.path.exists(line_mask_path):
        line_arr = np.array(Image.open(line_mask_path).convert("L"))
    else:
        line_arr = np.zeros((h, w), dtype=np.uint8)

    if os.path.exists(kp_mask_path):
        kp_arr = np.array(Image.open(kp_mask_path).convert("L"))
    else:
        kp_arr = np.zeros((h, w), dtype=np.uint8)

    # thicken lines to 2px width
    line_arr = thicken_mask(line_arr > 0, k=2)
    line_bool = line_arr > 0

    kp_bool = kp_arr > 0

    out = raw_arr.copy()

    if line_bool.any():
        mask = line_bool[..., None]
        out = np.where(
            mask,
            (1.0 - alpha_line) * out + alpha_line * line_color,
            out
        )

    if kp_bool.any():
        mask = kp_bool[..., None]
        out = np.where(
            mask,
            (1.0 - alpha_point) * out + alpha_point * point_color,
            out
        )

    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)
    print("wrote", out_path)


def main():
    if len(sys.argv) != 5:
        print("usage: python overlay_patch_lines_points_color.py <raw_dir> <lines_thin_dir> <keypoints_dir> <out_dir>")
        sys.exit(1)

    raw_dir = sys.argv[1]
    lines_dir = sys.argv[2]
    kp_dir = sys.argv[3]
    out_dir = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(lines_dir)):
        if not fname.endswith("_thin.png"):
            continue

        graph_base = os.path.splitext(fname)[0]
        graph_name = graph_base.replace("_thin", "")
        base = base_from_graph_name(graph_name)

        raw_name = raw_name_from_base(base)
        raw_path = os.path.join(raw_dir, raw_name)

        line_mask_path = os.path.join(lines_dir, fname)
        kp_mask_name = graph_name + "_keypoints_r3.png"
        kp_mask_path = os.path.join(kp_dir, kp_mask_name)

        if not os.path.exists(raw_path):
            print("skip, no raw patch:", raw_path)
            continue

        out_name = raw_name.replace(".png", "_overlay_color.png")
        out_path = os.path.join(out_dir, out_name)

        overlay_single(raw_path, line_mask_path, kp_mask_path, out_path)


if __name__ == "__main__":
    main()
