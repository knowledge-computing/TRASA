"""
script batch-extracts keypoints from all vtp files in a folder.
for each <name>.vtp it saves:
- <name>_keypoints.npy : array of (x_pixel, y_pixel)
- <name>_keypoints.png : binary mask with keypoints

usage example:
python extract_keypoints_batch.py vtp_dir out_keypoints_dir
"""

import sys
import os
import numpy as np
import pyvista as pv
from PIL import Image

def extract_keypoints(vtp_path):
    mesh = pv.read(vtp_path)
    pts = np.array(mesh.points)[:, :2]   # normalized x,y in 0..1
    return pts

def normalize_to_pixels(pts, h, w):
    px = pts[:, 0] * (w - 1)
    py = pts[:, 1] * (h - 1)
    return np.stack([px, py], axis=1)

def make_keypoint_mask(pts_pix, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in pts_pix:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            mask[yi, xi] = 255
    return mask

def main():
    if len(sys.argv) != 3:
        print("usage: python extract_keypoints_batch.py <vtp_dir> <out_dir>")
        sys.exit(1)

    vtp_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    h, w = 256, 256   # g256 patches

    for fname in sorted(os.listdir(vtp_dir)):
        if not fname.endswith(".vtp"):
            continue

        in_path = os.path.join(vtp_dir, fname)
        base = os.path.splitext(fname)[0]

        out_npy = os.path.join(out_dir, base + "_keypoints.npy")
        out_png = os.path.join(out_dir, base + "_keypoints.png")

        pts_norm = extract_keypoints(in_path)
        pts_pix = normalize_to_pixels(pts_norm, h, w)

        np.save(out_npy, pts_pix)

        mask = make_keypoint_mask(pts_pix, h, w)
        Image.fromarray(mask).save(out_png)

        print(f"processed {fname}")

if __name__ == "__main__":
    main()
