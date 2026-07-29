"""
script converts vtp graph nodes into a keypoint mask made of small circles.

for each <name>.vtp it writes:
<output_dir>/<name>_keypoints_r3.png

usage example:
python vtp_to_keypoints_mask.py vtp_dir out_dir
"""

import os
import sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


patch_size = 256       # for g256 dataset
radius = 3             # circle radius in pixels


def load_keypoints(vtp_path, size):
    mesh = pv.read(vtp_path)
    pts = np.array(mesh.points)[:, :2]  # normalized x,y
    px = pts[:,0] * (size - 1)
    py = pts[:,1] * (size - 1)
    pts_pix = np.stack([px, py], axis=1)
    return pts_pix


def draw_keypoints(size, pts, r):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    for (x, y) in pts:
        cx = int(round(x))
        cy = int(round(y))
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, fill=255)

    return img


def main():
    if len(sys.argv) != 3:
        print("usage: python vtp_to_keypoints_mask.py <vtp_dir> <out_dir>")
        sys.exit(1)

    vtp_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(vtp_dir)):
        if not fname.endswith(".vtp"):
            continue

        vtp_path = os.path.join(vtp_dir, fname)
        base = os.path.splitext(fname)[0]

        pts = load_keypoints(vtp_path, patch_size)
        img = draw_keypoints(patch_size, pts, radius)

        out_path = os.path.join(out_dir, base + "_keypoints_r3.png")
        img.save(out_path)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
