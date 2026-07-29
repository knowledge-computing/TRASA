"""
script converts a vtp polyline graph into a 1-pixel raster line mask.

it reads <input_dir>/<name>.vtp and writes
<output_dir>/<name>_thin.png

usage example:
python vtp_to_raster_lines.py vtp_dir out_dir
"""

import os
import sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


patch_size = 256  # g256 patches


def load_line_segments(vtp_path, size):
    mesh = pv.read(vtp_path)
    pts = np.array(mesh.points)[:, :2]   # normalized x,y
    px = pts[:,0] * (size - 1)
    py = pts[:,1] * (size - 1)
    pts_pix = np.stack([px, py], axis=1)

    segs = []
    lines = mesh.lines
    i = 0
    n = len(lines)

    while i < n:
        npts = int(lines[i])
        idxs = lines[i+1 : i+1+npts]
        i += 1 + npts

        if npts < 2:
            continue

        for j in range(npts - 1):
            u = int(idxs[j])
            v = int(idxs[j+1])
            x1, y1 = pts_pix[u]
            x2, y2 = pts_pix[v]
            segs.append((x1, y1, x2, y2))

    return segs


def draw_segments(size, segs):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in segs:
        draw.line(
            (int(round(x1)), int(round(y1)),
             int(round(x2)), int(round(y2))),
            fill=255,
            width=1
        )
    return img


def main():
    if len(sys.argv) != 3:
        print("usage: python vtp_to_raster_lines.py <vtp_dir> <out_dir>")
        sys.exit(1)

    vtp_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(vtp_dir)):
        if not fname.endswith(".vtp"):
            continue

        vtp_path = os.path.join(vtp_dir, fname)
        base = os.path.splitext(fname)[0]

        segs = load_line_segments(vtp_path, patch_size)
        img = draw_segments(patch_size, segs)

        out_path = os.path.join(out_dir, base + "_thin.png")
        img.save(out_path)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
