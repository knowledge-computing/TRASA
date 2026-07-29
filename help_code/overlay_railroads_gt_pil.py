"""
script overlays railroad gt (lines + keypoints) on ldtr patches without using cv2.
it reads:
- raw patches from <raw_dir>
- vtp graphs from <vtp_dir>
- keypoints .npy from <kp_dir>
and saves overlay images to <out_dir>.

assumes file naming like:
- raw/sample_8_5_5_data.png
- vtp/sample_8_5_5_graph.vtp
- keypoints_GT/sample_8_5_5_graph_keypoints.npy

usage example:
python overlay_railroads_gt_pil.py raw_dir vtp_dir kp_dir out_dir
"""

import sys
import os
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

def load_keypoints(kp_path):
    pts = np.load(kp_path)
    return pts

def load_line_segments_from_vtp(vtp_path, h, w):
    mesh = pv.read(vtp_path)
    pts_norm = np.array(mesh.points)[:, :2]
    px = pts_norm[:, 0] * (w - 1)
    py = pts_norm[:, 1] * (h - 1)
    pts_pix = np.stack([px, py], axis=1)

    lines = mesh.lines
    segments = []

    i = 0
    n_lines_array = len(lines)
    while i < n_lines_array:
        npts = int(lines[i])
        idxs = lines[i + 1 : i + 1 + npts]
        i += 1 + npts

        if npts < 2:
            continue

        for j in range(len(idxs) - 1):
            p1 = pts_pix[int(idxs[j])]
            p2 = pts_pix[int(idxs[j + 1])]
            segments.append((p1[0], p1[1], p2[0], p2[1]))

    return segments

def draw_overlay_pil(raw_img_path, segments, keypoints, out_path):
    img = Image.open(raw_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # red lines
    for (x1, y1, x2, y2) in segments:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        draw.line([p1, p2], fill=(255, 0, 0), width=3)

    # yellow keypoints
    r = 3
    for (x, y) in keypoints:
        cx = int(round(x))
        cy = int(round(y))
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, outline=(255, 255, 0), fill=(255, 255, 0))

    img.save(out_path)

def main():
    if len(sys.argv) < 5:
        print("usage: python overlay_railroads_gt_pil.py <raw_dir> <vtp_dir> <kp_dir> <out_dir>")
        sys.exit(1)

    raw_dir = sys.argv[1]
    vtp_dir = sys.argv[2]
    kp_dir = sys.argv[3]
    out_dir = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".png"):
            continue

        raw_base = os.path.splitext(fname)[0]          # sample_8_5_5_data
        vtp_base = raw_base.replace("_data", "_graph") # sample_8_5_5_graph

        raw_path = os.path.join(raw_dir, fname)
        vtp_path = os.path.join(vtp_dir, vtp_base + ".vtp")
        kp_path = os.path.join(kp_dir, vtp_base + "_keypoints.npy")
        out_path = os.path.join(out_dir, raw_base + "_overlay.png")

        if not os.path.exists(vtp_path):
            print(f"skip {fname}: no vtp {vtp_path}")
            continue
        if not os.path.exists(kp_path):
            print(f"skip {fname}: no keypoints {kp_path}")
            continue

        tmp_img = Image.open(raw_path)
        w, h = tmp_img.size
        tmp_img.close()

        segments = load_line_segments_from_vtp(vtp_path, h, w)
        keypoints = load_keypoints(kp_path)

        draw_overlay_pil(raw_path, segments, keypoints, out_path)
        print(f"saved overlay for {fname} -> {out_path}")

if __name__ == "__main__":
    main()
