"""
script analyzes spacing between consecutive nodes in ldtr railroad vtp graphs.

for each <name>.vtp in <vtp_dir> it:
- converts normalized coords to pixel coords (0..255)
- collects lengths of all edges between consecutive polyline vertices
- computes basic stats per patch
- writes a csv file with one row per patch

usage example:
python generate_node_spacing_stats.py \
    /users/4/kirsa002/IMOLA/LDTR_KC/LDTR_mine/datasets/railroads/train_data_g256_noprune/vtp \
    spacing_stats_train.csv
"""

import os
import sys
import csv
import numpy as np
import pyvista as pv

patch_size = 256  # g256 patches

def collect_edge_lengths(vtp_path, size):
    mesh = pv.read(vtp_path)

    pts_norm = np.array(mesh.points)[:, :2]  # x,y in [0,1]
    px = pts_norm[:, 0] * (size - 1)
    py = pts_norm[:, 1] * (size - 1)
    pts_pix = np.stack([px, py], axis=1)

    lines = mesh.lines
    lens = []

    i = 0
    n = len(lines)
    while i < n:
        npts = int(lines[i])
        idxs = lines[i + 1 : i + 1 + npts]
        i += 1 + npts

        if npts < 2:
            continue

        # distances along this polyline
        for j in range(len(idxs) - 1):
            u = int(idxs[j])
            v = int(idxs[j + 1])
            p1 = pts_pix[u]
            p2 = pts_pix[v]
            d = float(np.linalg.norm(p1 - p2))
            lens.append(d)

    return np.array(lens, dtype=float)

def main():
    if len(sys.argv) != 3:
        print("usage: python generate_node_spacing_stats.py <vtp_dir> <out_csv>")
        sys.exit(1)

    vtp_dir = sys.argv[1]
    out_csv = sys.argv[2]

    rows = []
    all_lengths = []

    for fname in sorted(os.listdir(vtp_dir)):
        if not fname.endswith(".vtp"):
            continue

        vtp_path = os.path.join(vtp_dir, fname)
        base = os.path.splitext(fname)[0]

        lens = collect_edge_lengths(vtp_path, patch_size)
        if lens.size == 0:
            print(f"warn: no edges in {fname}")
            continue

        all_lengths.append(lens)

        row = {
            "patch": base,
            "num_segments": int(lens.size),
            "mean": float(np.mean(lens)),
            "median": float(np.median(lens)),
            "min": float(np.min(lens)),
            "max": float(np.max(lens)),
            "p10": float(np.percentile(lens, 10)),
            "p90": float(np.percentile(lens, 90)),
        }
        rows.append(row)
        print(f"processed {fname}")

    # write per-patch stats
    fieldnames = [
        "patch",
        "num_segments",
        "mean",
        "median",
        "min",
        "max",
        "p10",
        "p90",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # optional: print global summary
    if all_lengths:
        all_lens = np.concatenate(all_lengths)
        print(f"global mean spacing: {np.mean(all_lens):.3f}")
        print(f"global median spacing: {np.median(all_lens):.3f}")
        print(f"global p10 / p90: {np.percentile(all_lens,10):.3f} / {np.percentile(all_lens,90):.3f}")

if __name__ == "__main__":
    main()
