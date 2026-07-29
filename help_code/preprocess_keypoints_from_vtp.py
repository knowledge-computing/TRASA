"""
script creates "gt keypoints preprocessed" masks by resampling vtp polylines so node spacing is consistent.

it:
- loads per-patch .vtp polyline(s)
- rescales normalized vtp coords [0,1] -> pixel coords [0, patch_size-1]
- resamples points along each polyline at constant spacing (pixels)
- rasterizes:
  - keypoints_r3_pre (disk radius r)
  - endpoints_r3_pre (disk radius r) for endpoints only (used for ntc viz)
- can sweep multiple spacings in one run and saves to separate output folders per spacing
- optionally writes json files with resampled points (one per patch per spacing)

usage example (single spacing):
  python preprocess_keypoints_from_vtp.py \
    --raw-dir /path/to/raw \
    --vtp-dir /path/to/vtp \
    --out-keypoints-dir /path/to/keypoints_r3_pre \
    --out-endpoints-dir /path/to/endpoints_r3_pre \
    --out-json-dir /path/to/keypoints_pre_json \
    --limit 200 \
    --spacing-px 8 \
    --patch-size 256

usage example (spacing sweep):
  python preprocess_keypoints_from_vtp.py \
    --raw-dir /path/to/raw \
    --vtp-dir /path/to/vtp \
    --out-keypoints-dir /path/to/keypoints_r3_pre \
    --out-endpoints-dir /path/to/endpoints_r3_pre \
    --out-json-dir /path/to/keypoints_pre_json \
    --limit 200 \
    --spacing-px-list 20,16,12,10,8,6,4 \
    --patch-size 256
"""

import os
import json
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw

try:
    import pyvista as pv
except Exception:
    pv = None


def patch_id_to_base(patch_id: int) -> str:
    return f"sample_{int(patch_id):06d}"


def patch_id_to_data_filename(patch_id: int) -> str:
    return f"{patch_id_to_base(patch_id)}_data.png"


def patch_id_to_vtp_filename(patch_id: int) -> str:
    return f"{patch_id_to_base(patch_id)}_graph.vtp"


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


def resample_polyline(points_xy: np.ndarray, spacing_px: float) -> np.ndarray:
    if len(points_xy) < 2:
        return points_xy

    deltas = points_xy[1:] - points_xy[:-1]
    seg_lens = np.sqrt((deltas ** 2).sum(axis=1))
    total = float(seg_lens.sum())
    if total <= 1e-6:
        return points_xy[:1]

    num = max(2, int(math.floor(total / spacing_px)) + 1)
    target_s = np.linspace(0.0, total, num=num)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

    out = []
    j = 0
    for s in target_s:
        while j < len(seg_lens) - 1 and s > cum[j + 1]:
            j += 1
        s0, s1 = cum[j], cum[j + 1]
        t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        p = points_xy[j] * (1.0 - t) + points_xy[j + 1] * t
        out.append(p)

    return np.asarray(out, dtype=np.float32)


def clamp_xy(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    xy = xy.copy()
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    return xy


def draw_disks(points_xy: np.ndarray, w: int, h: int, r: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    for x, y in points_xy:
        x0 = int(round(x - r))
        y0 = int(round(y - r))
        x1 = int(round(x + r))
        y1 = int(round(y + r))
        dr.ellipse([x0, y0, x1, y1], fill=255)
    return np.array(img, dtype=np.uint8)


def load_vtp_polylines(vtp_path: str, patch_size: int) -> list[np.ndarray]:
    if pv is None:
        raise RuntimeError("pyvista is not available. install pyvista/vtk or run in env that has it.")

    mesh = pv.read(vtp_path)
    lines = mesh.lines
    pts = np.asarray(mesh.points, dtype=np.float32)

    polylines = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        idx = lines[i + 1:i + 1 + n]
        i = i + 1 + n

        poly = pts[idx][:, :2]                      # normalized xy in [0,1]
        poly = poly * float(patch_size - 1)         # -> pixel coords [0, patch_size-1]
        polylines.append(poly.astype(np.float32))

    return polylines


def spacing_to_tag(spacing: float) -> str:
    # stable folder tag, e.g., 8 -> s08, 12 -> s12, 4.5 -> s04p5
    if abs(spacing - round(spacing)) < 1e-6:
        return f"s{int(round(spacing)):02d}"
    s = f"{spacing:.2f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    return f"s{s}"


def make_out_dirs(base_keypoints_dir: str, base_endpoints_dir: str, base_json_dir: str, spacing: float):
    tag = spacing_to_tag(spacing)

    kp_dir = base_keypoints_dir
    ep_dir = base_endpoints_dir
    js_dir = base_json_dir

    if base_keypoints_dir.endswith("keypoints_r3_pre") or "keypoints_r3_pre" in base_keypoints_dir:
        kp_dir = base_keypoints_dir.replace("keypoints_r3_pre", f"keypoints_r3_pre_{tag}")
    else:
        kp_dir = os.path.join(base_keypoints_dir, f"keypoints_r3_pre_{tag}")

    if base_endpoints_dir.endswith("endpoints_r3_pre") or "endpoints_r3_pre" in base_endpoints_dir:
        ep_dir = base_endpoints_dir.replace("endpoints_r3_pre", f"endpoints_r3_pre_{tag}")
    else:
        ep_dir = os.path.join(base_endpoints_dir, f"endpoints_r3_pre_{tag}")

    if base_json_dir:
        if base_json_dir.endswith("keypoints_pre_json") or "keypoints_pre_json" in base_json_dir:
            js_dir = base_json_dir.replace("keypoints_pre_json", f"keypoints_pre_json_{tag}")
        else:
            js_dir = os.path.join(base_json_dir, f"keypoints_pre_json_{tag}")

    os.makedirs(kp_dir, exist_ok=True)
    os.makedirs(ep_dir, exist_ok=True)
    if js_dir:
        os.makedirs(js_dir, exist_ok=True)

    return kp_dir, ep_dir, js_dir, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--vtp-dir", required=True)
    ap.add_argument("--out-keypoints-dir", required=True)
    ap.add_argument("--out-endpoints-dir", required=True)
    ap.add_argument("--out-json-dir", default="")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--spacing-px", type=float, default=8.0)
    ap.add_argument("--spacing-px-list", type=str, default="")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--r", type=int, default=3)
    args = ap.parse_args()

    if args.spacing_px_list.strip():
        spacings = [float(x.strip()) for x in args.spacing_px_list.split(",") if x.strip()]
    else:
        spacings = [float(args.spacing_px)]

    patch_ids = list_patch_ids(args.raw_dir, args.limit)
    print(f"found {len(patch_ids)} patches (limit={args.limit})")
    print(f"spacings: {spacings}")

    for spacing in spacings:
        kp_dir, ep_dir, js_dir, tag = make_out_dirs(
            args.out_keypoints_dir,
            args.out_endpoints_dir,
            args.out_json_dir,
            spacing,
        )

        print(f"writing outputs for spacing {spacing} -> {tag}")
        print(f"  keypoints:  {kp_dir}")
        print(f"  endpoints:  {ep_dir}")
        if js_dir:
            print(f"  json:       {js_dir}")

        processed = 0
        for k, pid in enumerate(patch_ids, start=1):
            base = patch_id_to_base(pid)
            vtp_path = os.path.join(args.vtp_dir, patch_id_to_vtp_filename(pid))
            if not os.path.exists(vtp_path):
                continue

            polylines = load_vtp_polylines(vtp_path, args.patch_size)

            all_pts = []
            endpoints = []

            for poly in polylines:
                if len(poly) == 0:
                    continue

                res = resample_polyline(poly, spacing)
                res = clamp_xy(res, args.patch_size, args.patch_size)
                if len(res) == 0:
                    continue

                all_pts.append(res)

                if len(res) >= 2:
                    endpoints.append(res[0:1])
                    endpoints.append(res[-1:])

            if len(all_pts) == 0:
                kp = np.zeros((args.patch_size, args.patch_size), dtype=np.uint8)
                ep = np.zeros_like(kp)
                pts_out = []
                ends_out = []
            else:
                pts = np.concatenate(all_pts, axis=0)
                kp = draw_disks(pts, args.patch_size, args.patch_size, args.r)

                if len(endpoints) > 0:
                    ep_pts = np.concatenate(endpoints, axis=0)
                    ep = draw_disks(ep_pts, args.patch_size, args.patch_size, args.r)
                    ends_out = ep_pts.tolist()
                else:
                    ep = np.zeros_like(kp)
                    ends_out = []

                pts_out = pts.tolist()

            kp_path = os.path.join(kp_dir, f"{base}_graph_keypoints_r3_pre.png")
            ep_path = os.path.join(ep_dir, f"{base}_graph_endpoints_r3_pre.png")

            Image.fromarray(kp).save(kp_path)
            Image.fromarray(ep).save(ep_path)

            if js_dir:
                jpath = os.path.join(js_dir, f"{base}_graph_keypoints_pre.json")
                with open(jpath, "w") as f:
                    json.dump(
                        {
                            "patch_id": pid,
                            "spacing_px": spacing,
                            "points_xy": pts_out,
                            "endpoints_xy": ends_out,
                        },
                        f,
                        indent=2,
                    )

            processed += 1
            if processed % 25 == 0 or processed == len(patch_ids):
                print(f"  {tag}: processed {processed}/{len(patch_ids)}")

        print(f"done for spacing {spacing} ({tag})")

    print("done")


if __name__ == "__main__":
    main()
