"""
script makes fuzzy node probability masks by BLURRING preprocessed keypoints.

this does NOT create new nodes or expand support:
- it only softens existing keypoint disks

usage example:
  python make_keypoints_fuzzy_from_pre.py \
    --keypoints-pre-dir /path/to/keypoints_r3_pre \
    --out-fuzzy-dir /path/to/keypoints_fuzzy_pre \
    --limit 200 \
    --blur-radius 1.5
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageFilter


def list_pngs(d: str, limit: int) -> list[str]:
    names = [n for n in os.listdir(d) if n.endswith(".png")]
    names.sort()
    return names[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints-pre-dir", required=True)
    ap.add_argument("--out-fuzzy-dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--blur-radius", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out_fuzzy_dir, exist_ok=True)

    names = list_pngs(args.keypoints_pre_dir, args.limit)
    print(f"found {len(names)} masks (limit={args.limit})")

    for i, n in enumerate(names, start=1):
        path = os.path.join(args.keypoints_pre_dir, n)

        img = Image.open(path).convert("L")

        # gaussian blur = local fuzziness only
        blurred = img.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))

        # optional: rescale to full 0..255 for visibility
        arr = np.array(blurred, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0

        out = arr.astype(np.uint8)

        out_name = n.replace(".png", "_fuzzy.png")
        out_path = os.path.join(args.out_fuzzy_dir, out_name)
        Image.fromarray(out).save(out_path)

        if i % 25 == 0 or i == len(names):
            print(f"processed {i}/{len(names)}")

    print("done")


if __name__ == "__main__":
    main()
