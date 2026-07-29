"""
script creates fuzzy versions of 1px line masks and 3px keypoint masks
using gaussian blur.

for each <name>_thin.png in lines_thin_dir it writes:
    <name>_thin_fuzzy.png  in out_lines_dir

for each <name>_keypoints_r3.png in keypoints_r3_dir it writes:
    <name>_keypoints_r3_fuzzy.png in out_kp_dir

usage example:
python make_fuzzy_masks.py lines_thin_dir keypoints_r3_dir out_lines_dir out_kp_dir
"""

import os
import sys
from PIL import Image, ImageFilter


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def make_fuzzy(src_path, dst_path, blur_radius):
    img = Image.open(src_path).convert("L")
    fuzzy = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    fuzzy.save(dst_path)


def main():
    if len(sys.argv) != 5:
        print("usage: python make_fuzzy_masks.py <lines_thin_dir> <keypoints_r3_dir> <out_lines_dir> <out_kp_dir>")
        sys.exit(1)

    thin_dir = sys.argv[1]
    kp_dir = sys.argv[2]
    out_thin = sys.argv[3]
    out_kp = sys.argv[4]

    make_dir(out_thin)
    make_dir(out_kp)

    # process thin line masks
    for fname in sorted(os.listdir(thin_dir)):
        if not fname.endswith("_thin.png"):
            continue

        src = os.path.join(thin_dir, fname)
        dst = os.path.join(out_thin, fname.replace("_thin.png", "_thin_fuzzy.png"))
        make_fuzzy(src, dst, blur_radius=1)
        print("wrote", dst)

    # process keypoint masks
    for fname in sorted(os.listdir(kp_dir)):
        if not fname.endswith("_keypoints_r3.png"):
            continue

        src = os.path.join(kp_dir, fname)
        dst = os.path.join(out_kp, fname.replace("_keypoints_r3.png", "_keypoints_r3_fuzzy.png"))
        make_fuzzy(src, dst, blur_radius=1)
        print("wrote", dst)


if __name__ == "__main__":
    main()
    