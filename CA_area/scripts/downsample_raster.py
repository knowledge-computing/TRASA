import argparse
from pathlib import Path

import rasterio
from rasterio.enums import Resampling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()

    with rasterio.open(args.input) as src:
        new_height = max(1, int(src.height * args.scale))
        new_width = max(1, int(src.width * args.scale))

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        new_transform = src.transform * src.transform.scale(
            src.width / float(new_width),
            src.height / float(new_height)
        )

        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(data)

    print("Saved:", out_path)
    print("New size:", new_width, "x", new_height)


if __name__ == "__main__":
    main()
