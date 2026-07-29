import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame_gdf = gpd.read_file(args.frame)

    with rasterio.open(args.raster) as src:
        if frame_gdf.crs != src.crs:
            frame_gdf = frame_gdf.to_crs(src.crs)

        geoms = [geom for geom in frame_gdf.geometry if geom is not None and not geom.is_empty]

        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(out_path), "w", **out_meta) as dst:
        dst.write(out_image)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
