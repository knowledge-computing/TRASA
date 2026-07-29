import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np


def drop_probability(row):
    code = row["mtfcc_code"]
    length = row["seg_len"]

    # probability of missing a segment
    probs = {
        "S1100": 0.02,  # primary
        "S1200": 0.08,  # secondary
        "S1400": 0.22,  # local streets
        "S1630": 0.18,  # ramps
        "S1740": 0.35,  # private roads
        "S1500": 0.40   # trails
    }

    p = probs.get(code, 0.20)

    # short segments are much easier to miss
    if length < 0.00015:
        p += 0.25
    elif length < 0.00035:
        p += 0.15
    elif length < 0.00075:
        p += 0.05

    return min(p, 0.95)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    gdf = gpd.read_file(args.input)

    # approximate segment length
    gdf["seg_len"] = gdf.geometry.length

    rng = np.random.RandomState(args.seed)

    gdf["drop_prob"] = gdf.apply(drop_probability, axis=1)

    gdf["keep"] = rng.rand(len(gdf)) > gdf["drop_prob"]

    pred = gdf[gdf["keep"]].copy()

    pred = pred.drop(columns=["seg_len", "drop_prob", "keep"], errors="ignore")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    pred.to_file(args.output, driver="GPKG")

    print("GT features:", len(gdf))
    print("Pred features:", len(pred))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()