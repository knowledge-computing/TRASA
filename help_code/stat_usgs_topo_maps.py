"""
script analyzes USGS topographic .tif maps stored in /scratch.global/chen7924/topo_bucket, based on metadata CSVs from http://prd-tnm.s3-website-us-west-2.amazonaws.com/?prefix=StagedProducts/Maps/Metadata

- filters metadata entries matching that scale and having .tif files presented
- saves a CSV with full metadata of all matching maps

python stat_usgs_topo_maps.py <target_scale> | python stat_usgs_topo_maps.py 62500
"""

import os
import sys
import pandas as pd
from collections import defaultdict
from itertools import combinations

# === GET TARGET SCALE FROM COMMAND LINE ===
if len(sys.argv) != 2:
    print("usage: python stat_usgs_topo_maps.py <target_scale>")
    sys.exit(1)

try:
    target_scale = int(sys.argv[1])
except ValueError:
    print("❌ Invalid scale value. Please provide a number like 62500 or 24000.")
    sys.exit(1)

# === CONFIGURATION ===
tif_dir = "/scratch.global/chen7924/topo_bucket"
metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata")

# Metadata files: (relative_path, filename_col, scale_col)
csv_files = [
    ("historicaltopo/historicaltopo.csv", "product_filename", "map_scale"),
    ("ustopo_current/ustopo_current.csv", "product_filename", "map_scale"),
    ("ustopo_historical/ustopo_historical.csv", "product_filename", "map_scale"),
    ("topomaps_all_legacy/topomaps_all.csv", "Download Product S3", "Scale")
]

# === LOAD .tif FILENAMES ===
print(f"📂 Reading .tif files from: {tif_dir}")
tif_files = {f for f in os.listdir(tif_dir) if f.endswith(".tif")}
print(f"✔️  Found {len(tif_files)} .tif files.\n")

# === OVERLAP CHECK PREPROCESSING ===
source_to_tifnames = defaultdict(set)

for rel_path, filename_col, scale_col in csv_files:
    path = os.path.join(metadata_dir, rel_path)
    if not os.path.exists(path):
        continue
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        continue
    if filename_col.startswith("Download"):
        df[filename_col] = df[filename_col].str.extract(r"/([^/]+\.pdf)")
    df["tif_name"] = df[filename_col].str.replace(".pdf", ".tif", regex=False)
    source_to_tifnames[rel_path] = set(df["tif_name"].dropna().unique())

# === OVERLAP REPORT ===
print("🔍 Overlap analysis between metadata sources:")
for (src1, set1), (src2, set2) in combinations(source_to_tifnames.items(), 2):
    overlap = set1 & set2
    print(f"• {src1} ∩ {src2} → {len(overlap)} overlapping maps")
    if overlap:
        print(f"  e.g., {list(overlap)[:3]}\n")

# === FILTERED MATCHES FOR TARGET SCALE ===
matched_df_list = []

for rel_path, filename_col, scale_col in csv_files:
    path = os.path.join(metadata_dir, rel_path)
    if not os.path.exists(path):
        continue
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        continue

    if filename_col.startswith("Download"):
        df[filename_col] = df[filename_col].str.extract(r"/([^/]+\.pdf)")
    df["tif_name"] = df[filename_col].str.replace(".pdf", ".tif", regex=False)

    if scale_col in df.columns:
        df[scale_col] = pd.to_numeric(df[scale_col], errors="coerce")

    # Filter by target scale and matching .tif
    df_filtered = df[(df[scale_col] == target_scale) & (df["tif_name"].isin(tif_files))]
    matched_df_list.append(df_filtered)

# === SAVE AND REPORT ===
if matched_df_list:
    df_all = pd.concat(matched_df_list, ignore_index=True)
    output_path = os.path.join(metadata_dir, f"matched_{target_scale}_maps.csv")
    df_all.to_csv(output_path, index=False)
    print(f"\n💾 Saved full metadata for scale {target_scale} maps to: {output_path}")

    # Size statistics
    size_col = "product_filesize"
    if size_col in df_all.columns:
        df_all[size_col] = pd.to_numeric(df_all[size_col], errors="coerce")
        total_bytes = df_all[size_col].sum(skipna=True)
        avg_bytes = df_all[size_col].mean(skipna=True)

        print(f"\n📦 Storage statistics for scale 1:{target_scale} maps:")
        print(f"• Count: {len(df_all)} maps")
        print(f"• Total size: {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.2f} GB)")
        print(f"• Average size per map: {avg_bytes / (1024**2):.2f} MB")
    else:
        print("⚠️ No filesize data available for statistics.")
else:
    print(f"⚠️ No matches found for scale 1:{target_scale}")
