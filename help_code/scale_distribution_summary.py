import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# === CONFIGURATION ===
tif_dir = "/scratch.global/chen7924/topo_bucket"
metadata_dir = os.path.join(os.path.dirname(__file__), "..", "metadata")

csv_files = [
    ("historicaltopo/historicaltopo.csv", "product_filename", "map_scale"),
    ("ustopo_current/ustopo_current.csv", "product_filename", "map_scale"),
    ("ustopo_historical/ustopo_historical.csv", "product_filename", "map_scale"),
    ("topomaps_all_legacy/topomaps_all.csv", "Download Product S3", "Scale")
]

# === LOAD ALL TIF NAMES ===
print(f"📂 Reading .tif files from: {tif_dir}")
tif_files = {f for f in os.listdir(tif_dir) if f.endswith(".tif")}
print(f"✔️  Found {len(tif_files)} .tif files\n")

# === READ AND COMBINE METADATA ===
all_rows = []

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

    df = df[df["tif_name"].isin(tif_files)]
    all_rows.append(df)

# === COMBINE AND SUMMARIZE ===
df_all = pd.concat(all_rows, ignore_index=True)

df_all = df_all[["tif_name", "product_filesize", "map_scale"]]
df_all["product_filesize"] = pd.to_numeric(df_all["product_filesize"], errors="coerce")
df_all["map_scale"] = pd.to_numeric(df_all["map_scale"], errors="coerce")

df_all = df_all.dropna(subset=["map_scale"])
df_all["map_scale"] = df_all["map_scale"].astype(int)

# === GROUP BY SCALE ===
summary = df_all.groupby("map_scale").agg(
    num_maps=("tif_name", "count"),
    total_bytes=("product_filesize", "sum"),
    avg_bytes=("product_filesize", "mean")
).reset_index()

summary["total_MB"] = summary["total_bytes"] / 1024**2
summary["total_GB"] = summary["total_bytes"] / 1024**3
summary["avg_MB"] = summary["avg_bytes"] / 1024**2

# === SAVE CSV SUMMARY ===
summary_path = os.path.join(metadata_dir, "scale_distribution_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"💾 Summary saved to: {summary_path}")

# === PLOT WITH ANNOTATIONS ===
summary = summary.sort_values("map_scale")
x = summary["map_scale"].astype(str)
y = summary["num_maps"]
avg_sizes = summary["avg_MB"]
total_sizes = summary["total_MB"]

plt.figure(figsize=(14, 7))
bars = plt.bar(x, y, color='skyblue')
plt.title("Map count by scale")
# plt.xlabel("Scale (1:X)")
plt.ylabel("Number of maps")
plt.xticks(rotation=45)

# Add count above bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 50, f"{int(height)}", ha='center', va='bottom', fontsize=9)

ax = plt.gca()
ax2 = ax.secondary_xaxis('bottom')
ax2.set_ticks(ax.get_xticks())

tick_labels = []
for avg, total_mb, total_gb in zip(avg_sizes, total_sizes, summary["total_GB"]):
    total_str = f"{total_gb:.1f}GB" if total_gb >= 1 else f"{total_mb:.1f}MB"
    tick_labels.append(f"Avg: {avg:.1f}MB\nTot: {total_str}")

ax2.set_xticklabels(tick_labels)
ax2.tick_params(axis='x', labelsize=9, pad=15)
ax2.spines['bottom'].set_position(('outward', 50))


plt.tight_layout()
plot_path = os.path.join(metadata_dir, "scale_distribution_plot_annotated.png")
plt.savefig(plot_path)
print(f"📊 Annotated plot saved to: {plot_path}")
