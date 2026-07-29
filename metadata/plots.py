import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/users/4/kirsa002/IMOLA/metadata/la_public_clean - la_public_clean.csv'
out_dir = '/users/4/kirsa002/IMOLA/help_code/plots'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

print("Columns:")
for c in df.columns:
    print("-", c)

# Detect columns
year_col = next((c for c in df.columns if 'year' in c.lower()), None)
scale_col = next((c for c in df.columns if 'scale' in c.lower()), None)
processed_col = next(
    (c for c in df.columns if any(x in c.lower() for x in ['processed', 'output', 'prediction', 'link'])),
    None
)

print("\nDetected:")
print("year_col =", year_col)
print("scale_col =", scale_col)
print("processed_col =", processed_col)

plots_created = []

# 1. Year distribution
if year_col:
    year_vals = pd.to_numeric(df[year_col], errors='coerce').dropna()
    if not year_vals.empty:
        plt.figure(figsize=(10, 5))
        year_vals.astype(int).hist(bins=30)
        plt.title("Distribution of Maps by Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        out = os.path.join(out_dir, "year_distribution.png")
        plt.savefig(out, bbox_inches='tight', dpi=200)
        plt.close()
        plots_created.append(out)

# 2. Scale distribution
if scale_col:
    scale_vals = df[scale_col].dropna().astype(str)
    if not scale_vals.empty:
        plt.figure(figsize=(12, 5))
        scale_vals.value_counts().head(20).plot(kind='bar')
        plt.title("Top 20 Map Scales")
        plt.xlabel("Scale")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        out = os.path.join(out_dir, "scale_distribution.png")
        plt.savefig(out, bbox_inches='tight', dpi=200)
        plt.close()
        plots_created.append(out)

# 3. Processed vs not processed
if processed_col:
    plt.figure(figsize=(6, 4))
    processed_flag = df[processed_col].notna()
    processed_flag.value_counts().plot(kind='bar')
    plt.title("Processed vs Not Processed")
    plt.xlabel("Has processed output")
    plt.ylabel("Count")
    out = os.path.join(out_dir, "processed_distribution.png")
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    plots_created.append(out)

# 4. Year vs processed
if year_col and processed_col:
    tmp = df.copy()
    tmp['processed_flag'] = tmp[processed_col].notna()
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors='coerce')
    tmp = tmp.dropna(subset=[year_col])

    if not tmp.empty:
        grouped = tmp.groupby([year_col, 'processed_flag']).size().unstack(fill_value=0)
        plt.figure(figsize=(12, 5))
        grouped.plot(kind='line')
        plt.title("Processed vs Not Processed Over Years")
        plt.xlabel("Year")
        plt.ylabel("Count")
        out = os.path.join(out_dir, "processed_over_time.png")
        plt.savefig(out, bbox_inches='tight', dpi=200)
        plt.close()
        plots_created.append(out)

# 5. Scale vs processed
if scale_col and processed_col:
    tmp = df.copy()
    tmp['processed_flag'] = tmp[processed_col].notna()
    grouped = tmp.groupby([scale_col, 'processed_flag']).size().unstack(fill_value=0)

    if not grouped.empty:
        plt.figure(figsize=(12, 6))
        grouped.head(20).plot(kind='bar')
        plt.title("Processed vs Not Processed by Scale (Top 20)")
        plt.xlabel("Scale")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        out = os.path.join(out_dir, "scale_vs_processed.png")
        plt.savefig(out, bbox_inches='tight', dpi=200)
        plt.close()
        plots_created.append(out)

print("\nCreated plots:")
for p in plots_created:
    print(p)