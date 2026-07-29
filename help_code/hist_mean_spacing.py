"""
script visualizes node spacing statistics from spacing_stats_*.csv.
it creates:
- histogram of mean spacing across patches
- boxplots of mean / median / p10 / p90
- scatter plot of mean vs median spacing

usage example:
python vis_node_spacing.py spacing_stats_train.csv spacing_plots_train
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def plot_hist_mean(df, out_dir):
    plt.figure()
    plt.hist(df["mean"], bins=20)
    plt.xlabel("mean node spacing (pixels)")
    plt.ylabel("number of patches")
    plt.title("distribution of mean node spacing")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "hist_mean_spacing.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")

def plot_box_summary(df, out_dir):
    plt.figure()
    data = [df["mean"], df["median"], df["p10"], df["p90"]]
    labels = ["mean", "median", "p10", "p90"]
    plt.boxplot(data, labels=labels)
    plt.ylabel("spacing (pixels)")
    plt.title("node spacing summary across patches")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "box_spacing_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")

def plot_scatter_mean_median(df, out_dir):
    plt.figure()
    plt.scatter(df["mean"], df["median"], s=10)
    plt.xlabel("mean spacing (pixels)")
    plt.ylabel("median spacing (pixels)")
    plt.title("mean vs median spacing per patch")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "scatter_mean_vs_median.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")

def main():
    if len(sys.argv) != 3:
        print("usage: python vis_node_spacing.py <spacing_stats.csv> <out_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2]

    create_dir(out_dir)

    df = pd.read_csv(csv_path)

    # basic sanity check
    required_cols = {"mean", "median", "p10", "p90"}
    if not required_cols.issubset(df.columns):
        print("error: csv file does not contain required columns:", required_cols)
        sys.exit(1)

    plot_hist_mean(df, out_dir)
    plot_box_summary(df, out_dir)
    plot_scatter_mean_median(df, out_dir)

    # optional: print global summary for console
    print(f"global mean spacing: {df['mean'].mean():.3f}")
    print(f"global median of medians: {df['median'].median():.3f}")

if __name__ == "__main__":
    main()
