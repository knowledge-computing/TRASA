"""
script builds a mock error analysis pipeline for sam-road predictions.
it generates fake but reasonable metrics, plots, and tables so you can
see what everything will look like before plugging in real data.

usage example:
python error_analysis_mock.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_dir = "error_analysis_mock_out"


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def generate_mock_dataset(n_patches=80, seed=42):
    """
    generate a fake dataset of per-patch metrics that looks realistic.
    later you can replace this with real metrics from gt vs predictions.
    """
    rng = np.random.default_rng(seed)

    patch_ids = np.arange(1, n_patches + 1)

    # line metrics: assume decent but not perfect model
    line_precision = np.clip(rng.normal(loc=0.82, scale=0.07, size=n_patches), 0.3, 1.0)
    line_recall = np.clip(rng.normal(loc=0.78, scale=0.08, size=n_patches), 0.3, 1.0)
    line_iou = (line_precision * line_recall) / (line_precision + line_recall - line_precision * line_recall + 1e-8)

    # node metrics
    node_precision = np.clip(rng.normal(loc=0.80, scale=0.08, size=n_patches), 0.3, 1.0)
    node_recall = np.clip(rng.normal(loc=0.75, scale=0.10, size=n_patches), 0.2, 1.0)

    # recall for special node types
    branch_node_recall = np.clip(node_recall + rng.normal(loc=0.05, scale=0.05, size=n_patches), 0.2, 1.0)
    end_node_recall = np.clip(node_recall + rng.normal(loc=-0.03, scale=0.05, size=n_patches), 0.2, 1.0)

    # sizes of graphs
    num_gt_nodes = rng.integers(low=15, high=80, size=n_patches)
    num_pred_nodes = num_gt_nodes + rng.integers(low=-5, high=10, size=n_patches)

    num_gt_edges = rng.integers(low=20, high=100, size=n_patches)
    num_pred_edges = num_gt_edges + rng.integers(low=-10, high=15, size=n_patches)

    # structural errors
    # make them correlate a bit with line_iou (worse iou -> more errors)
    base_err_scale = 1.5 * (1.0 - line_iou) * 20.0

    num_break_errors = rng.poisson(lam=np.clip(base_err_scale, 0.0, 10.0))
    num_spurious_junctions = rng.poisson(lam=np.clip(base_err_scale * 0.4, 0.0, 6.0))
    num_missed_junctions = rng.poisson(lam=np.clip(base_err_scale * 0.6, 0.0, 8.0))
    num_dangling_segments = rng.poisson(lam=np.clip(base_err_scale * 0.3, 0.0, 5.0))

    # difficulty labels
    difficulty = rng.choice(["easy", "medium", "hard"], size=n_patches, p=[0.3, 0.5, 0.2])

    df = pd.DataFrame(
        {
            "patch_id": patch_ids,
            "difficulty": difficulty,
            "line_precision": line_precision,
            "line_recall": line_recall,
            "line_iou": line_iou,
            "node_precision": node_precision,
            "node_recall": node_recall,
            "branch_node_recall": branch_node_recall,
            "end_node_recall": end_node_recall,
            "num_gt_nodes": num_gt_nodes,
            "num_pred_nodes": num_pred_nodes,
            "num_gt_edges": num_gt_edges,
            "num_pred_edges": num_pred_edges,
            "num_break_errors": num_break_errors,
            "num_spurious_junctions": num_spurious_junctions,
            "num_missed_junctions": num_missed_junctions,
            "num_dangling_segments": num_dangling_segments,
        }
    )

    return df


def summarize_global_metrics(df):
    """
    print and save a small table summarizing global metrics.
    """
    metrics = [
        "line_precision",
        "line_recall",
        "line_iou",
        "node_precision",
        "node_recall",
        "branch_node_recall",
        "end_node_recall",
    ]

    summary = df[metrics].agg(["mean", "std", "min", "max"])
    print("global metric summary:")
    print(summary.round(3))

    summary.to_csv(os.path.join(out_dir, "global_metric_summary.csv"))


def summarize_structural_errors(df):
    """
    build a small table of averages for structural error types.
    """
    cols = [
        "num_break_errors",
        "num_spurious_junctions",
        "num_missed_junctions",
        "num_dangling_segments",
    ]

    summary = df[cols].agg(["mean", "median", "max"]).round(2)
    print("\nstructural error summary:")
    print(summary)

    summary.to_csv(os.path.join(out_dir, "structural_error_summary.csv"))


def group_by_difficulty(df):
    """
    show how metrics vary by patch difficulty.
    """
    metrics = [
        "line_iou",
        "node_recall",
        "branch_node_recall",
        "end_node_recall",
    ]

    grouped = df.groupby("difficulty")[metrics].agg(["mean", "std"]).round(3)
    print("\nmetrics grouped by difficulty:")
    print(grouped)

    grouped.to_csv(os.path.join(out_dir, "metrics_by_difficulty.csv"))


def plot_hist_line_iou(df):
    plt.figure()
    plt.hist(df["line_iou"], bins=15)
    plt.xlabel("line iou")
    plt.ylabel("number of patches")
    plt.title("distribution of line iou across patches")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_line_iou.png"), dpi=200)
    plt.close()


def plot_hist_node_recall(df):
    plt.figure()
    plt.hist(df["node_recall"], bins=15)
    plt.xlabel("node recall")
    plt.ylabel("number of patches")
    plt.title("distribution of node recall across patches")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_node_recall.png"), dpi=200)
    plt.close()


def plot_box_errors(df):
    cols = [
        "num_break_errors",
        "num_spurious_junctions",
        "num_missed_junctions",
        "num_dangling_segments",
    ]
    plt.figure()
    plt.boxplot([df[c] for c in cols], labels=cols, showfliers=True)
    plt.ylabel("count per patch")
    plt.title("structural error distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_structural_errors.png"), dpi=200)
    plt.close()


def plot_line_iou_vs_breaks(df):
    plt.figure()
    plt.scatter(df["line_iou"], df["num_break_errors"], s=20)
    plt.xlabel("line iou")
    plt.ylabel("number of break errors")
    plt.title("relationship between line iou and breaks")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_line_iou_vs_breaks.png"), dpi=200)
    plt.close()


def plot_difficulty_boxplots(df):
    """
    boxplots of key metrics grouped by difficulty level.
    """
    metrics = ["line_iou", "node_recall"]
    difficulties = ["easy", "medium", "hard"]

    for metric in metrics:
        plt.figure()
        data = [df[df["difficulty"] == d][metric].values for d in difficulties]
        plt.boxplot(data, labels=difficulties)
        plt.ylabel(metric)
        plt.title(f"{metric} by difficulty")
        plt.tight_layout()
        fname = f"box_{metric}_by_difficulty.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def save_per_patch_table(df):
    """
    write a per-patch csv with all metrics, suitable for filtering and sorting.
    """
    df_sorted = df.sort_values("line_iou")
    df_sorted.to_csv(os.path.join(out_dir, "per_patch_metrics.csv"), index=False)


def main():
    make_dir(out_dir)

    df = generate_mock_dataset(n_patches=80, seed=42)

    # summaries
    summarize_global_metrics(df)
    summarize_structural_errors(df)
    group_by_difficulty(df)
    save_per_patch_table(df)

    # plots
    plot_hist_line_iou(df)
    plot_hist_node_recall(df)
    plot_box_errors(df)
    plot_line_iou_vs_breaks(df)
    plot_difficulty_boxplots(df)

    print("\nmock error analysis artifacts written to:", out_dir)
    print("you can now open png and csv files to see how tables and plots will look.")


if __name__ == "__main__":
    main()
