#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Extract image type from image_path
# ---------------------------------------------------------
def extract_image_type(path: str) -> str:
    """
    Example: 'llm_transfer_attack/SD_related_566.png'
    We extract 'SD_related'.
    """
    if not isinstance(path, str) or "/" not in path:
        return "none"

    filename = path.split("/")[-1]  # SD_related_566.png
    base = filename.split(".")[0]   # SD_related_566
    parts = base.split("_")
    # print(parts)
    # if len(parts) >= 2:
    return "_".join(parts[:-1])  # SD_related
    # return base


# ---------------------------------------------------------
# Normalize: convert counts to percentage per group
# ---------------------------------------------------------
def normalize_group(df, group_col, sy_col="synergy_type"):
    """
    Convert counts to percentages within each group.
    Input df must contain columns group_col, sy_col.

    Output is a pivoted DataFrame:
        index = synergy_type
        columns = group_col
        values = normalized percentages
    """
    # Count raw numbers
    c = df.groupby([group_col, sy_col])["id"].count().reset_index()
    c.rename(columns={"id": "count"}, inplace=True)

    # Total samples for each group
    total = c.groupby(group_col)["count"].transform("sum")
    c["ratio"] = c["count"] / total

    # Pivot into heatmap/table-friendly structure
    pivot = c.pivot(index=sy_col, columns=group_col, values="ratio").fillna(0)

    return c, pivot


# ---------------------------------------------------------
# Plotting (normalized)
# ---------------------------------------------------------
def plot_heatmap(pivot, title, out_path, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_normalized_bar(c, group_col, title, out_path):
    """
    c: long-form count+ratio table from normalize_group()
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(data=c, x="synergy_type", y="ratio", hue=group_col)
    plt.title(title)
    plt.ylabel("ratio within group")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze relation between synergy_type and input sample metadata (normalized version)")
    parser.add_argument("--synergy_csv", required=True, help="CSV from synergy analysis (contains id, synergy_type, ...)")
    parser.add_argument("--meta_csv", required=True, help="CSV containing id, format, policy, image_path, from")
    parser.add_argument("--out_dir", required=True, help="Output directory for figures")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    syn = pd.read_csv(args.synergy_csv)
    meta = pd.read_csv(args.meta_csv)

    # Merge
    merged = pd.merge(syn, meta, on="id", how="inner")

    # Process image_type
    merged["image_type"] = merged["image_path"].apply(extract_image_type)

    # ---------------------------------------------------------
    # Format
    # ---------------------------------------------------------
    c_fmt, pivot_fmt = normalize_group(merged, "format")
    plot_heatmap(
        pivot_fmt,
        "Normalized ratio: Format vs Synergy Type",
        os.path.join(args.out_dir, "format_vs_synergy_heatmap.png")
    )
    plot_normalized_bar(
        c_fmt,
        "format",
        "Normalized bars: Format vs Synergy Type",
        os.path.join(args.out_dir, "format_vs_synergy_bar.png")
    )

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------
    c_pol, pivot_pol = normalize_group(merged, "policy")
    plot_heatmap(
        pivot_pol,
        "Normalized ratio: Policy vs Synergy Type",
        os.path.join(args.out_dir, "policy_vs_synergy_heatmap.png")
    )
    plot_normalized_bar(
        c_pol,
        "policy",
        "Normalized bars: Policy vs Synergy Type",
        os.path.join(args.out_dir, "policy_vs_synergy_bar.png")
    )

    # ---------------------------------------------------------
    # Image Type
    # ---------------------------------------------------------
    c_img, pivot_img = normalize_group(merged, "image_type")
    plot_heatmap(
        pivot_img,
        "Normalized ratio: Image Type vs Synergy Type",
        os.path.join(args.out_dir, "image_type_vs_synergy_heatmap.png")
    )
    plot_normalized_bar(
        c_img,
        "image_type",
        "Normalized bars: Image Type vs Synergy Type",
        os.path.join(args.out_dir, "image_type_vs_synergy_bar.png")
    )

    # ---------------------------------------------------------
    # Dataset source ("from")
    # ---------------------------------------------------------
    c_from, pivot_from = normalize_group(merged, "from")
    plot_heatmap(
        pivot_from,
        "Normalized ratio: From vs Synergy Type",
        os.path.join(args.out_dir, "from_vs_synergy_heatmap.png")
    )
    plot_normalized_bar(
        c_from,
        "from",
        "Normalized bars: From vs Synergy Type",
        os.path.join(args.out_dir, "from_vs_synergy_bar.png")
    )

    merged.to_csv(os.path.join(args.out_dir, "merged_synergy_meta.csv"), index=False)
    print("All analysis completed.")
    print("Merged data saved.")


if __name__ == "__main__":
    main()