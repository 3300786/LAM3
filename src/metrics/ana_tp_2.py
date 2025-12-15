#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze cross-model synergy consistency for samples and plot distribution.

Input CSV format:
- rows: sample ids
- columns: model names
- one column is sample id (default "id")
- each cell: "(x, x, x)" where x in {0,1}, representing (A, B, AB)

This script:
1) parses the triplets,
2) counts strong positive & strong adversarial synergy per sample,
3) classifies sample-level synergy type,
4) prints distribution and saves a detailed CSV,
5) plots a distribution with categories:
   - pure negative: -3, -2, -1 (neg_count=3,2,1; pos_count=0)
   - unknown: 0 (pos_count=0, neg_count=0)
   - pure positive: 1, 2, 3 (pos_count=1,2,3; neg_count=0)
   - unstable: pos_count == neg_count > 0 (aggregated into one bar)
"""

import argparse
import os
import re
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------

TRIPLET_PATTERN = re.compile(r"[-]?\d+")


def parse_triplet(cell: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse a string like "(0, 0, 1)" into (0, 0, 1).
    Returns None if the cell is empty or cannot be parsed.
    """
    if cell is None:
        return None
    if isinstance(cell, float) and pd.isna(cell):
        return None

    s = str(cell).strip()
    if not s:
        return None

    nums = TRIPLET_PATTERN.findall(s)
    if len(nums) != 3:
        # malformed cell; you can choose to raise instead
        return None

    try:
        a, b, ab = map(int, nums)
        return a, b, ab
    except ValueError:
        return None


def is_strong_positive(triplet: Tuple[int, int, int]) -> bool:
    """Strong positive synergy: (0, 0, 1)."""
    a, b, ab = triplet
    return a == 0 and b == 0 and ab == 1


def is_strong_adversarial(triplet: Tuple[int, int, int]) -> bool:
    """Strong adversarial synergy: (1, 1, 0)."""
    a, b, ab = triplet
    return a == 1 and b == 1 and ab == 0


# ---------------------------------------------------------
# Classification logic
# ---------------------------------------------------------

def classify_sample(pos_count: int, neg_count: int) -> str:
    """
    Classify sample based on counts of strong positive / strong adversarial synergy
    across models.

    Rules:
    - unknown: pos=0 and neg=0
    - unstable: pos=neg>0
    - positive_only: pos>0, neg=0
    - negative_only: neg>0, pos=0
    - pos_dominant_mixed: pos>neg>0
    - neg_dominant_mixed: neg>pos>0
    """
    if pos_count == 0 and neg_count == 0:
        return "unknown"
    if pos_count == neg_count and pos_count > 0:
        return "unstable"
    if pos_count > 0 and neg_count == 0:
        return "positive_only"
    if neg_count > 0 and pos_count == 0:
        return "negative_only"
    if pos_count > neg_count and neg_count > 0:
        return "pos_dominant_mixed"
    if neg_count > pos_count and pos_count > 0:
        return "neg_dominant_mixed"
    # fallback (should not happen with current rules)
    return "other"


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_distribution(
    df: pd.DataFrame,
    fig_path: str,
    pos_col: str = "strong_pos_count",
    neg_col: str = "strong_neg_count",
) -> None:
    """
    Plot distribution of:
    1. pure negative: -3, -2, -1 (neg_count=3,2,1; pos_count=0)
    2. unknown: 0 (pos=0, neg=0)
    3. pure positive: 1, 2, 3 (pos_count=1,2,3; neg_count=0)
    4. unstable: pos_count == neg_count > 0 (aggregated one bar)

    All bars are shown in a single figure.
    """

    # Bins (x-axis labels)
    bin_labels = ["-3", "-2", "-1" ,"0","1","2","3" "unstable"]
    counts = {lab: 0 for lab in bin_labels}

    for _, row in df.iterrows():
        pos = int(row[pos_col])
        neg = int(row[neg_col])

        # pure negative, capped at 3
        if pos == 0 and neg in (1, 2, 3):
            bin_label = f"-{neg}"
            counts[bin_label] += 1
        # pure positive, capped at 3
        elif neg == 0 and pos in (1, 2, 3):
            bin_label = f"{pos}"
            counts[bin_label] += 1
        # unknown
        elif pos == 0 and neg == 0:
            counts["0"] += 1
        # unstable: pos == neg > 0 (aggregated)
        elif pos == neg and pos > 0:
            counts["unstable"] += 1
        else:
            # samples not falling into the specified categories
            # are ignored for this particular plot
            continue

    # Prepare bar positions and heights
    x_idx = list(range(len(bin_labels)))
    heights = [counts[lab] for lab in bin_labels]

    plt.figure(figsize=(8, 4))
    plt.bar(x_idx, heights)
    plt.xticks(
        x_idx,
        [-3,-2,-1,0, 1, 2, 3, "unstable"],
    )
    plt.xlabel("Synergy pattern category")
    plt.ylabel("Number of samples")
    plt.title("Cross-model synergy consistency distribution")
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {fig_path}")


# ---------------------------------------------------------
# Core analysis
# ---------------------------------------------------------

def analyze_synergy(
    csv_path: str,
    id_column: str = "id",
    output_path: Optional[str] = None,
    fig_path: Optional[str] = None,
) -> None:
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"id column '{id_column}' not found in CSV.")

    # treat all non-id columns as model outputs
    model_cols = [c for c in df.columns if c != id_column]
    if not model_cols:
        raise ValueError("No model columns found (only id column present).")

    # prepare result columns
    pos_counts = []
    neg_counts = []
    synergy_types = []

    # optional per-model flags (for debugging / visualization)
    per_model_pos = {col: [] for col in model_cols}
    per_model_neg = {col: [] for col in model_cols}

    for _, row in df.iterrows():
        pos = 0
        neg = 0

        for col in model_cols:
            triplet = parse_triplet(row[col])
            if triplet is None:
                per_model_pos[col].append(0)
                per_model_neg[col].append(0)
                continue

            is_pos = int(is_strong_positive(triplet))
            is_neg = int(is_strong_adversarial(triplet))

            pos += is_pos
            neg += is_neg

            per_model_pos[col].append(is_pos)
            per_model_neg[col].append(is_neg)

        pos_counts.append(pos)
        neg_counts.append(neg)
        synergy_types.append(classify_sample(pos, neg))

    # assemble result DataFrame
    result = df.copy()
    result["strong_pos_count"] = pos_counts
    result["strong_neg_count"] = neg_counts
    result["synergy_type"] = synergy_types

    # per-model indicator columns (optional)
    for col in model_cols:
        result[f"{col}_strong_pos"] = per_model_pos[col]
        result[f"{col}_strong_neg"] = per_model_neg[col]

    # print distribution summaries
    print("=== Synergy type distribution ===")
    print(result["synergy_type"].value_counts().sort_index())
    print()

    print("=== Strong positive count distribution ===")
    print(result["strong_pos_count"].value_counts().sort_index())
    print()

    print("=== Strong adversarial count distribution ===")
    print(result["strong_neg_count"].value_counts().sort_index())
    print()

    # save detailed CSV
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = base + "_synergy_analysis.csv"
    result.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")

    # plot distribution if requested
    if fig_path is not None:
        plot_distribution(result, fig_path)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-model synergy consistency for samples and plot distribution."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV path (rows=id, cols=models, cells='(x, x, x)').",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="id",
        help="Name of the id column (default: 'id').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: <input>_synergy_analysis.csv).",
    )
    parser.add_argument(
        "--fig",
        type=str,
        default=None,
        help="Output figure path (e.g., outputs/synergy_dist.png). "
             "If not provided, no figure will be generated.",
    )

    args = parser.parse_args()
    analyze_synergy(
        args.csv,
        id_column=args.id_col,
        output_path=args.out,
        fig_path=args.fig,
    )


if __name__ == "__main__":
    main()
