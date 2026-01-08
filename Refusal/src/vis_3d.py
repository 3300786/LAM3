import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

# ==========================================
# Color Mapping
# ==========================================
BEHAVIOR_COLORS = {
    # --- Jailbreak (Red/Orange) ---
    "J0_ACTIONABLE_COMPLIANCE": "#d62728",  # Deep Red (Actionable)
    "J1_PARTIAL_COMPLIANCE": "#ff7f0e",  # Orange (Partial/Leaked)
    "J2_THEORETICAL_COMPLIANCE": "#ffbb78",  # Light Orange (Theoretical)

    # --- Policy Violation (Yellow) ---
    "U0_POLICY_VIOLATION": "#bcbd22",  # Olive (Low risk unsafe)

    # --- Safe / Benign (Green) ---
    "S0_BENIGN_COMPLIANCE": "#2ca02c",  # Green (Safe)
    "R0_REFUSAL": "#1f77b4",  # Blue (Refusal)
    "R1_SAFE_REDIRECT": "#17becf",  # Cyan (Redirect)

    # --- Failures / Other (Gray/Purple) ---
    "M0_MISINTERPRETATION": "#7f7f7f",  # Gray (Misinterpretation)
    "E1_DECEPTIVE_COMPLIANCE": "#9467bd",  # Purple (Empty 'Sure')
    "E0_OTHER_FAILURE": "#8c564b"  # Brown
}


def parse_tag(tag):
    """提取标签中的数字，例如 'I2' -> 2"""
    try:
        return int(tag[1])
    except:
        return -1


def load_and_group_data(judged_file):
    print(f"[Vis3D] Loading {judged_file}...")
    grouped_points = defaultdict(list)
    all_points = []

    with open(judged_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            res = rec.get('judge_result', {})
            group = rec.get('group', 'unknown')

            # 获取三轴标签
            i_tag = res.get('axis_I', 'I0')
            a_tag = res.get('axis_II', 'A0')
            h_tag = res.get('axis_III', 'H0')
            beh = res.get('behavior_type', 'E0_OTHER_FAILURE')

            # 解析坐标
            x = parse_tag(i_tag)  # I: Severity
            y = parse_tag(a_tag)  # A: Alignment
            z = parse_tag(h_tag)  # H: Harmfulness

            if x >= 0 and y >= 0 and z >= 0:
                point = {
                    "x": x, "y": y, "z": z, "beh": beh, "group": group
                }
                grouped_points[group].append(point)
                all_points.append(point)

    return all_points, grouped_points


def plot_single_group(data_points, group_name, output_dir):
    if not data_points: return

    # Jittering Logic
    jitter_scale = 0.25
    xs, ys, zs, colors = [], [], [], []

    for p in data_points:
        noise = np.random.uniform(-jitter_scale, jitter_scale, 3)
        xs.append(p['x'] + noise[0])
        ys.append(p['y'] + noise[1])
        zs.append(p['z'] + noise[2])
        colors.append(BEHAVIOR_COLORS.get(p['beh'], '#333333'))

    # 安全的文件名
    safe_name = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in group_name])
    group_dir = os.path.join(output_dir, "3d_plots")
    os.makedirs(group_dir, exist_ok=True)

    # --- 3D Plot ---
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c=colors, marker='o', s=20, alpha=0.6, edgecolors='w', linewidth=0.1)

    # Labels
    ax.set_xlabel('Axis-I: Severity')
    ax.set_ylabel('Axis-II: Alignment')
    ax.set_zlabel('Axis-III: Harmfulness')

    # Ticks
    ax.set_xticks(range(4))
    ax.set_xticklabels(['I0', 'I1', 'I2', 'I3'])
    ax.set_yticks(range(5))
    ax.set_yticklabels(['A0', 'A1', 'A2', 'A3', 'A4'])
    ax.set_zticks(range(4))
    ax.set_zticklabels(['H0', 'H1', 'H2', 'H3'])

    # Pane styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Legend
    from matplotlib.lines import Line2D
    # Only show behaviors present in this group
    present_behs = set(p['beh'] for p in data_points)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                              markerfacecolor=v, markersize=8)
                       for k, v in BEHAVIOR_COLORS.items() if k in present_behs]

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize='x-small',
                  title="Behavior")

    plt.title(f"3D Distribution: {group_name} (n={len(data_points)})", fontsize=12)

    # Save Default View
    save_path = os.path.join(group_dir, f"3d_view_{safe_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # --- 2D Heatmap (Alignment vs Harmfulness) ---
    plot_2d_heatmap(data_points, group_name, group_dir)


def plot_2d_heatmap(data_points, group_name, output_dir):
    grid = np.zeros((4, 5))  # rows=H, cols=A
    for p in data_points:
        h, a = p['z'], p['y']
        if 0 <= h < 4 and 0 <= a < 5:
            grid[h, a] += 1

    total = len(data_points)

    plt.figure(figsize=(6, 5), dpi=100)
    plt.imshow(grid, cmap='Reds', origin='lower', vmin=0)

    # Annotate
    for h in range(4):
        for a in range(5):
            count = int(grid[h, a])
            if count > 0:
                # Dynamic text color based on background intensity
                max_val = np.max(grid)
                color = 'white' if count > max_val * 0.5 else 'black'
                plt.text(a, h, str(count), ha='center', va='center', color=color, fontweight='bold')

    plt.xticks(range(5), ['A0\nMisint', 'A1\nRefuse', 'A2\nRedir', 'A3\nPart', 'A4\nFull'])
    plt.yticks(range(4), ['H0\nNone', 'H1\nAbstr', 'H2\nIncpl', 'H3\nAction'])
    plt.xlabel("Axis-II: Alignment")
    plt.ylabel("Axis-III: Harmfulness")
    plt.title(f"Align vs Harm: {group_name}")
    plt.tight_layout()

    safe_name = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in group_name])
    plt.savefig(os.path.join(output_dir, f"heatmap_{safe_name}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to final_judged.jsonl")
    parser.add_argument("--out_dir", type=str, default="Refusal/outputs/3d")
    args = parser.parse_args()

    all_points, grouped_points = load_and_group_data(args.input)

    if not all_points:
        print("No valid data found.")
        return

    print(f"[Vis3D] Found {len(grouped_points)} groups.")

    # 1. Plot Overall
    print("Plotting Overall...")
    plot_single_group(all_points, "Overall", args.out_dir)

    # 2. Plot Per Group
    for group, points in tqdm(grouped_points.items(), desc="Plotting Groups"):
        if len(points) < 5: continue  # Skip tiny groups
        plot_single_group(points, group, args.out_dir)

    print(f"[Vis3D] All plots saved to {os.path.join(args.out_dir, '3d_plots')}")


if __name__ == "__main__":
    main()