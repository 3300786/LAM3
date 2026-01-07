import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import os
import argparse
from tqdm import tqdm

# 复用之前的颜色定义
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
    "E0_OTHER_FAILURE": "#c49c94",  # Brown (Other)
}


def parse_tag(tag):
    """提取标签中的数字，例如 'I2' -> 2"""
    try:
        return int(tag[1])
    except:
        return -1


def plot_3d_density(judged_file, output_dir):
    print(f"[Vis3D] Loading {judged_file}...")

    # 1. Load Data
    data_points = []

    with open(judged_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            res = rec.get('judge_result', {})

            # 获取三轴标签
            i_tag = res.get('axis_I', 'I0')
            a_tag = res.get('axis_II', 'A0')
            h_tag = res.get('axis_III', 'H0')
            beh = res.get('behavior_type', 'E0_OTHER_FAILURE')
            # if beh.startswith("M0"):
            # if a_tag == 'A3' and h_tag == 'H0':
            #     print(beh, i_tag, a_tag, h_tag)
            # 解析坐标
            x = parse_tag(i_tag)  # I: Severity
            y = parse_tag(a_tag)  # A: Alignment
            z = parse_tag(h_tag)  # H: Harmfulness

            if x >= 0 and y >= 0 and z >= 0:
                data_points.append({
                    "x": x, "y": y, "z": z, "beh": beh
                })

    if not data_points:
        print("No valid data found.")
        return

    # 2. Apply Jittering (核心步骤：防止重叠)
    # 偏移量范围 [-0.25, 0.25]，保证不同整数坐标的点云不会混在一起
    jitter_scale = 0.25

    xs, ys, zs, colors = [], [], [], []

    for p in data_points:
        # 添加随机噪声
        noise = np.random.uniform(-jitter_scale, jitter_scale, 3)
        xs.append(p['x'] + noise[0])
        ys.append(p['y'] + noise[1])
        zs.append(p['z'] + noise[2])
        colors.append(BEHAVIOR_COLORS.get(p['beh'], '#333333'))

    # 3. Plotting
    print(f"[Vis3D] Plotting {len(data_points)} points in 3D space...")
    os.makedirs(output_dir, exist_ok=True)

    # 创建高分辨率画布
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点
    # alpha=0.6 让密集的区域颜色更深
    # s=20 点的大小
    scatter = ax.scatter(xs, ys, zs, c=colors, marker='o', s=20, alpha=0.6, edgecolors='w', linewidth=0.1)

    # 4. 设置坐标轴标签和刻度
    ax.set_xlabel('Axis-I: Intent Severity')
    ax.set_ylabel('Axis-II: Intent Alignment')
    ax.set_zlabel('Axis-III: Response Harmfulness')

    # 设置离散刻度标签
    ax.set_xticks(range(4))
    ax.set_xticklabels(['I0\nBenign', 'I1\nMild', 'I2\nDangr', 'I3\nSevere'])

    ax.set_yticks(range(5))
    ax.set_yticklabels(['A0\nMisint', 'A1\nRefuse', 'A2\nRedir', 'A3\nPart', 'A4\nFull'])

    ax.set_zticks(range(4))
    ax.set_zticklabels(['H0\nNone', 'H1\nAbstr', 'H2\nIncpl', 'H3\nAction'])

    # 设置网格背景颜色，增强立体感
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # 添加 Legend (手动构建)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                              markerfacecolor=v, markersize=8) for k, v in BEHAVIOR_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize='small', title="Behavior Type")

    plt.title("3-Axis Behavioral Distribution (Jittered Density)", fontsize=14)

    # 5. 保存多个视角 (3D图静态很难看全)

    # View 1: 默认视角
    save_path = os.path.join(output_dir, "3d_distribution_view_default.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")

    # View 2: 俯视 (关注 Severity vs Alignment)
    ax.view_init(elev=90, azim=-90)
    plt.savefig(os.path.join(output_dir, "3d_distribution_view_top.png"), bbox_inches='tight')

    # View 3: 侧视 (关注 Alignment vs Harmfulness) -> 观察越狱阶梯
    ax.view_init(elev=0, azim=0)  # 看 Y-Z 平面
    plt.savefig(os.path.join(output_dir, "3d_distribution_view_side.png"), bbox_inches='tight')

    plt.close()

    # ==========================================
    # 额外：生成 2D 投影热力图 (更适合论文分析)
    # ==========================================
    plot_2d_heatmap(data_points, output_dir)


def plot_2d_heatmap(data_points, output_dir):
    """
    绘制 II (Alignment) vs III (Harmfulness) 的热力图
    这是最关键的平面，展示模型是否"对齐得越好，危害越大"
    """
    # Grid 5x4 (A0-A4, H0-H3)
    grid = np.zeros((4, 5))  # rows=H, cols=A

    for p in data_points:
        h, a = p['z'], p['y']
        if 0 <= h < 4 and 0 <= a < 5:
            grid[h, a] += 1

    # Normalize
    total = len(data_points)
    if total > 0:
        grid_pct = grid / total

    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(grid, cmap='Reds', origin='lower')

    # Annotate counts
    for h in range(4):
        for a in range(5):
            count = int(grid[h, a])
            if count > 0:
                plt.text(a, h, str(count), ha='center', va='center', color='black' if count < total * 0.5 else 'white')

    plt.xticks(range(5), ['A0', 'A1', 'A2', 'A3', 'A4'])
    plt.yticks(range(4), ['H0', 'H1', 'H2', 'H3'])
    plt.xlabel("Axis-II: Alignment")
    plt.ylabel("Axis-III: Harmfulness")
    plt.title("Alignment vs Harmfulness Heatmap")
    plt.colorbar(label='Count')

    plt.savefig(os.path.join(output_dir, "2d_heatmap_align_harm.png"))
    plt.close()
    print(f"Saved: 2d_heatmap_align_harm.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to final_judged.jsonl")
    parser.add_argument("--out_dir", type=str, default="Refusal/outputs")
    args = parser.parse_args()

    plot_3d_density(args.input, args.out_dir)