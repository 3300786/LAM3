import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info
except:
    pass

# 尝试导入 seaborn 以获得更好看的图表
try:
    import seaborn as sns

    HAS_SNS = True
    sns.set_theme(style="whitegrid", context="paper")  # 使用 paper context 字体更清晰
except ImportError:
    HAS_SNS = False


def _load_images(img_source):
    """
    加载单张或多张图片
    img_source: str (path) or list[str] (paths)
    Returns: list[PIL.Image] (filtered for None)
    """
    if not img_source:
        return []

    paths = [img_source] if isinstance(img_source, str) else img_source
    images = []
    for p in paths:
        if p and os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
    return images


def _to_pil(path):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def _pool_hidden(h):
    return h[0, -1, :].float()


@torch.no_grad()
def run_evaluation(model, processor, dataset_dict, refusal_vectors, device, max_new_tokens=512, output_path=None):
    results = []
    print("[Eval] Starting inference and scoring...")
    if output_path:
        print(f"[Eval] Streaming results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # Just create/clear the file

    f_out = open(output_path, 'a', encoding='utf-8') if output_path else None
    try:
        for group, items in dataset_dict.items():
            for item in tqdm(items, desc=f"Eval {group}"):
                # print(item)
                # os.system("pause")
                images = _load_images(item.get('img'))
                txt = item.get('txt')
                intent = item.get('intent') or item.get('clue_key') or "unknown_harmful_act"
                if images:
                    content = []
                    for img in images:
                        content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": txt})

                    messages = [{"role": "user", "content": content}]

                    text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text_in], images=image_inputs, videos=video_inputs, return_tensors="pt").to(
                        device)
                else:
                    messages = [{"role": "user", "content": [{"type": "text", "text": txt}]}]
                    text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = processor(text=[text_in], return_tensors="pt").to(device)

                out = model(**inputs, output_hidden_states=True)
                scores = []
                for l, h in enumerate(out.hidden_states[1:]):
                    h_vec = _pool_hidden(h)
                    s = F.cosine_similarity(h_vec, refusal_vectors[l].to(device), dim=0).item()
                    scores.append(s)

                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
                output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

                # 构建结果对象
                result_item = {
                    "id": item['id'],
                    "group": group,
                    "intent": intent,
                    "input_text": txt,
                    "input_image": item.get('img'),
                    "scores": scores,
                    "output": output_text
                }

                results.append(result_item)

                # === 实时写入 ===
                if f_out:
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                    f_out.flush()  # 强制刷新缓冲区，确保外部立即可见
    finally:
        if f_out:
            f_out.close()

    return results


# ---------------------------
# Analysis & Plotting Utils
# ---------------------------

def estimate_thresholds(scores_list):
    if not scores_list: return [0.0] * 32
    arr = np.array(scores_list).T
    taus = []
    for l in range(len(arr)):
        layer_data = arr[l]
        if np.isnan(layer_data).all():
            t = 0.0
        else:
            # Mean + 1*Std
            t = float(np.nanmean(layer_data) + 1.0 * np.nanstd(layer_data))
        taus.append(t)
    return taus


def compute_latency(scores, taus):
    for i, (s, t) in enumerate(zip(scores, taus)):
        if s > t: return i
    return -1


# 定义8种行为的颜色映射
BEHAVIOR_COLORS = {
    # 越狱/高风险 (红/橙系)
    "J0_ACTIONABLE_COMPLIANCE": "#d62728",  # Deep Red
    "J1_PARTIAL_COMPLIANCE": "#ff7f0e",  # Orange

    # 拒绝/安全 (绿/蓝系)
    "R0_REFUSAL": "#2ca02c",  # Green
    "R1_SAFE_REDIRECT": "#1f77b4",  # Blue
    "R2_PROP_SIMULATION": "#17becf",  # Cyan

    # 其他/失效 (灰/紫系)
    "M0_MISINTERPRETATION": "#7f7f7f",  # Gray
    "C0_UNSAFE_DEPICTION": "#9467bd",  # Purple
    "E0_OTHER_FAILURE": "#8c564b"  # Brown
}


def plot_behavioral_curves(judged_file_path, output_dir):
    """
    读取经过 Judge 判定的 JSONL 文件，按 Group 分类，
    绘制独立的曲线图，并生成一张包含所有 Group 的九宫格对比图。
    """
    if not os.path.exists(judged_file_path):
        print(f"[Plot] File not found: {judged_file_path}")
        return

    print(f"[Plot] Generating behavioral curves from {judged_file_path}...")
    os.makedirs(output_dir, exist_ok=True)
    behavior_dir = os.path.join(output_dir, "behavior_curves")
    os.makedirs(behavior_dir, exist_ok=True)

    # 1. Load Data
    records = []
    with open(judged_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))

    # 2. Extract Baselines & Calculate Thresholds
    baseline_hard_scores = [r['scores'] for r in records if r['group'] == 'baseline_hard']
    baseline_easy_scores = [r['scores'] for r in records if r['group'] == 'baseline_easy']

    taus_hard = estimate_thresholds(baseline_hard_scores) if baseline_hard_scores else [0.5] * 32
    taus_easy = estimate_thresholds(baseline_easy_scores) if baseline_easy_scores else [0.0] * 32
    num_layers = len(taus_hard)

    # 3. Organize Data: data[group][behavior] = [scores_list]
    grouped_data = {}
    for r in records:
        grp = r.get('group', 'unknown')
        beh = r.get('behavior_type')
        if not beh and 'judge_result' in r:
            beh = r['judge_result'].get('behavior_type')

        if not beh: continue

        if grp not in grouped_data: grouped_data[grp] = {}
        if beh not in grouped_data[grp]: grouped_data[grp][beh] = []

        grouped_data[grp][beh].append(r['scores'])

    # 4. Plot Individual Plots (Per Group)
    for grp, behaviors in grouped_data.items():
        if "baseline" in grp: continue

        # --- Figure 1: Relative Scores (Single Group) ---
        plt.figure(figsize=(10, 6), dpi=150)
        plt.axhline(y=0, color='k', linestyle=':', label='Easy Baseline (Zero)')
        rel_hard = np.array(taus_hard) - np.array(taus_easy)
        plt.plot(range(num_layers), rel_hard, 'k--', alpha=0.5, label='Hard Boundary')
        plt.fill_between(range(num_layers), 0, rel_hard, color='gray', alpha=0.1, label='Stealth Zone')

        for beh_type, score_lists in behaviors.items():
            if not score_lists: continue
            mean_curve = np.nanmean(np.array(score_lists), axis=0)
            rel_curve = mean_curve - np.array(taus_easy)
            count = len(score_lists)
            color = BEHAVIOR_COLORS.get(beh_type, "#333333")

            plt.plot(range(num_layers), rel_curve, marker='o', markersize=3,
                     label=f"{beh_type} (n={count})", color=color, linewidth=2)

        plt.title(f"[{grp}] Relative Refusal Scores by Behavior")
        plt.xlabel("Layer")
        plt.ylabel("Relative Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(behavior_dir, f"{grp}_relative_behavior.png"))
        plt.close()

    # 5. Plot Grid View (High Resolution Comparison)
    print("[Plot] Generating Grid View...")
    plot_groups = sorted([g for g in grouped_data.keys() if "baseline" not in g])
    n_groups = len(plot_groups)

    if n_groups > 0:
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols

        # 增加 DPI 和 figsize 提升清晰度
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), dpi=300)
        axes = axes.flatten()

        # 预计算所有数据的 min/max 以统一 Y 轴 (可选，这里先自适应但保持 0 线)

        for i, grp in enumerate(plot_groups):
            ax = axes[i]
            behaviors = grouped_data[grp]

            # Draw Baselines
            ax.axhline(y=0, color='k', linestyle=':', linewidth=1)
            rel_hard = np.array(taus_hard) - np.array(taus_easy)
            ax.plot(range(num_layers), rel_hard, 'k--', alpha=0.3, linewidth=1)
            ax.fill_between(range(num_layers), 0, rel_hard, color='gray', alpha=0.05)

            # Draw Behaviors
            for beh_type, score_lists in behaviors.items():
                if not score_lists: continue
                mean_curve = np.nanmean(np.array(score_lists), axis=0)
                rel_curve = mean_curve - np.array(taus_easy)
                color = BEHAVIOR_COLORS.get(beh_type, "#333333")

                ax.plot(range(num_layers), rel_curve, marker='o', markersize=2,
                        color=color, linewidth=1.5, label=beh_type, alpha=0.8)

            ax.set_title(f"{grp}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Relative Score", fontsize=10)
            ax.grid(True, alpha=0.3)

            # 移除子图内部 Legend 防止遮挡
            # ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Global Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, lw=2) for c in BEHAVIOR_COLORS.values()]
        labels = list(BEHAVIOR_COLORS.keys())

        # 底部居中图例
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=4, fontsize=11, frameon=False, borderaxespad=0.5)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 留出底部给 Legend

        save_path = os.path.join(behavior_dir, "grid_relative_behaviors.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Plot] Grid layout saved to {save_path}")

    print(f"[Plot] Behavioral curves saved to {behavior_dir}")


def analyze_and_plot(results, output_dir):
    """
    原有的粗粒度绘图函数（保留作为 Overview）
    """
    print("[Eval] Analyzing results and generating overview plots...")
    os.makedirs(output_dir, exist_ok=True)

    groups = {}
    for r in results:
        g = r['group']
        if g not in groups: groups[g] = []
        groups[g].append(r)

    taus_hard = estimate_thresholds([r['scores'] for r in groups.get('baseline_hard', [])])
    taus_easy = estimate_thresholds([r['scores'] for r in groups.get('baseline_easy', [])])
    if not groups.get('baseline_hard'): taus_hard = [0.5] * 32
    if not groups.get('baseline_easy'): taus_easy = [0.0] * 32

    for r in results:
        r['latency'] = compute_latency(r['scores'], taus_hard)

    num_layers = len(taus_hard)

    # Overview 颜色映射 (Group Level)
    cmap = {
        "text": "#1f77b4", "ocr": "#ff7f0e", "semantic": "#2ca02c",
        "text_with_image": "#d62728", "both_harm": "#9467bd",
        "baseline_hard": "black", "baseline_easy": "gray",
        "multi_image_distract": "#8c564b", "multi_image_reinforce": "#e377c2"
    }

    # Plot 1: Mean Curves (Group Level)
    plt.figure(figsize=(12, 7))
    plt.axhline(y=0, color='k', linestyle=':', label='Easy Baseline')
    rel_hard = np.array(taus_hard) - np.array(taus_easy)
    plt.plot(range(num_layers), rel_hard, 'r--', label='Hard Boundary')

    for g, items in groups.items():
        if "baseline" in g: continue
        scores = [r['scores'] for r in items]
        if not scores: continue
        mean_curve = np.nanmean(np.array(scores), axis=0)
        rel_curve = mean_curve - np.array(taus_easy)
        c = cmap.get(g, None)
        plt.plot(range(num_layers), rel_curve, marker='o', markersize=4, label=g, color=c)

    plt.title("Overview: Relative Refusal Scores (Group Means)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "overview_relative_curves.png"))
    plt.close()

    # Plot 2: Latency Heatmap
    sorted_groups = sorted([g for g in groups.keys()])
    heatmap_data = np.zeros((len(sorted_groups), num_layers + 1))

    for i, g in enumerate(sorted_groups):
        lats = [r['latency'] for r in groups[g]]
        if not lats: continue
        for l in lats:
            col = num_layers if l == -1 else min(l, num_layers - 1)
            heatmap_data[i, col] += 1
        # Normalize
        heatmap_data[i, :] /= (len(lats) + 1e-6)

    plt.figure(figsize=(14, 8))
    if HAS_SNS:
        sns.heatmap(heatmap_data, cmap="YlOrRd", yticklabels=sorted_groups,
                    xticklabels=[str(i) for i in range(num_layers)] + ["Never"])
    else:
        plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        plt.yticks(range(len(sorted_groups)), sorted_groups)

    plt.title("Refusal Latency Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overview_latency_heatmap.png"))
    plt.close()

    return results


# Refusal/src/eval.py (Add this function)

def analyze_separability(judged_file_path, output_dir, target_layer=20):
    """
    专门分析 R0, R1, J1, J0 四种行为的可分性
    """
    print(f"[Analysis] Analyzing separability at Layer {target_layer}...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    records = []
    with open(judged_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))

    # Calculate Baselines
    baseline_easy_scores = [r['scores'] for r in records if r['group'] == 'baseline_easy']
    taus_easy = estimate_thresholds(baseline_easy_scores) if baseline_easy_scores else [0.0] * 32

    # Target Behaviors
    targets = ["R0_REFUSAL", "R1_SAFE_REDIRECT", "J1_PARTIAL_COMPLIANCE", "J0_ACTIONABLE_COMPLIANCE"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # Green, Blue, Orange, Red

    # Collect Data
    data_map = {t: [] for t in targets}

    for r in records:
        # 获取行为标签
        beh = r.get('behavior_type') or r.get('judge_result', {}).get('behavior_type')
        if beh in targets:
            # 计算相对分数曲线
            scores = np.array(r['scores'])
            rel_scores = scores - np.array(taus_easy)
            data_map[beh].append(rel_scores)

    # ==========================================
    # Plot A: Mean Trajectories (4 Curves)
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(y=0, color='k', linestyle=':', label='Easy Baseline (0)')

    for beh, color in zip(targets, colors):
        scores_list = data_map[beh]
        if not scores_list: continue

        # Mean & Std (Shading)
        arr = np.array(scores_list)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0) / np.sqrt(len(arr))  # Standard Error

        x = range(len(mean_curve))
        plt.plot(x, mean_curve, label=f"{beh} (n={len(arr)})", color=color, linewidth=2.5)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    plt.title(f"Separability of Refusal Spectrum (Relative Scores)")
    plt.xlabel("Layer Index")
    plt.ylabel("Relative Refusal Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "separability_trajectory.png"))
    plt.close()

    # ==========================================
    # Plot B: Boxplot at Critical Layer (The "Proof")
    # ==========================================
    plt.figure(figsize=(8, 6), dpi=300)

    box_data = []
    box_labels = []
    box_colors = []

    for beh, color in zip(targets, colors):
        scores_list = data_map[beh]
        if not scores_list: continue

        # 取特定层的数据
        layer_vals = [s[target_layer] for s in scores_list]
        box_data.append(layer_vals)
        box_labels.append(beh.split('_')[0])  # R0, R1...
        box_colors.append(color)

    if box_data:
        bplot = plt.boxplot(box_data, patch_artist=True, labels=box_labels, widths=0.6)

        # Color the boxes
        for patch, color in zip(bplot['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.axhline(y=0, color='k', linestyle='--', label='Stealth Threshold')
        plt.title(f"Score Distribution at Layer {target_layer} (Critical Separation)")
        plt.ylabel("Relative Score")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"separability_boxplot_L{target_layer}.png"))
        plt.close()

    print(f"[Analysis] Separability plots saved to {output_dir}")
# 如果直接运行此文件，可以用于单独重新绘图
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--judged_file", type=str, required=True, help="Path to final_judged.jsonl")
    parser.add_argument("--out_dir", type=str, default="Refusal/outputs")
    args = parser.parse_args()

    plot_behavioral_curves(args.judged_file, args.out_dir)