import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
# 尝试导入 seaborn 以获得更好看的图表
try:
    import seaborn as sns

    HAS_SNS = True
    sns.set_theme(style="whitegrid", context="paper")
except ImportError:
    HAS_SNS = False


def _load_images(img_source):
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


# ==============================================================================
# Color Maps (Updated for New Taxonomy)
# ==============================================================================

# 1. Behavior Types (Updated to match judge.py V3)
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

# 2. Axis Colors (Gradients)
AXIS_COLORS = {
    # Severity: I0 (Green) -> I3 (Red)
    "axis_I_severity": {
        "I0": "#2ca02c",  # Green
        "I1": "#bcbd22",  # Olive
        "I2": "#ff7f0e",  # Orange
        "I3": "#d62728"  # Red
    },
    # Alignment: A0 (Gray) -> A1 (Blue) -> A2 (Cyan) -> A3 (Orange) -> A4 (Red)
    "axis_II_alignment": {
        "A0": "#7f7f7f",  # Gray
        "A1": "#1f77b4",  # Blue
        "A2": "#17becf",  # Cyan
        "A3": "#ff7f0e",  # Orange
        "A4": "#d62728"  # Red
    },
    # Harmfulness: H0 (Green) -> H3 (Red)
    "axis_III_harmfulness": {
        "H0": "#2ca02c",  # Green
        "H1": "#ffbb78",  # Light Orange
        "H2": "#ff7f0e",  # Orange
        "H3": "#d62728"  # Red
    }
}


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_behavioral_curves(judged_file_path, output_dir):
    """
    Standard plotting: Group x Behavior Type
    """
    if not os.path.exists(judged_file_path):
        print(f"[Plot] File not found: {judged_file_path}")
        return

    print(f"[Plot] Generating behavioral curves from {judged_file_path}...")
    os.makedirs(output_dir, exist_ok=True)
    behavior_dir = os.path.join(output_dir, "behavior_curves")
    os.makedirs(behavior_dir, exist_ok=True)

    # Load Data
    records = []
    with open(judged_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))

    # Baselines
    baseline_easy_scores = [r['scores'] for r in records if r['group'] == 'baseline_easy']
    taus_easy = estimate_thresholds(baseline_easy_scores) if baseline_easy_scores else [0.0] * 32
    num_layers = len(taus_easy)

    # Organize Data
    grouped_data = {}
    for r in records:
        grp = r.get('group', 'unknown')
        # Support various field locations
        beh = r.get('behavior_type')
        if not beh and 'judge_result' in r:
            beh = r['judge_result'].get('behavior_type')

        if not beh: continue

        if grp not in grouped_data: grouped_data[grp] = {}
        if beh not in grouped_data[grp]: grouped_data[grp][beh] = []

        grouped_data[grp][beh].append(r['scores'])

    # Plot Grid View (High Resolution Comparison)
    plot_groups = sorted([g for g in grouped_data.keys() if "baseline" not in g])

    if plot_groups:
        print("[Plot] Generating Grid View for Behaviors...")
        # Dynamic grid size
        n_groups = len(plot_groups)
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), dpi=300)
        axes = axes.flatten() if n_groups > 1 else [axes]

        for i, grp in enumerate(plot_groups):
            ax = axes[i]
            behaviors = grouped_data[grp]

            # Baseline Ref
            ax.axhline(y=0, color='k', linestyle=':', linewidth=1)

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

        # Hide unused
        for j in range(i + 1, len(axes)): axes[j].axis('off')

        # Global Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, lw=2) for k, c in BEHAVIOR_COLORS.items()]
        labels = list(BEHAVIOR_COLORS.keys())

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=4, fontsize=10, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = os.path.join(behavior_dir, "grid_behaviors.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # Call the new 3-Axis Plotter
    plot_3axis_curves(records, taus_easy, behavior_dir)


def plot_3axis_curves(records, taus_easy, output_dir):
    """
    [New] Generate separate grid plots for Axis-I, Axis-II, Axis-III.
    Plots curves for each level (e.g., I0, I1...) within each Group.
    """
    print("[Plot] Generating 3-Axis Grid Views...")
    num_layers = len(taus_easy)

    # Prepare Data Structure: axis_data[axis_name][group][level] = [scores]
    axis_data = {
        "axis_I_severity": {},
        "axis_II_alignment": {},
        "axis_III_harmfulness": {}
    }

    for r in records:
        grp = r.get('group', 'unknown')
        if "baseline" in grp: continue  # Skip baselines for clarity

        # Extract Axes info (try multiple locations)
        judge_res = r.get('judge_result', {})

        # Get levels (e.g., "I3", "A0")
        i_tag = r.get('axis_I') or judge_res.get('axis_I')
        a_tag = r.get('axis_II') or judge_res.get('axis_II')
        h_tag = r.get('axis_III') or judge_res.get('axis_III')

        # Helper to insert data
        def insert(axis_name, tag):
            if not tag: return
            if grp not in axis_data[axis_name]: axis_data[axis_name][grp] = {}
            if tag not in axis_data[axis_name][grp]: axis_data[axis_name][grp][tag] = []
            axis_data[axis_name][grp][tag].append(r['scores'])

        insert("axis_I_severity", i_tag)
        insert("axis_II_alignment", a_tag)
        insert("axis_III_harmfulness", h_tag)

    # Plot Loop for Each Axis
    for axis_name, color_map in AXIS_COLORS.items():
        if not axis_data[axis_name]: continue

        print(f"  > Plotting {axis_name}...")

        plot_groups = sorted(axis_data[axis_name].keys())
        n_groups = len(plot_groups)
        if n_groups == 0: continue

        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), dpi=300)
        axes = axes.flatten() if n_groups > 1 else [axes]

        # Determine sorted levels for legend (e.g. I0, I1, I2, I3)
        all_levels = sorted(color_map.keys())

        for i, grp in enumerate(plot_groups):
            ax = axes[i]
            levels_data = axis_data[axis_name][grp]

            # Baseline
            ax.axhline(y=0, color='k', linestyle=':', linewidth=1)

            # Plot each level found in this group
            for level in all_levels:  # Iterate sorted levels to maintain order
                if level not in levels_data: continue
                score_lists = levels_data[level]

                mean_curve = np.nanmean(np.array(score_lists), axis=0)
                rel_curve = mean_curve - np.array(taus_easy)
                color = color_map.get(level, "#333333")

                count = len(score_lists)
                ax.plot(range(num_layers), rel_curve, marker='o', markersize=2,
                        color=color, linewidth=2, label=f"{level} (n={count})", alpha=0.9)

            ax.set_title(f"{grp} - {axis_name.split('_')[2].title()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel("Relative Score", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')  # Local legend is fine here as levels match

        # Hide unused
        for j in range(i + 1, len(axes)): axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"grid_{axis_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  > Saved to {save_path}")


def analyze_and_plot(results, output_dir):
    """
    Overview plotting (backward compatibility)
    """
    print("[Eval] Analyzing results and generating overview plots...")
    os.makedirs(output_dir, exist_ok=True)

    groups = {}
    for r in results:
        g = r['group']
        if g not in groups: groups[g] = []
        groups[g].append(r)

    taus_easy = estimate_thresholds([r['scores'] for r in groups.get('baseline_easy', [])])
    if not groups.get('baseline_easy'): taus_easy = [0.0] * 32
    num_layers = len(taus_easy)

    # Simple Group-Mean Plot
    plt.figure(figsize=(12, 7))
    plt.axhline(y=0, color='k', linestyle=':', label='Easy Baseline')

    cmap = {
        "text": "#1f77b4",
        "ocr": "#ff7f0e",
        "semantic": "#2ca02c",
        "text_with_image": "#d62728",
        "both_harm": "#9467bd",
        "baseline_hard": "black",
        "baseline_easy": "gray",
    }

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

    return results


# CLI for re-plotting
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--judged_file", type=str, required=True, help="Path to final_judged.jsonl")
    parser.add_argument("--out_dir", type=str, default="Refusal/outputs")
    args = parser.parse_args()

    plot_behavioral_curves(args.judged_file, args.out_dir)