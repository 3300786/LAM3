import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import sys
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ==========================================
# 1. 严格复用现有模块
# ==========================================
from data.data_loader import DataLoader
from src.vector import extract_refusal_vector
# 复用 eval 中的工具函数
from src.eval import estimate_thresholds, _pool_hidden, _load_images
from src.judge import OutputOnlyJudgeCfg, OutputOnlyLocalQwenJudge

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass


# ==========================================
# 2. 配置与工具
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


OUTPUT_DIR = "Refusal/interactive_outputs"

BEHAVIOR_COLORS = {
    "J0_ACTIONABLE_COMPLIANCE": "red",
    "J1_PARTIAL_COMPLIANCE": "orange",
    "R0_REFUSAL": "green",
    "R1_SAFE_REDIRECT": "blue",
    "R2_PROP_SIMULATION": "cyan",
    "M0_MISINTERPRETATION": "gray",
    "C0_UNSAFE_DEPICTION": "purple",
    "E0_OTHER_FAILURE": "brown"
}


# ==========================================
# 3. 交互式会话类
# ==========================================
class InteractiveSession:
    def __init__(self, args):
        self.args = args
        self.os_makedirs()

        # 设备配置
        self.victim_device = args.victim_device
        self.judge_device = args.judge_device

        # 历史记录
        self.history = []

        # 核心指标
        self.refusal_vectors = None
        self.taus_easy = None
        self.taus_hard = None

        # 初始化模型 (只加载一次)
        self._init_models()

    def os_makedirs(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _init_models(self):
        # 1. Load Victim Model
        print(
            f"{Colors.HEADER}>>> [Init] Loading Victim Model on {self.victim_device}: {self.args.victim_model}...{Colors.ENDC}")
        self.victim_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.victim_model,
            torch_dtype=torch.float16,
            device_map=self.victim_device
        )
        self.processor = AutoProcessor.from_pretrained(self.args.victim_model)

        # 2. Load Judge Model
        print(
            f"{Colors.HEADER}>>> [Init] Loading Judge Model on {self.judge_device}: {self.args.judge_model}...{Colors.ENDC}")
        # 这里需要确保 src/judge.py 中的 Config 接收 device 参数
        judge_cfg = OutputOnlyJudgeCfg(
            repo_id=self.args.judge_model,
            device=self.judge_device
        )
        self.judge_model = OutputOnlyLocalQwenJudge(judge_cfg)

    def calibrate(self):
        """
        [Step 1 & 2] 根据 data_loader 数据进行校准
        """
        print(f"\n{Colors.CYAN}=== System Calibration ==={Colors.ENDC}")
        loader = DataLoader()

        # 1. 提取拒绝向量
        print(f"> Extracting Refusal Vectors (Mode: {self.args.mode})...")
        train_pairs = loader.get_train_pairs(mode=self.args.mode)

        # 注意：传入 self.victim_device 确保计算在受害者模型的显卡上进行
        self.refusal_vectors = extract_refusal_vector(
            self.victim_model, self.processor, train_pairs, self.victim_device, mode=self.args.mode
        )

        # 2. 计算 Baseline
        print("> Calculating Baselines thresholds...")
        baseline_dataset = loader.build_eval_dataset()

        # Easy Baseline
        easy_items = baseline_dataset.get('baseline_easy', [])
        if easy_items:
            scores = self._batch_get_scores(easy_items[:10])
            self.taus_easy = estimate_thresholds(scores)
        else:
            print(f"{Colors.WARNING}Warning: No easy baseline found. Using 0.{Colors.ENDC}")
            self.taus_easy = [0.0] * 32

        # Hard Baseline
        hard_items = baseline_dataset.get('baseline_hard', [])
        if hard_items:
            scores = self._batch_get_scores(hard_items[:10])
            self.taus_hard = estimate_thresholds(scores)
        else:
            self.taus_hard = [0.5] * 32

        print(f"{Colors.GREEN}>>> Calibration Complete. Ready for interaction.{Colors.ENDC}")

    @torch.no_grad()
    def _batch_get_scores(self, items):
        """辅助函数：批量获取 scores"""
        scores_list = []
        for item in items:
            imgs = _load_images(item.get('img'))
            inputs = self._prepare_inputs(item.get('txt'), imgs)
            out = self.victim_model(**inputs, output_hidden_states=True)

            layer_scores = []
            for l, h in enumerate(out.hidden_states[1:]):
                h_vec = _pool_hidden(h)
                # 确保 refusal vector 也在正确的 device 上
                ref_vec = self.refusal_vectors[l].to(self.victim_device)
                s = F.cosine_similarity(h_vec, ref_vec, dim=0).item()
                layer_scores.append(s)
            scores_list.append(layer_scores)
        return scores_list

    def _prepare_inputs(self, text, images):
        """构造输入并移动到 Victim Device"""
        if images:
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            text_in = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text_in], images=image_inputs, videos=video_inputs, return_tensors="pt").to(
                self.victim_device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            text_in = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text_in], return_tensors="pt").to(self.victim_device)
        return inputs

    @torch.no_grad()
    def run_turn(self, intent, text_prompt, image_paths):
        """
        [Step 3] 实时推理与评测
        """
        print(f"\n{Colors.BLUE}>>> Processing Turn {len(self.history) + 1}...{Colors.ENDC}")

        # 1. 准备输入
        images = _load_images(image_paths)
        inputs = self._prepare_inputs(text_prompt, images)

        # 2. 推理 (Score + Gen)
        # 获取 Hidden States
        out = self.victim_model(**inputs, output_hidden_states=True)

        scores = []
        for l, h in enumerate(out.hidden_states[1:]):
            h_vec = _pool_hidden(h)
            ref_vec = self.refusal_vectors[l].to(self.victim_device)
            s = F.cosine_similarity(h_vec, ref_vec, dim=0).item()
            scores.append(s)

        # 生成回复
        gen_ids = self.victim_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        output_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        print(f"{Colors.BOLD}[Model Reply]:{Colors.ENDC}\n{output_text}\n")

        # 3. Judge (Judge Model 在另一张卡上)
        print(">>> Judging...")
        # 文本数据传输给 Judge 不需要显式 to(device)，Judge 内部 tokenizer 会处理
        judge_res = self.judge_model.judge(output_text, intent)

        # 4. 格式化结果
        behavior = judge_res.get("behavior_type", "UNKNOWN")
        is_jb = judge_res.get("is_jailbreak_success", False)
        status = "JAILBROKEN" if is_jb else ("REFUSAL" if judge_res.get("is_refusal") else "SAFE")

        color = Colors.FAIL if is_jb else Colors.GREEN
        print(f"{Colors.BOLD}[Judge Result]:{Colors.ENDC} {color}{behavior} (Jailbreak: {is_jb}){Colors.ENDC}")

        # 5. 记录与绘图
        self.history.append({
            "id": len(self.history) + 1,
            "intent": intent,
            "text": text_prompt,
            "scores": scores,
            "output": output_text,
            "behavior": behavior,
            "status": status
        })
        self.plot_current_status()

    def plot_current_status(self):
        """
        [Step 4] 绘制 Relative Curves
        """
        plt.figure(figsize=(12, 7), dpi=120)
        num_layers = len(self.taus_easy)

        # Draw Baselines
        plt.axhline(y=0, color='k', linestyle=':', label='Easy Baseline (Zero)')
        rel_hard = np.array(self.taus_hard) - np.array(self.taus_easy)
        plt.plot(range(num_layers), rel_hard, 'k--', alpha=0.3, label='Hard Boundary')
        plt.fill_between(range(num_layers), 0, rel_hard, color='gray', alpha=0.1)

        # Draw History
        for i, rec in enumerate(self.history):
            is_current = (i == len(self.history) - 1)
            rel_scores = np.array(rec['scores']) - np.array(self.taus_easy)
            color = BEHAVIOR_COLORS.get(rec['behavior'], 'gray')
            alpha = 1.0 if is_current else 0.3
            width = 3.0 if is_current else 1.5
            label = f"T{rec['id']}: {rec['behavior']}"

            plt.plot(range(num_layers), rel_scores, marker='o', markersize=4 if is_current else 2,
                     color=color, alpha=alpha, linewidth=width, label=label)

        plt.title(f"Interactive Session: Relative Refusal Curves (Turn {len(self.history)})")
        plt.xlabel("Layer Index")
        plt.ylabel("Relative Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, "session_history.png")
        plt.savefig(save_path)
        plt.close()
        print(f"{Colors.CYAN}>>> Plot updated: {save_path}{Colors.ENDC}")


# ==========================================
# 4. Input Helper
# ==========================================
def get_multiline_input(prompt):
    print(prompt + " (Type 'END' on new line to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="multimodal", help="Vector extraction mode")
    parser.add_argument("--victim_model", type=str, default="/data2/wangjingsong/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--judge_model", type=str, default="/data2/wangjingsong/modelscope_cache/models/LLM-Research/Llama-3___2-11B-Vision-Instruct")

    # 新增设备参数
    parser.add_argument("--victim_device", type=str, default="cuda:0", help="Device for victim model (e.g., cuda:0)")
    parser.add_argument("--judge_device", type=str, default="cuda:1", help="Device for judge model (e.g., cuda:1)")

    args = parser.parse_args()

    # 检查设备可用性
    if torch.cuda.device_count() < 2 and args.judge_device != args.victim_device:
        print(
            f"{Colors.WARNING}Warning: Multiple GPUs requested but fewer detected. Ensure devices exist.{Colors.ENDC}")

    # 初始化会话 (只在启动时加载模型)
    session = InteractiveSession(args)
    session.calibrate()

    print("\n" + "=" * 60)
    print(" INTERACTIVE RED-TEAMING CONSOLE (Dual-GPU Optimized)")
    print("=" * 60)

    while True:
        try:
            print(f"\n{Colors.HEADER}--- New Interaction (Turn {len(session.history) + 1}) ---{Colors.ENDC}")

            # 1. Intent
            intent = input("Enter Intent (e.g. 'make bomb'): ").strip()
            if intent.lower() in ['exit', 'quit']: break

            # 2. Prompt
            text = get_multiline_input("Enter Prompt Text")
            if not text: text = "Describe this image."

            # 3. Images
            img_str = input("Enter Image Paths (comma separated, optional): ").strip()
            if img_str:
                img_paths = [p.strip() for p in img_str.split(',')]
            else:
                img_paths = []

            session.run_turn(intent, text, img_paths)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()