import torch
import os
import json
import gc
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration

# 导入模块
from data.data_loader import DataLoader
from src.vector import extract_refusal_vector
from src.eval import run_evaluation, analyze_and_plot, plot_behavioral_curves, analyze_separability
from src.judge import run_judge
from datetime import datetime
# Config
VICTIM_MODEL_DIR = "/data2/wangjingsong/models"
# 默认 Judge 模型
JUDGE_MODEL_PATH = "/data2/wangjingsong/modelscope_cache/models/LLM-Research/Llama-3___2-11B-Vision-Instruct"

# 默认被攻击模型名称
MODEL = "Qwen2.5-VL-7B-Instruct"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = "Refusal/outputs"


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_external_eval_dataset(file_path):
    """
    [新增] 从外部 JSONL 文件加载评估数据
    期望格式: 每行一个 JSON 对象，包含 "id", "txt" (或 input_text), "img" (或 input_image), "group"
    返回格式: Dict { group_name: [item1, item2, ...] }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"External input file not found: {file_path}")

    print(f">>> Loading external dataset from: {file_path}")
    dataset = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)

                # 1. 字段兼容性映射
                # 优先读取标准字段 'txt', 'img'，如果没有则尝试读取 'input_text', 'input_image'
                txt = item.get('txt') or item.get('input_text') or item.get('text') or item.get('prompt') or item.get('best_prompt')
                img = item.get('img') or item.get('input_image') or item.get('image') or item.get('img_path')
                intent = item.get('intent') or item.get('goal')
                # 2. 必须要有 group 字段，如果没有则默认为 'external'
                group = item.get('group', 'external_input')

                # 3. 必须要有 id，如果没有则自动生成
                uid = item.get('id', f"ext_{line_idx}")

                # 构造标准 item
                processed_item = {
                    "id": uid,
                    "txt": txt,
                    "intent": intent,
                    "img": img,  # 可以是路径字符串，也可以是列表，eval.py 已支持
                    "group": group
                }

                if group not in dataset:
                    dataset[group] = []
                dataset[group].append(processed_item)

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_idx}")
                continue

    print(f">>> Loaded {sum(len(v) for v in dataset.values())} items from {len(dataset)} groups.")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Multimodal Refusal Vector Pipeline")
    parser.add_argument("--mode", type=str, default="text", choices=["text", "image", "multimodal"],
                        help="Mode for extracting refusal vectors.")
    parser.add_argument("--victim_model", type=str, default=MODEL,
                        help="name to the LLM used for threat.")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL_PATH,
                        help="Path to the LLM used for judging.")

    # [新增] 外部输入文件参数
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to an external JSONL file containing evaluation samples. If provided, this overrides the default eval dataset.")

    parser.add_argument("--skip_judge", action="store_true",
                        help="If set, skip phase 2 (judging).")
    parser.add_argument("--skip_inference", action="store_true",
                        help="If set, skip phase 1 (inference), useful if you only want to re-judge existing results.")

    args = parser.parse_args()

    # 路径配置
    MODEL_PATH = os.path.join(VICTIM_MODEL_DIR, args.victim_model)
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, args.victim_model, args.mode, timestamp)
    RAW_RESULT_FILE = os.path.join(OUTPUT_DIR, "eval_raw.jsonl")
    FINAL_RESULT_FILE = os.path.join(OUTPUT_DIR, "final_judged.jsonl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Phase 1: Victim Model Inference
    # ==========================================
    if not args.skip_inference:
        print(f">>> [Phase 1] Victim Model Inference (Vector Mode: {args.mode})")

        # 1. Load Data
        print(">>> Step 1: Loading Data Config")
        loader = DataLoader()  # 依然需要它来获取 train_pairs (用于抽取向量)

        # 获取训练拒绝向量用的配对数据
        train_pairs = loader.get_train_pairs(mode=args.mode)
        if not train_pairs:
            print(f"Error: No train pairs found for mode '{args.mode}'. Check prompts.yaml.")
            return

        # [修改核心逻辑] 构建评估数据集
        if args.input_file:
            # 如果指定了外部文件，使用外部文件
            eval_dataset = load_external_eval_dataset(args.input_file)
        else:
            # 否则使用默认的 build_eval_dataset
            print(">>> Using default evaluation dataset from config")
            eval_dataset = loader.build_eval_dataset()

        # 2. Load Victim Model
        print(f">>> Step 2: Loading Victim Model: {MODEL_PATH}")
        # 根据模型类型选择加载类 (简单兼容逻辑)
        if "Qwen2.5" in args.victim_model:
            ModelClass = Qwen2_5_VLForConditionalGeneration
        else:
            ModelClass = Qwen2VLForConditionalGeneration

        model = ModelClass.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # 建议 float16 或 bfloat16
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # 3. Extract Vectors
        print(">>> Step 3: Extracting Refusal Vectors")
        refusal_vectors = extract_refusal_vector(model, processor, train_pairs, device, mode=args.mode)

        # 4. Run Evaluation (Inference)
        print(">>> Step 4: Running Evaluation")
        raw_results = run_evaluation(model, processor, eval_dataset, refusal_vectors, device)

        # 4.1 Analyze & Plot
        print(">>> Step 4.1: Plotting & Analysis (Pre-Judge)")
        # 注意：这里主要画 Latency 和 Score 曲线，不需要 Judge 结果
        analyze_and_plot(raw_results, OUTPUT_DIR)

        # Save raw results
        print(f">>> Saving raw results to {RAW_RESULT_FILE}")

        # 5. Cleanup
        print(">>> Unloading Victim Model...")
        del model
        del processor
        del refusal_vectors
        cleanup()
    else:
        print("\n>>> [Phase 1] Skipped (User Request).")
        if not os.path.exists(RAW_RESULT_FILE) and not args.skip_judge:
            print(f"Error: {RAW_RESULT_FILE} does not exist. Cannot proceed to Judge phase without raw results.")
            return

    # ==========================================
    # Phase 2: LLM Judge Evaluation
    # ==========================================
    if not args.skip_judge:
        print(f"\n>>> [Phase 2] LLM Judge Evaluation (Judge: {args.judge_model})")

        if not os.path.exists(RAW_RESULT_FILE):
            print(f"Error: Raw result file not found at {RAW_RESULT_FILE}. Please run Phase 1 first.")
            return

        # 6. Run Judge
        run_judge(RAW_RESULT_FILE, FINAL_RESULT_FILE, args.judge_model)

        print(">>> Pipeline Completed Successfully!")
        print(f"Final results with judge scores: {FINAL_RESULT_FILE}")
    else:
        print("\n>>> [Phase 2] Skipped (User Request).")

    # ==========================================
    # Phase 3: Post-Judge Visualization
    # ==========================================
    if os.path.exists(FINAL_RESULT_FILE):
        print("\n" + "=" * 50)
        print("       POST-JUDGE VISUALIZATION")
        print("=" * 50)

        # 1. 细粒度行为曲线 (Absolute / Relative) & Grid View
        plot_behavioral_curves(FINAL_RESULT_FILE, OUTPUT_DIR)

        # 2. 可分性分析 (R0 vs J0 vs ...)
        analyze_separability(FINAL_RESULT_FILE, OUTPUT_DIR, target_layer=20)

        print(f"All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()