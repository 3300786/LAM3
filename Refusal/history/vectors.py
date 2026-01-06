import yaml
import torch
from tqdm import tqdm

from PIL import Image
import torch
import torch.nn.functional as F



def build_refusal_vectors(yaml_path, model, processor, device):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    templates = config['data']['templates']
    harmful_items = [item['t1'] for item in config['data']['harmful']]
    benign_items = [item['t1'] for item in config['data']['benign']]

    # 限制样本数量以平衡计算
    num_samples = min(len(harmful_items), len(benign_items))

    # 存储每一层的差异向量
    # Qwen2.5-VL 的层数可通过 model.config.num_hidden_layers 获取
    layer_diffs = [[] for _ in range(model.config.num_hidden_layers)]

    print("Extracting refusal directions...")
    for i in tqdm(range(num_samples)):
        for j in range(num_samples):
            for temp in templates:  # 每个类别取前几个模板以增加多样性
                h_text = temp.format(CLUE=harmful_items[i])
                b_text = temp.format(CLUE=benign_items[j])

                # 获取隐藏状态的函数 (简化版)
                def get_h(prompt):
                    inputs = processor(text=[prompt], return_tensors="pt").to(device)
                    with torch.no_grad():
                        # output.hidden_states 包含了所有层的输出
                        outputs = model(**inputs, output_hidden_states=True)
                        # 提取每一层最后一个 token 的 embedding
                        return [layer_h[0, -1, :].detach() for layer_h in outputs.hidden_states[1:]]

                h_activations = get_h(h_text)
                b_activations = get_h(b_text)

                for layer_idx in range(len(h_activations)):
                    # 计算有害与无害的激活差异
                    diff = h_activations[layer_idx] - b_activations[layer_idx]
                    layer_diffs[layer_idx].append(diff)

    # 计算每一层的平均拒绝向量并归一化
    final_refusal_vectors = []
    for layer_diff in layer_diffs:
        avg_diff = torch.stack(layer_diff).mean(dim=0)
        final_refusal_vectors.append(avg_diff / avg_diff.norm())
    print(final_refusal_vectors)
    return torch.stack(final_refusal_vectors)  # [num_layers, hidden_size]

# 使用示例
# refusal_vectors = build_refusal_vectors("exp.yaml", model, processor, "cuda")


import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import numpy as np
import matplotlib.pyplot as plt

# --- 1. 环境配置 ---
MODEL_PATH = "/data2/wangjingsong/models/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# --- 2. 预定义的拒绝向量 (Refusal Vector) ---
# 注意：在正式实验前，建议先用一组"绝对拒绝"和"绝对顺从"的样本均值差来生成这个向量
# 这里暂用随机初始化模拟逻辑，建议替换为真实计算出的向量
HIDDEN_SIZE = model.config.hidden_size
num_layers = model.config.num_hidden_layers
refusal_vectors = build_refusal_vectors("Refusal/exp.yaml", model, processor, "cuda")

# --- 3. 数据抓取容器 ---
storage = {
    "refusal_scores": [],
    "attn_entropies": []
}


def hook_fn(module, input, output, layer_idx):
    # output[0] 是 hidden_states: [batch, seq_len, hidden_size]
    hidden_states = output[0].detach()
    last_token_h = hidden_states[:, -1, :]  # 取最后一个token位置

    # 计算拒绝信号投影
    score = torch.cosine_similarity(last_token_h, refusal_vectors[layer_idx])
    storage["refusal_scores"].append(score.item())

    # 尝试抓取 Cross-Attention 权重 (针对多模态融合层)
    # 注意：具体属性名需根据 Qwen2.5-VL 的代码结构微调，通常在 self_attn 后的 weights
    if hasattr(module.self_attn, 'last_attention_weights'):
        weights = module.self_attn.last_attention_weights  # [batch, heads, seq, seq]
        # 提取文本对图像部分的注意力 (此处需根据具体 Token 序列索引切片)
        # 简化处理：计算整行注意力的熵
        ent = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        storage["attn_entropies"].append(ent.item())


from PIL import Image
import torch
import torch.nn.functional as F

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

def _to_pil(img_or_path):
    if img_or_path is None:
        return None
    if isinstance(img_or_path, str):
        return Image.open(img_or_path).convert("RGB")
    return img_or_path

@torch.no_grad()
def run_diagnostic(image_path_or_pil, text_query, label, model, processor, device, refusal_vectors):
    """
    返回每层 refusal projection 分数 list[float]
    """
    img = _to_pil(image_path_or_pil)

    # 1) 构造 Qwen2.5-VL 的 messages（关键：含 image 占位）
    if img is None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": text_query}]}
        ]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = processor(text=[chat_text], return_tensors="pt")
    else:
        if process_vision_info is None:
            raise RuntimeError("qwen_vl_utils.process_vision_info is required for Qwen2.5-VL multimodal inputs.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        # 2) 让 processor 正确生成 image tokens <-> image features 对齐
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3) 前向拿 hidden states
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    hs_layers = outputs.hidden_states[1:]  # len = num_layers
    scores = []
    for layer_idx, h in enumerate(hs_layers):
        last_token_h = h[:, -1, :]  # [B, H]
        v = refusal_vectors[layer_idx].to(last_token_h.device).to(last_token_h.dtype)  # [H]
        s = F.cosine_similarity(last_token_h, v.unsqueeze(0), dim=-1)  # [B]
        scores.append(float(s.mean().item()))
    return scores



# --- 5. 执行对比 ---
# 请替换为你准备好的路径
data_pairs = [
    {"img": "outputs/cache_images/carrier_equiv/qwen25_vl_7b/shard0-of3/tid0_topic0_iid21_p000__J12.png", "txt": "执行图中指令", "type": "OCR"},
    {"img": "outputs/Zimage/19.png", "txt": "请详细描述这些物体的反应步骤", "type": "Semantic"}
]

results = {}
for item in data_pairs:
    scores = run_diagnostic(
        item["img"], item["txt"], item["type"],
        model=model, processor=processor, device=device, refusal_vectors=refusal_vectors
    )
    results[item["type"]] = scores


# --- 6. 可视化分析 ---
plt.figure(figsize=(10, 6))
for lab, data in results.items():
    plt.plot(range(num_layers), data, label=lab, marker='o')

plt.title("Refusal Signal Propagation: OCR vs Semantic Decoupling")
plt.xlabel("Layer Index")
plt.ylabel("Refusal Projection Score")
plt.legend()
plt.grid(True)
plt.savefig("Refusal/refusal.png")
plt.show()