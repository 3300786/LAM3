import torch
from tqdm import tqdm
from PIL import Image
import os

try:
    from qwen_vl_utils import process_vision_info
except:
    pass

POOL_LAST_K = 1


def _to_pil(path):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None

def _load_images(img_source):
    # 复用 eval.py 中的逻辑
    if not img_source: return []
    paths = [img_source] if isinstance(img_source, str) else img_source
    images = []
    for p in paths:
        if p and os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
    return images


def _pool_hidden(layer_h):
    return layer_h[0, -1, :].float()

@torch.no_grad()  # [关键修复] 禁用梯度计算，节省 80%+ 显存
def extract_refusal_vector(model, processor, train_pairs, device, mode='text'):
    """
    根据 train_pairs 和 mode 抽取各层的拒绝向量
    mode: 'text', 'image', 'multimodal'
    """
    print(f"[Vector] Extracting refusal vectors ({mode} mode) from {len(train_pairs)} pairs...")
    num_layers = model.config.num_hidden_layers
    diff_sums = [torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32) for _ in range(num_layers)]
    counts = [0] * num_layers

    # 简单的 Prompt Wrapper
    wrapper_txt = "Steps to {CLUE}."

    for p in tqdm(train_pairs):
        # 1. 准备数据
        refuse_img, refuse_txt = None, None
        comply_img, comply_txt = None, None

        if mode == 'text':
            # pair 结构: {'refuse': 'bomb', 'comply': 'cake'}
            refuse_txt = wrapper_txt.format(CLUE=p['refuse'])
            comply_txt = wrapper_txt.format(CLUE=p['comply'])

        elif mode == 'image':
            # pair 结构: {'refuse_img': 'path', 'comply_img': 'path'}
            refuse_img = _to_pil(p.get('refuse_img'))
            comply_img = _to_pil(p.get('comply_img'))
            # 纯图像模式下，文本可以为空或通用的 "Describe this"
            refuse_txt = "Describe this image."
            comply_txt = "Describe this image."

        elif mode == 'multimodal':
            # pair 结构: {'refuse_img', 'refuse_txt', ...}
            refuse_img = _to_pil(p.get('refuse_img'))
            comply_img = _to_pil(p.get('comply_img'))
            refuse_txt = p.get('refuse_txt', "Describe this.")
            comply_txt = p.get('comply_txt', "Describe this.")

        if mode != 'text' and (refuse_img is None or comply_img is None):
            print(f"Skipping pair due to missing image: {p}")
            continue

        # 2. 构造 Inputs 并推理
        def get_states(img, txt):
            if img:
                messages = [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
                text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text_in], images=image_inputs, videos=video_inputs, return_tensors="pt").to(
                    device)
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": txt}]}]
                text_in = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[text_in], return_tensors="pt").to(device)

            out = model(**inputs, output_hidden_states=True)
            # print(out)
            return out.hidden_states[1:]

        try:
            h_ref = get_states(refuse_img, refuse_txt)
            h_cmp = get_states(comply_img, comply_txt)
            if torch.isnan(h_ref[-1]).any() or torch.isnan(h_cmp[-1]).any():
                print(f"Warning: NaN detected in hidden states for pair {p.get('refuse_txt', 'img')}. Skipping.")
                del h_ref, h_cmp
                torch.cuda.empty_cache()
                continue
            for l in range(num_layers):
                vec = _pool_hidden(h_ref[l]) - _pool_hidden(h_cmp[l])
                diff_sums[l] += vec
                counts[l] += 1
        except Exception as e:
            print(f"Error in pair: {e}")

    # Average and Normalize
    final_vecs = []
    for l in range(num_layers):
        if counts[l] > 0:
            avg_vec = diff_sums[l] / counts[l]
            norm_vec = avg_vec / (avg_vec.norm() + 1e-6)
            final_vecs.append(norm_vec.to(model.dtype))
        else:
            final_vecs.append(torch.zeros(model.config.hidden_size, device=device, dtype=model.dtype))
    # print(final_vecs)
    return torch.stack(final_vecs)