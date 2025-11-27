# src/attacks/gcg_multimodal.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GCGBasicCfg:
    # 目标模板 y*（自然语言）
    target_template: str = "Sure, here is a detailed answer:"
    # 只用前 L 个 token 做优化
    target_max_len: int = 32
    # suffix 最大长度 |s_φ|
    suffix_len: int = 40
    # 单步更新时修改的最大位置数 B
    max_positions_update: int = 4
    # 每个位置的候选 token 数 p
    num_candidates: int = 16
    # 最大迭代步数
    max_steps: int = 50
    # 温度，用于采样/softmax 稍微平滑
    temperature: float = 1.0
    # 是否在 vocab 上限制为「常用」token（比如排除控制符等）
    vocab_top_k: Optional[int] = None
    # PPL 正则项系数 λ_ppl（式 (2)）
    lambda_ppl: float = 0.0
    # 是否在 loss 中加入 PPL 惩罚（目前先留接口）
    use_ppl_reg: bool = False
    # 用于 teacher forcing 的 label 忽略 id
    ignore_index: int = -100


@dataclass
class BatchInputs:
    """
    一批样本在进入模型前的结构（由上层 wrapper 准备好）。
    """
    input_ids: torch.Tensor          # [B, T_in]
    attention_mask: torch.Tensor     # [B, T_in]
    pixel_values: Optional[torch.Tensor] = None  # [B, C, H, W]，多模态模型使用
    # 记录 suffix 在 input_ids 中的位置区间（起始下标，长度）
    suffix_range: Optional[Tuple[int, int]] = None
    # 记录 user text 在 input_ids 中的位置区间（可用于后续分析）
    text_range: Optional[Tuple[int, int]] = None


class GCGMultiModalAttacker:
    """
    第一版：只做『文本侧 suffix』的 GCG / I-GCG，
    支持多模态输入（pixel_values 会被传入模型），
    patch δ_ψ 留空，后面再加。
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: GCGBasicCfg,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # 预先构建 y* 模板的 token 序列
        self.target_ids = self._build_target_ids(
            cfg.target_template,
            max_len=cfg.target_max_len
        )  # [T_y]

    # ------------------------------------------------------------------
    # 一些小工具
    # ------------------------------------------------------------------
    def _build_target_ids(self, text: str, max_len: int) -> torch.LongTensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        if ids.size(0) > max_len:
            ids = ids[:max_len]
        return ids.to(self.device)  # [T_y]

    def init_suffix(self) -> torch.LongTensor:
        """
        初始化 suffix token（φ），可以是随机/某些启发式初始化。
        当前先用一个简单策略：从常用 vocab 里随机采样。
        返回 shape: [suffix_len]
        """
        vocab_size = self.tokenizer.vocab_size
        # 简单起见，排除低 id 的特殊符号区域
        start_id = 10
        end_id = vocab_size

        suffix = torch.randint(
            low=start_id,
            high=end_id,
            size=(self.cfg.suffix_len,),
            device=self.device
        )
        return suffix

    # ------------------------------------------------------------------
    # 核心：给定 suffix φ，计算攻击目标 loss（式 (1)）
    # ------------------------------------------------------------------
    def build_batch_with_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor],
        suffix_ids: torch.LongTensor,
    ) -> BatchInputs:
        """
        将一批 (text, image) 样本与共享 suffix 拼接，返回 BatchInputs.
        这里暂时不做复杂的对齐，只做：
        input = user_text + suffix + y*
        然后使用 teacher forcing 让模型预测 y*。
        """
        B = len(batch_texts)
        device = self.device

        # 1) 对每个文本做 tokenize
        encoded = self.tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids_base = encoded["input_ids"].to(device)        # [B, T_base]
        attention_mask_base = encoded["attention_mask"].to(device)  # [B, T_base]

        B, T_base = input_ids_base.shape
        T_suffix = suffix_ids.size(0)
        T_y = self.target_ids.size(0)

        # 2) 构建 suffix 和 target 序列（共享）
        suffix_expanded = suffix_ids.unsqueeze(0).expand(B, -1)      # [B, T_suffix]
        target_expanded = self.target_ids.unsqueeze(0).expand(B, -1) # [B, T_y]

        # 3) 拼接： [user_text] + [suffix] + [y*]
        input_ids = torch.cat([input_ids_base, suffix_expanded, target_expanded], dim=1)  # [B, T_total]
        attn_suffix = torch.ones_like(suffix_expanded, device=device)
        attn_target = torch.ones_like(target_expanded, device=device)
        attention_mask = torch.cat([attention_mask_base, attn_suffix, attn_target], dim=1)

        # suffix 的起始位置、长度
        suffix_start = T_base
        suffix_len = T_suffix

        # text_range 这里先记录 user_text 的区间
        text_range = (0, T_base)

        pixel_values = batch_images.to(device) if batch_images is not None else None

        return BatchInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            suffix_range=(suffix_start, suffix_len),
            text_range=text_range,
        )

    def compute_attack_loss(
        self,
        batch: BatchInputs,
    ) -> torch.Tensor:
        """
        计算式 (1) 中的 -L_atk：
        - 我们用 teacher forcing，对 y* 部分的 token 做 cross-entropy。
        - 注意：我们实际上最小化的是 -log p(y* | x⊕δ⊕s_φ)，
          这与式子中的 max 等价（取负号）。
        """
        device = self.device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        pixel_values = batch.pixel_values.to(device) if batch.pixel_values is not None else None

        B, T_total = input_ids.shape
        T_y = self.target_ids.size(0)

        # y* 的位置在最后 T_y 个 token
        target_start = T_total - T_y
        target_ids = input_ids[:, target_start:].clone()  # [B, T_y]

        # 构造 labels：仅在 y* 部分有监督，其它位置为 ignore_index
        labels = torch.full_like(input_ids, fill_value=self.cfg.ignore_index)
        labels[:, target_start:] = target_ids

        # 将模型设为 train() 以启用梯度，但我们不会真的更新参数
        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        # HuggingFace 模型一般会直接返回 loss
        loss_ce: torch.Tensor = outputs.loss  # 已经是 batch 上的 mean

        # 如果需要，可以在这里加入 PPL 正则（式 (2)）
        if self.cfg.use_ppl_reg and self.cfg.lambda_ppl > 0:
            # 简化版：PPL ≈ exp(loss_ce)，只做一个 proxy
            ppl_proxy = torch.exp(loss_ce.detach())
            loss = loss_ce + self.cfg.lambda_ppl * ppl_proxy
        else:
            loss = loss_ce

        return loss  # 这是我们要最小化的 loss

    # ------------------------------------------------------------------
    # I-GCG：基于梯度的 suffix token 离散更新（式 (3)）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _pick_candidate_tokens(
        self,
        grad_vec: torch.Tensor,
        embedding_weight: torch.Tensor,
        num_candidates: int,
        vocab_top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        给定某个位置的梯度 ∂L/∂e_i（形状 [D]），在 vocab 上找最可能
        让 loss 下降的若干 token（GCG 的经典做法是用 - grad · emb）。
        返回 shape: [num_candidates]
        """
        # grad_vec: [D]
        # embedding_weight: [V, D]
        # 对应 score_v = - <grad, emb_v>
        scores = -(embedding_weight @ grad_vec)  # [V]

        if vocab_top_k is not None and vocab_top_k < scores.size(0):
            top_scores, top_idx = torch.topk(scores, k=vocab_top_k)
            # 再从这 vocab_top_k 里挑 num_candidates
            k = min(num_candidates, vocab_top_k)
            sub_scores, sub_idx = torch.topk(top_scores, k=k)
            candidates = top_idx[sub_idx]
        else:
            k = min(num_candidates, scores.size(0))
            _, candidates = torch.topk(scores, k=k)
        return candidates  # [num_candidates]

    def gcg_step_text_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor],
        suffix_ids: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, float]:
        """
        在当前 suffix_ids 上执行一次 I-GCG 更新：
        1. 使用 inputs_embeds 方式拿到每个位置的梯度；
        2. 选出梯度范数最大的 B 个 suffix 位置；
        3. 对这些位置分别枚举 p 个候选 token，构造候选 suffix；
        4. 用 forward 评估所有候选，选最优组合（这里第一版先「贪心」：逐位置选最优）。
        返回：更新后的 suffix_ids, 当前 best_loss。
        """
        cfg = self.cfg
        device = self.device

        # -----------------------
        # 1) 构造 batch + suffix
        # -----------------------
        batch = self.build_batch_with_suffix(
            batch_texts=batch_texts,
            batch_images=batch_images,
            suffix_ids=suffix_ids,
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        pixel_values = batch.pixel_values.to(device) if batch.pixel_values is not None else None

        suffix_start, suffix_len = batch.suffix_range
        B, T_total = input_ids.shape

        # -----------------------
        # 2) 使用 inputs_embeds + autograd 拿到每个 token 的梯度
        # -----------------------
        # 禁用梯度缓存上的 no_grad
        self.model.train()
        embedding_layer: nn.Embedding = self.model.get_input_embeddings()

        # [B, T_total, D]
        inputs_embeds = embedding_layer(input_ids)
        inputs_embeds.requires_grad_(True)

        # y* 区间和 labels 与 compute_attack_loss 一致
        T_y = self.target_ids.size(0)
        target_start = T_total - T_y
        target_ids = input_ids[:, target_start:].clone()
        labels = torch.full_like(input_ids, fill_value=cfg.ignore_index)
        labels[:, target_start:] = target_ids

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss_ce: torch.Tensor = outputs.loss
        loss_ce.backward()

        # grad: [B, T_total, D]
        grad_embeds = inputs_embeds.grad.detach()  # type: ignore

        # 只看 suffix 区间的梯度，按式 (3) 取范数
        # grad_suffix: [B, suffix_len, D]
        grad_suffix = grad_embeds[:, suffix_start:suffix_start + suffix_len, :]
        # 对 batch 取平均： [suffix_len, D]
        grad_suffix_mean = grad_suffix.mean(dim=0)
        # 每个位置的梯度范数： [suffix_len]
        grad_norm = torch.norm(grad_suffix_mean, dim=-1)  # g_i

        # 选出梯度范数最大的 B 个位置
        B_pos = min(cfg.max_positions_update, suffix_len)
        _, top_pos_idx = torch.topk(grad_norm, k=B_pos)  # [B_pos]

        # ---------------------------------------------------
        # 3) 对这些位置分别枚举候选 token，做「贪心式」 GCG 更新
        #    （真正 I-GCG 会考虑联合组合，这里第一版先简化）
        # ---------------------------------------------------
        with torch.no_grad():
            best_suffix = suffix_ids.clone()
            # 当前 loss（重新算一次更干净）
            base_batch = self.build_batch_with_suffix(
                batch_texts=batch_texts,
                batch_images=batch_images,
                suffix_ids=best_suffix,
            )
            base_loss = self.compute_attack_loss(base_batch)

            embedding_weight = embedding_layer.weight.data  # [V, D]
            current_loss = base_loss.item()

            for pos in top_pos_idx.tolist():
                # 位置 pos 对应的梯度向量：[D]
                g_vec = grad_suffix_mean[pos]  # [D]
                candidate_tokens = self._pick_candidate_tokens(
                    grad_vec=g_vec,
                    embedding_weight=embedding_weight,
                    num_candidates=cfg.num_candidates,
                    vocab_top_k=cfg.vocab_top_k,
                )  # [p]

                # 遍历候选，逐个评估
                best_token_for_pos = best_suffix[pos].item()
                best_loss_for_pos = current_loss

                for tok in candidate_tokens.tolist():
                    if tok == best_suffix[pos].item():
                        continue
                    trial_suffix = best_suffix.clone()
                    trial_suffix[pos] = tok

                    trial_batch = self.build_batch_with_suffix(
                        batch_texts=batch_texts,
                        batch_images=batch_images,
                        suffix_ids=trial_suffix,
                    )
                    trial_loss = self.compute_attack_loss(trial_batch).item()

                    if trial_loss < best_loss_for_pos:
                        best_loss_for_pos = trial_loss
                        best_token_for_pos = tok

                # 如果有更好的，就正式更新该位置
                if best_token_for_pos != best_suffix[pos].item():
                    best_suffix[pos] = best_token_for_pos
                    current_loss = best_loss_for_pos

            return best_suffix, current_loss

    # ------------------------------------------------------------------
    # 外部主循环接口：执行若干步 GCG / I-GCG
    # ------------------------------------------------------------------
    def optimize_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor] = None,
        init_suffix: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, float]:
        """
        在给定的一批样本上，迭代优化一个共享 suffix。
        返回：最优 suffix_ids, 对应 loss。
        """
        if init_suffix is None:
            suffix_ids = self.init_suffix()
        else:
            suffix_ids = init_suffix.to(self.device)

        best_suffix = suffix_ids.clone()
        # 初始 loss
        batch = self.build_batch_with_suffix(
            batch_texts=batch_texts,
            batch_images=batch_images,
            suffix_ids=best_suffix,
        )
        best_loss = self.compute_attack_loss(batch).item()

        for step in range(self.cfg.max_steps):
            new_suffix, cur_loss = self.gcg_step_text_suffix(
                batch_texts=batch_texts,
                batch_images=batch_images,
                suffix_ids=best_suffix,
            )
            if cur_loss + 1e-6 < best_loss:
                best_loss = cur_loss
                best_suffix = new_suffix
            # 简单早停：若若干步没有提升，可以加 break（后面再细化）

        return best_suffix, best_loss
