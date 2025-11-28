# src/attacks/gcg_multimodal.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    # 单步更新时修改的最大位置数 B（I-GCG 中的 K）
    max_positions_update: int = 4
    # 每个位置的候选 token 数 p
    num_candidates: int = 16
    # 最大迭代步数
    max_steps: int = 50
    # 温度，用于采样/softmax 稍微平滑（目前未显式用，可留作扩展）
    temperature: float = 1.0
    # 是否在 vocab 上限制为「常用」token（比如排除控制符等）
    vocab_top_k: Optional[int] = None
    # PPL 正则项系数 λ_ppl（式 (2)）
    lambda_ppl: float = 0.0
    # 是否在 loss 中加入 PPL 惩罚（当前用 CE 的 exp 作为 proxy）
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
    文本侧 suffix 的 GCG / I-GCG。
    当前版本：
      - 只优化文本 suffix（φ），支持多模态输入（pixel_values 透传给模型）。
      - 图像 patch δ_ψ 留待后续扩展。
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
    # 小工具
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
        当前先用一个简单策略：从 vocab 中排除一小段特殊符号后随机采样。
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
    # 构造 batch + 计算攻击目标 loss（式 (1)）
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
           input = [user_text] + [suffix] + [y*]
        然后用 teacher forcing 对 y* 做 CE。
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

        input_ids_base = encoded["input_ids"].to(device)            # [B, T_base]
        attention_mask_base = encoded["attention_mask"].to(device)  # [B, T_base]

        B, T_base = input_ids_base.shape
        T_suffix = suffix_ids.size(0)
        T_y = self.target_ids.size(0)

        # 2) 构建 suffix 和 target 序列（共享）
        suffix_expanded = suffix_ids.unsqueeze(0).expand(B, -1)      # [B, T_suffix]
        target_expanded = self.target_ids.unsqueeze(0).expand(B, -1) # [B, T_y]

        # 3) 拼接： [user_text] + [suffix] + [y*]
        input_ids = torch.cat(
            [input_ids_base, suffix_expanded, target_expanded],
            dim=1,
        )  # [B, T_total]
        attn_suffix = torch.ones_like(suffix_expanded, device=device)
        attn_target = torch.ones_like(target_expanded, device=device)
        attention_mask = torch.cat(
            [attention_mask_base, attn_suffix, attn_target],
            dim=1,
        )

        # suffix 的起始位置、长度（注意：suffix 在中间）
        suffix_start = T_base            # 全 batch 一致
        suffix_len = T_suffix
        text_range = (0, T_base)

        pixel_values = batch_images.to(device) if batch_images is not None else None

        return BatchInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            suffix_range=(suffix_start, suffix_len),
            text_range=text_range,
        )

    def _apply_ppl_reg(self, loss_ce: torch.Tensor) -> torch.Tensor:
        """
        内部 PPL proxy：用 exp(CE) 近似 perplexity。
        这里只用于优化目标正则；真正的 PPL 评估用外部的小模型。
        """
        if not (self.cfg.use_ppl_reg and self.cfg.lambda_ppl > 0.0):
            return loss_ce

        # CE 是 batch 平均 NLL，直接 exp 作为 PPL proxy
        ppl_proxy = torch.exp(loss_ce.detach())  # 标量
        loss = loss_ce + self.cfg.lambda_ppl * ppl_proxy
        return loss

    def compute_attack_loss(
        self,
        batch: BatchInputs,
    ) -> torch.Tensor:
        """
        计算式 (1) 中的损失：
        - 用 teacher forcing，对 y* 部分的 token 做 cross-entropy。
        - 实际最小化的是 L = CE + λ·PPL_proxy（如果开启 reg）。
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

        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=False,
        )
        loss_ce: torch.Tensor = outputs.loss  # batch 上均值 CE

        loss = self._apply_ppl_reg(loss_ce)
        return loss

    # ------------------------------------------------------------------
    # I-GCG 中对 inputs_embeds 的前向 / 反向
    # ------------------------------------------------------------------
    def _forward_loss_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        给定 inputs_embeds（而不是 input_ids）计算 loss。
        用于在 I-GCG 中对 embedding 反向。
        """
        self.model.train()
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=False,
        )
        loss_ce: torch.Tensor = outputs.loss
        loss = self._apply_ppl_reg(loss_ce)
        return loss

    # ------------------------------------------------------------------
    # I-GCG：基于梯度的 suffix token 离散更新
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _pick_candidate_tokens(
        self,
        grad_vec: torch.Tensor,          # [D]
        embedding_weight: torch.Tensor,  # [V, D]
        num_candidates: int,
        vocab_top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        给定某个位置的梯度 ∂L/∂e_i（形状 [D]），在 vocab 上找最可能
        让 loss 下降的若干 token（GCG 的经典做法：score_v = - <grad, emb_v>）。
        返回 shape: [num_candidates]
        """
        # grad_vec: [D]
        # embedding_weight: [V, D]
        scores = -(embedding_weight @ grad_vec)  # [V]

        if vocab_top_k is not None and vocab_top_k < scores.size(0):
            # 先取前 vocab_top_k，再从里面取 num_candidates
            top_scores, top_idx = torch.topk(scores, k=vocab_top_k)
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
        suffix_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        I-GCG 风格的单步更新：
          - 先在当前 suffix 下做一次前向 + 反向，拿到 suffix 段的梯度；
          - 按梯度范数选出 K 个位置（K = cfg.max_positions_update）；
          - 对每个位置，按梯度方向在 vocab 中挑出 p 个候选 token（p = cfg.num_candidates）；
          - 对这 K 个位置做 greedy 更新：逐个位置尝试所有候选 token，保留能降低 loss 的替换。
        返回：
          new_suffix: 更新后的 suffix（如果没有提升则保持不变）
          base_loss: 该步结束时的 loss_total 值（CE + λ·PPL_proxy）
        """
        cfg = self.cfg
        device = self.device

        # 1) 构 batch
        batch = self.build_batch_with_suffix(
            batch_texts=batch_texts,
            batch_images=batch_images,
            suffix_ids=suffix_ids,
        )
        input_ids = batch.input_ids.to(device)           # [B, T_total]
        attention_mask = batch.attention_mask.to(device) # [B, T_total]
        pixel_values = batch.pixel_values.to(device) if batch.pixel_values is not None else None

        B, T_total = input_ids.size()
        T_suffix = suffix_ids.size(0)
        T_y = self.target_ids.size(0)

        # suffix 在中间： [text T_base][suffix T_suffix][target T_y]
        # 因此：
        T_base = T_total - T_suffix - T_y
        suffix_start = T_base
        target_start = T_total - T_y

        # 2) 为 I-GCG 构造 labels：只监督 y* 段
        target_ids = input_ids[:, target_start:].clone()  # [B, T_y]
        labels = torch.full_like(input_ids, fill_value=cfg.ignore_index)
        labels[:, target_start:] = target_ids

        # 3) 用 inputs_embeds 做一次带梯度的前向
        embedding_layer: nn.Embedding = self.model.get_input_embeddings()
        embedding_weight = embedding_layer.weight  # [V, D]

        raw_embeds = embedding_layer(input_ids)  # [B, T_total, D]
        inputs_embeds = raw_embeds.detach().clone()
        inputs_embeds.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        loss = self._forward_loss_with_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss.backward()

        grad_embeds = inputs_embeds.grad.detach()        # [B, T_total, D]
        grad_suffix = grad_embeds[:, suffix_start:suffix_start + T_suffix, :]  # [B, T_suffix, D]

        # 4) 按梯度范数选出 top-K 位置
        grad_norm = grad_suffix.pow(2).sum(dim=-1).mean(dim=0)  # [T_suffix]
        K = min(cfg.max_positions_update, T_suffix)
        _, topk_pos = torch.topk(grad_norm, k=K, largest=True)

        new_suffix = suffix_ids.clone()
        base_loss = loss.detach().item()

        # 5) 对每个被选中的位置做 greedy 更新
        for pos in topk_pos.tolist():
            abs_pos = suffix_start + pos  # 在整个序列中的绝对位置

            # 该位置的梯度方向（对 vocab scoring 用）
            g_pos = grad_suffix[:, pos, :].mean(dim=0)  # [D]

            # 按梯度方向挑出 P 个候选 token
            cand_token_ids = self._pick_candidate_tokens(
                grad_vec=g_pos,
                embedding_weight=embedding_weight,
                num_candidates=cfg.num_candidates,
                vocab_top_k=cfg.vocab_top_k,
            )

            best_pos_loss = base_loss
            best_token_id = int(new_suffix[pos].item())

            # 逐个候选 token 尝试替换，计算新的 loss_total
            for token_id in cand_token_ids.tolist():
                if int(token_id) == int(new_suffix[pos]):
                    continue

                trial_suffix = new_suffix.clone()
                trial_suffix[pos] = int(token_id)

                trial_batch = self.build_batch_with_suffix(
                    batch_texts=batch_texts,
                    batch_images=batch_images,
                    suffix_ids=trial_suffix,
                )
                trial_loss = self.compute_attack_loss(trial_batch).item()

                if trial_loss + 1e-6 < best_pos_loss:
                    best_pos_loss = trial_loss
                    best_token_id = int(token_id)

            # 如果找到更优的 token，就真正更新该位置，并更新 base_loss
            if best_pos_loss + 1e-6 < base_loss:
                new_suffix[pos] = best_token_id
                base_loss = best_pos_loss

        return new_suffix, base_loss

    # ------------------------------------------------------------------
    # 外部主循环接口：执行若干步 I-GCG（不强制打印日志）
    # ------------------------------------------------------------------
    def optimize_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor],
        init_suffix: Optional[torch.Tensor] = None,
        num_restarts: int = 3,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        多次随机重启的 I-GCG：
          - 每次从 init_suffix 或随机 suffix 出发，跑 cfg.max_steps 步；
          - 返回所有重启中最好的 suffix 和对应 loss_total。
        若 verbose=True，则会打印每次重启起点和每步 loss。
        """
        global_best_suffix: Optional[torch.Tensor] = None
        global_best_loss: float = float("inf")

        for r in range(num_restarts):
            if init_suffix is not None and r == 0:
                suffix_ids = init_suffix.clone()
            else:
                suffix_ids = self.init_suffix()

            # 初始 loss
            batch0 = self.build_batch_with_suffix(
                batch_texts=batch_texts,
                batch_images=batch_images,
                suffix_ids=suffix_ids,
            )
            best_loss = self.compute_attack_loss(batch0).item()
            best_suffix = suffix_ids.clone()

            if verbose:
                print(f"[opt] restart {r}, step -1, loss={best_loss:.4f}")

            # 逐步 I-GCG
            for step in range(self.cfg.max_steps):
                new_suffix, cur_loss = self.gcg_step_text_suffix(
                    batch_texts=batch_texts,
                    batch_images=batch_images,
                    suffix_ids=suffix_ids,
                )
                if cur_loss + 1e-6 < best_loss:
                    best_loss = cur_loss
                    best_suffix = new_suffix.clone()

                suffix_ids = new_suffix

                if verbose:
                    print(f"[opt] restart {r}, step {step}, loss={cur_loss:.4f}, best={best_loss:.4f}")

            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_suffix = best_suffix.clone()

        assert global_best_suffix is not None
        return global_best_suffix, global_best_loss
