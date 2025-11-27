# src/attacks/gcg_multimodal.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.utils.ppl_utils import compute_ppl


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
    # 温度（当前未使用，预留）
    temperature: float = 1.0
    # 是否在 vocab 上限制为「常用」token（比如排除控制符等）
    vocab_top_k: Optional[int] = None
    # PPL 正则项系数 λ_ppl
    lambda_ppl: float = 0.0
    # 是否在 loss 中加入 PPL 惩罚
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
    文本侧 suffix 的 GCG / I-GCG，
    支持多模态输入（pixel_values 会被传入模型），
    图像 patch δ_ψ 暂不实现。
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: GCGBasicCfg,
        device: Optional[torch.device] = None,
        # 由外部脚本传入的 PPL 模型（基于 src/utils/ppl_utils.py）
        ppl_model: Optional[torch.nn.Module] = None,
        ppl_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        ppl_device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # 预先构建 y* 模板的 token 序列
        self.target_ids = self._build_target_ids(
            cfg.target_template,
            max_len=cfg.target_max_len,
        )  # [T_y]

        # PPL 相关：由外部传入，避免内部再去读 models.yaml
        self.ppl_model = ppl_model
        self.ppl_tokenizer = ppl_tokenizer
        self.ppl_device = ppl_device if ppl_device is not None else self.device

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _build_target_ids(self, text: str, max_len: int) -> torch.LongTensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
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
        # 如果 suffix_len 为 0，直接返回空 tensor（用于 baseline）
        if self.cfg.suffix_len <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        vocab_size = self.tokenizer.vocab_size
        # 简单起见，排除低 id 的特殊符号区域
        start_id = 10
        end_id = vocab_size

        suffix = torch.randint(
            low=start_id,
            high=end_id,
            size=(self.cfg.suffix_len,),
            device=self.device,
        )
        return suffix

    # ------------------------------------------------------------------
    # 构造 BatchInputs
    # ------------------------------------------------------------------
    def build_batch_with_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor],
        suffix_ids: torch.LongTensor,
    ) -> BatchInputs:
        """
        将一批 (text, image) 样本与共享 suffix 拼接，返回 BatchInputs.

        这里暂时简化为：
            input = [user_text] + [suffix] + [y*]
        并用 teacher forcing 让模型预测 y*。
        """
        B = len(batch_texts)
        device = self.device

        # 1) 对每个文本做 tokenize
        encoded = self.tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids_base = encoded["input_ids"].to(device)             # [B, T_base]
        attention_mask_base = encoded["attention_mask"].to(device)   # [B, T_base]

        B, T_base = input_ids_base.shape
        T_suffix = suffix_ids.size(0)
        T_y = self.target_ids.size(0)

        # 2) 构建 suffix 和 target 序列（共享）
        if T_suffix > 0:
            suffix_expanded = suffix_ids.unsqueeze(0).expand(B, -1)       # [B, T_suffix]
            attn_suffix = torch.ones_like(suffix_expanded, device=device)
        else:
            suffix_expanded = torch.empty(B, 0, dtype=torch.long, device=device)
            attn_suffix = torch.empty(B, 0, dtype=torch.long, device=device)

        target_expanded = self.target_ids.unsqueeze(0).expand(B, -1)      # [B, T_y]
        attn_target = torch.ones_like(target_expanded, device=device)

        # 3) 拼接： [user_text] + [suffix] + [y*]
        input_ids = torch.cat(
            [input_ids_base, suffix_expanded, target_expanded],
            dim=1,
        )  # [B, T_total]

        attention_mask = torch.cat(
            [attention_mask_base, attn_suffix, attn_target],
            dim=1,
        )

        # suffix 的起始位置、长度
        suffix_start = T_base
        suffix_len = T_suffix

        # text_range 记录 user_text 的区间
        text_range = (0, T_base)

        pixel_values = batch_images.to(device) if batch_images is not None else None

        return BatchInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            suffix_range=(suffix_start, suffix_len),
            text_range=text_range,
        )

    # ------------------------------------------------------------------
    # PPL 计算：与 utils/ppl_utils.compute_ppl 对齐
    # ------------------------------------------------------------------
    def _compute_prefix_ppl(
        self,
        input_ids: torch.Tensor,
        target_start: int,
    ) -> float:
        """
        仅在启用 PPL 正则时调用。

        prefix = [user_text + suffix]（不含 y*）。
        对每个样本 prefix 单独调用 compute_ppl，再取平均。
        """
        if (
            not self.cfg.use_ppl_reg
            or self.cfg.lambda_ppl <= 0.0
            or self.ppl_model is None
            or self.ppl_tokenizer is None
        ):
            return 0.0

        device = self.ppl_device
        prefix_ids = input_ids[:, :target_start]  # [B, T_prefix]

        texts: List[str] = []
        for row in prefix_ids:
            row_ids = row.tolist()
            txt = self.tokenizer.decode(
                row_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if txt:
                texts.append(txt)

        if not texts:
            return 0.0

        sum_ppl = 0.0
        cnt = 0
        for t in texts:
            ppl_val = compute_ppl(
                model=self.ppl_model,
                tokenizer=self.ppl_tokenizer,
                text=t,
                device=device,
                max_length=512,
            )
            sum_ppl += float(ppl_val)
            cnt += 1

        if cnt == 0:
            return 0.0
        return sum_ppl / cnt

    # ------------------------------------------------------------------
    # 损失计算：用于 compute_attack_loss / I-GCG 内部前向
    # ------------------------------------------------------------------
    def compute_attack_loss(
        self,
        batch: BatchInputs,
    ) -> torch.Tensor:
        """
        计算总损失：

            L_total = CE(y* | x ⊕ s_φ) + λ_ppl * PPL(x ⊕ s_φ)

        其中 CE 通过 teacher forcing，只在 y* 部分有监督。
        PPL 正则只对 prefix = [text + suffix] 计算，用于鼓励 suffix 更可读。
        """
        device = self.device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        pixel_values = (
            batch.pixel_values.to(device)
            if batch.pixel_values is not None
            else None
        )

        B, T_total = input_ids.shape
        T_y = self.target_ids.size(0)

        # y* 的位置在最后 T_y 个 token
        target_start = T_total - T_y
        target_ids = input_ids[:, target_start:].clone()  # [B, T_y]

        # 构造 labels：仅在 y* 部分有监督，其它位置为 ignore_index
        labels = torch.full_like(input_ids, fill_value=self.cfg.ignore_index)
        labels[:, target_start:] = target_ids

        self.model.train()

        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values

        outputs = self.model(**kwargs)
        loss_ce: torch.Tensor = outputs.loss  # batch mean

        loss = loss_ce

        # 可选：加入 PPL 正则
        if self.cfg.use_ppl_reg and self.cfg.lambda_ppl > 0.0:
            ppl_value = self._compute_prefix_ppl(
                input_ids=input_ids,
                target_start=target_start,
            )
            loss = loss_ce + self.cfg.lambda_ppl * float(ppl_value)

        return loss

    def _forward_loss(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        与 compute_attack_loss 类似，但使用 inputs_embeds 作为输入，
        用于 I-GCG 中获取 embedding 梯度。

        注意：这里为了简化实现，只使用 CE，不包含 PPL 项；
        PPL 只参与 greedy candidate 打分（compute_attack_loss）而不参与梯度。
        """
        device = self.device
        inputs_embeds = inputs_embeds.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        pixel_values = (
            pixel_values.to(device) if pixel_values is not None else None
        )

        self.model.train()

        kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values

        outputs = self.model(**kwargs)
        loss_ce: torch.Tensor = outputs.loss
        return loss_ce

    # ------------------------------------------------------------------
    # I-GCG：基于梯度的 suffix token 离散更新（多 token 替换）
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
        让 loss 下降的若干 token（GCG 的经典做法是用 - grad · emb）。
        返回 shape: [num_candidates]
        """
        scores = -(embedding_weight @ grad_vec)  # [V]

        if vocab_top_k is not None and vocab_top_k < scores.size(0):
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
        I-GCG 风格单步更新：
        - 根据梯度选出 K 个位置（K = cfg.max_positions_update）
        - 对每个位置按梯度方向选出 P 个候选 token（P = cfg.num_candidates）
        - 对这 K 个位置做 greedy 更新：依次尝试替换，保留能降低 (CE + λ·PPL) 的选择
        """
        cfg = self.cfg
        device = self.device

        # 1) 构 batch
        batch = self.build_batch_with_suffix(
            batch_texts=batch_texts,
            batch_images=batch_images,
            suffix_ids=suffix_ids,
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        pixel_values = (
            batch.pixel_values.to(device)
            if batch.pixel_values is not None
            else None
        )

        # 从 batch 中拿 suffix 区间
        if batch.suffix_range is not None:
            suffix_start, T_suffix = batch.suffix_range
        else:
            # 没有 suffix（suffix_len=0）时直接返回
            loss = self.compute_attack_loss(batch)
            return suffix_ids.clone(), loss.item()

        if T_suffix == 0:
            loss = self.compute_attack_loss(batch)
            return suffix_ids.clone(), loss.item()

        T_total = input_ids.size(1)

        # --- 用 inputs_embeds 做 backward，拿梯度 ---
        self.model.train()
        embedding_layer: nn.Embedding = self.model.get_input_embeddings()
        embedding_weight = embedding_layer.weight.detach()  # [V, D]

        raw_embeds = embedding_layer(input_ids)            # [B, T_total, D]
        inputs_embeds = raw_embeds.detach().clone()
        inputs_embeds.requires_grad_(True)

        # 构造 labels：只监督 y* 部分
        T_y = self.target_ids.size(0)
        target_start = T_total - T_y
        target_ids = input_ids[:, target_start:].clone()
        labels = torch.full_like(input_ids, fill_value=cfg.ignore_index)
        labels[:, target_start:] = target_ids

        # 清梯度
        self.model.zero_grad(set_to_none=True)
        loss_ce = self._forward_loss(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss_ce.backward()

        grad_embeds = inputs_embeds.grad.detach()          # [B, T_total, D]
        grad_suffix = grad_embeds[:, suffix_start:suffix_start + T_suffix, :]  # [B, T_suffix, D]

        # 2) 按梯度范数，选出 top-K 位置
        grad_norm = grad_suffix.pow(2).sum(dim=-1).mean(dim=0)  # [T_suffix]
        K = min(cfg.max_positions_update, T_suffix)
        _, topk_pos = torch.topk(grad_norm, k=K, largest=True)

        # 先算当前 (CE + λ·PPL) 作为 base_loss
        base_loss = self.compute_attack_loss(batch).detach().item()
        new_suffix = suffix_ids.clone()

        # 3) I-GCG：对这 K 个位置做 greedy 更新
        for pos in topk_pos.tolist():
            # 该位置的梯度方向（对 vocab scoring 用），先对 batch 求平均
            g_pos = grad_suffix[:, pos, :].mean(dim=0)  # [D]

            # 按梯度方向挑 P 个候选 token
            cand_token_ids = self._pick_candidate_tokens(
                grad_vec=g_pos,
                embedding_weight=embedding_weight,
                num_candidates=cfg.num_candidates,
                vocab_top_k=cfg.vocab_top_k,
            )

            best_pos_loss = base_loss
            best_token_id = int(new_suffix[pos].item())

            # 逐个候选 token 尝试替换
            for token_id in cand_token_ids:
                token_id = int(token_id)
                if token_id == int(new_suffix[pos]):
                    continue

                trial_suffix = new_suffix.clone()
                trial_suffix[pos] = token_id

                trial_batch = self.build_batch_with_suffix(
                    batch_texts=batch_texts,
                    batch_images=batch_images,
                    suffix_ids=trial_suffix,
                )
                trial_loss = self.compute_attack_loss(trial_batch).item()

                if trial_loss + 1e-6 < best_pos_loss:
                    best_pos_loss = trial_loss
                    best_token_id = token_id

            # 如果找到更优的 token，就真正更新该位置，并更新 base_loss
            if best_pos_loss + 1e-6 < base_loss:
                new_suffix[pos] = best_token_id
                base_loss = best_pos_loss

        return new_suffix, base_loss

    # ------------------------------------------------------------------
    # 外部主循环接口：执行若干步 I-GCG（带重启）
    # ------------------------------------------------------------------
    def optimize_suffix(
        self,
        batch_texts: List[str],
        batch_images: Optional[torch.Tensor],
        init_suffix: Optional[torch.Tensor] = None,
        num_restarts: int = 1,
    ) -> Tuple[torch.Tensor, float]:
        """
        多次随机重启 + I-GCG，返回全局最优 suffix 及其 (CE+λ·PPL) loss。
        """
        global_best_suffix: Optional[torch.Tensor] = None
        global_best_loss: float = float("inf")

        for r in range(num_restarts):
            if init_suffix is not None and r == 0:
                suffix_ids = init_suffix.clone()
            else:
                suffix_ids = self.init_suffix()

            best_suffix = suffix_ids.clone()
            best_loss = float("inf")

            for _ in range(self.cfg.max_steps):
                new_suffix, cur_loss = self.gcg_step_text_suffix(
                    batch_texts=batch_texts,
                    batch_images=batch_images,
                    suffix_ids=suffix_ids,
                )

                if cur_loss + 1e-6 < best_loss:
                    best_loss = cur_loss
                    best_suffix = new_suffix.clone()

                suffix_ids = new_suffix

            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_suffix = best_suffix.clone()

        assert global_best_suffix is not None
        return global_best_suffix, global_best_loss
