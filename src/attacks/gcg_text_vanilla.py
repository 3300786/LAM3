# src/attacks/gcg_text_vanilla.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class VanillaGCGCfg:
    # 控制串初始文本（等价于原文中的 control_init）
    control_init: str = "! " * 20          # 约 20 个 "!" token
    max_steps: int = 50                    # GCG 最大迭代步数
    batch_size: int = 256                  # 每步候选控制串个数
    topk: int = 256                        # 每个位置考虑的 top-k token
    temp: float = 1.0                      # 采样温度（这里只用于退火）
    allow_non_ascii: bool = True          # 是否允许非 ASCII token（简化处理）
    anneal: bool = True                    # 是否使用退火接受策略
    anneal_from: int = 0
    # 退火接受概率 P(e, e')
    min_temperature: float = 1e-7

    # 安全拒绝的 test 前缀（用于后续可选的 jailbreak 检测，这里先不强依赖）
    test_prefixes: List[str] = None

    def __post_init__(self):
        if self.test_prefixes is None:
            self.test_prefixes = [
                "I'm sorry",
                "I apologize",
                "As an",
                "I'm just",
                "I'm an",
                "I cannot",
                "Hello!",
            ]


class VanillaGCGTextAttacker:
    """
    单模型、单 goal 的 GCG 文本攻击器（vanilla 版，贴近原始 GCG 思路）.

    prompt 结构（不考虑多轮对话）：
        [goal] + " " + [control] + " " + [target]
    其中：
        - goal:   用户危险 query（可以是多条 query 拼接后的长文本）
        - target: 我们期望模型输出的安全模板 y*
        - control: 待优化的后缀（初始为 "!!! ..."）

    目标：
        min_control  CE( logits(goal + control + target)[loss_slice], target_ids )
    """

    def __init__(
        self,
        model,
        tokenizer,
        goal: str,
        target: str,
        cfg: VanillaGCGCfg,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.goal = goal
        self.target = target
        self.cfg = cfg
        self.device = device

        # tokenizer 基本设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device).eval()

        # 预先 token 化 goal / target，不加 special tokens，方便我们手动拼接
        self.goal_ids = self.tokenizer(
            self.goal,
            add_special_tokens=False,
        )["input_ids"]
        self.target_ids = self.tokenizer(
            self.target,
            add_special_tokens=False,
        )["input_ids"]
        self.target_len = len(self.target_ids)

        # 控制串长度由 control_init 决定
        init_control_ids = self.tokenizer(
            self.cfg.control_init,
            add_special_tokens=False,
        )["input_ids"]
        if len(init_control_ids) == 0:
            raise ValueError("[VanillaGCG] control_init yields empty token sequence.")
        self.control_len = len(init_control_ids)

        print(f"[VanillaGCG] goal_len={len(self.goal_ids)}, "
              f"control_len={self.control_len}, target_len={self.target_len}")

        # 记录 slices（在完整 input_ids 中）
        # input_ids = goal_ids + control_ids + target_ids
        self.goal_slice = slice(0, len(self.goal_ids))
        self.control_slice = slice(self.goal_slice.stop,
                                   self.goal_slice.stop + self.control_len)
        self.target_slice = slice(self.control_slice.stop,
                                  self.control_slice.stop + self.target_len)
        # 典型 next-token CE：用 target 对应位置的前一个 logits
        self.loss_slice = slice(self.target_slice.start - 1,
                                self.target_slice.stop - 1)

        # CE loss
        self.criterion = nn.CrossEntropyLoss()

        # 以模型的 embedding 行数为准，而不是 tokenizer.vocab_size
        emb = self.model.get_input_embeddings()
        self.vocab_size = emb.weight.size(0)          # 真正的 embedding 词表大小
        self.emb_num = self.vocab_size                # 兼容后面使用

        # 初始控制串（token ids）
        self.control_ids_init = torch.tensor(
            init_control_ids, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------
    # 工具函数：embedding 相关
    # ------------------------------------------------------------------
    def _get_embedding_matrix(self) -> Tensor:
        emb = self.model.get_input_embeddings()
        w = emb.weight  # [V, D]
        # 这里 w.size(0) 应该等于 self.vocab_size
        assert w.size(0) == self.vocab_size, \
            f"Embedding rows {w.size(0)} != self.vocab_size {self.vocab_size}"
        return w

    def _get_embeddings(self, input_ids: Tensor) -> Tensor:
        # input_ids: [T]
        emb = self.model.get_input_embeddings()
        return emb(input_ids)  # [T, D]

    # ------------------------------------------------------------------
    # 构造完整 input_ids / attn_mask
    # ------------------------------------------------------------------
    def build_input_ids(self, control_ids: Tensor) -> Tensor:
        """
        control_ids: [C]
        返回 input_ids: [T_total] = goal + control + target
        """
        assert control_ids.dim() == 1
        if control_ids.size(0) != self.control_len:
            raise ValueError(
                f"[VanillaGCG] control_ids length mismatch: "
                f"expected {self.control_len}, got {control_ids.size(0)}"
            )

        ids = (
            torch.tensor(self.goal_ids, dtype=torch.long, device=self.device)
        )
        ids = torch.cat([ids, control_ids], dim=0)
        ids = torch.cat([
            ids,
            torch.tensor(self.target_ids, dtype=torch.long, device=self.device)
        ], dim=0)
        return ids  # [T_total]

    # ------------------------------------------------------------------
    # 目标损失：只用 CE(target)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_loss(self, control_ids: Tensor) -> float:
        """
        单次前向：给定 control_ids，返回对 target 的平均 CE loss.
        """
        input_ids = self.build_input_ids(control_ids).unsqueeze(0)  # [1, T]
        attention_mask = torch.ones_like(input_ids, device=self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = out.logits  # [1, T, V]

        # 目标是让 target_ids 预测正确
        targets = input_ids[0, self.target_slice]  # [T_target]
        logits_target = logits[0, self.loss_slice, :]  # [T_target, V]

        loss = self.criterion(logits_target, targets)
        return float(loss.item())

    # ------------------------------------------------------------------
    # GCG 的关键：对 control_slice 上的 token 做 one-hot embedding，并求梯度
    # ------------------------------------------------------------------
    def token_gradients(self, control_ids: Tensor) -> Tensor:
        """
        返回 control_slice 上每个 token 的梯度：
            grad: [control_len, vocab_size]
        """
        self.model.zero_grad(set_to_none=True)

        # 构造完整 input_ids
        input_ids = self.build_input_ids(control_ids)  # [T]
        input_ids = input_ids.to(self.device)

        # embedding 权重
        embed_weights = self._get_embedding_matrix()  # [V, D]
        T_control = self.control_len

        # control_slice 对应的原始 token id
        control_tok_ids = input_ids[self.control_slice]  # [C]

        # 确保所有 token id 都在 embedding 范围内
        assert control_tok_ids.max().item() < self.vocab_size, (
            f"control token id {control_tok_ids.max().item()} "
            f">= embedding size {self.vocab_size}"
        )

        # 构造 one-hot，表示控制位置上 token 的可微参数
        one_hot = torch.zeros(
            T_control,
            self.vocab_size,            # 用 embedding 行数
            device=self.device,
            dtype=embed_weights.dtype,
        )
        one_hot.scatter_(
            1,
            control_tok_ids.unsqueeze(1),
            torch.ones(T_control, 1, device=self.device, dtype=embed_weights.dtype),
        )
        one_hot.requires_grad_(True)

        # 把控制区域用 one-hot@W 替换，其它位置用普通 embedding
        with torch.no_grad():
            embeds_full = self._get_embeddings(input_ids)  # [T, D]

        control_embeds = one_hot @ embed_weights  # [C, D]

        embeds = torch.cat(
            [
                embeds_full[: self.control_slice.start, :],
                control_embeds,
                embeds_full[self.control_slice.stop :, :],
            ],
            dim=0,
        )  # [T, D]

        embeds = embeds.unsqueeze(0)  # [1, T, D]
        attention_mask = torch.ones(
            1, embeds.size(1), device=self.device, dtype=torch.long
        )

        # 前向 + CE
        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        logits = out.logits  # [1, T, V]

        targets = input_ids[self.target_slice]  # [T_target]
        logits_target = logits[0, self.loss_slice, :]  # [T_target, V]

        loss = self.criterion(logits_target, targets)
        loss.backward()

        grad = one_hot.grad.detach().clone()  # [C, V]

        # 清理
        self.model.zero_grad(set_to_none=True)
        del embeds, out, logits, loss, one_hot, control_embeds, embeds_full
        torch.cuda.empty_cache()

        # 归一化（和原文类似）
        grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-12)
        return grad  # [C, V]

    # ------------------------------------------------------------------
    # 采样候选控制串（对应 GCGPromptManager.sample_control）
    # ------------------------------------------------------------------
    def sample_controls(
        self,
        current_control: torch.LongTensor,
        grad: torch.Tensor,
        batch_size: Optional[int] = None,
        topk: Optional[int] = None,
        allow_non_ascii: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        """
        给定当前 control token 序列和梯度，按照 GCG 策略采样一批候选 control。

        Args:
            current_control: [C] 当前 control token ids
            grad:            [C, V] 每个位置对 vocab 的梯度
            batch_size:      采样候选数量 (默认取 cfg.population / cfg.batch_size)
            topk:            每个位置从梯度最优的前 k 个 token 中采样
            allow_non_ascii: 是否允许非 ASCII token

        Returns:
            cand_controls:   [B, C]，每一行是一个候选控制串
        """
        device = self.device
        C, V = grad.shape  # C = control_len, V = vocab_size

        # 1) batch_size / topk 默认值
        if batch_size is None:
            batch_size = getattr(self.cfg, "population", None) or getattr(
                self.cfg, "batch_size", 16
            )
        if topk is None:
            topk = getattr(self.cfg, "topk", V)
        topk = min(topk, V)

        # 2) 可选：屏蔽不允许的 token
        grad_for_topk = grad.clone()
        if not allow_non_ascii and getattr(self, "disallowed_token_ids", None) is not None:
            if self.disallowed_token_ids.numel() > 0:
                grad_for_topk[:, self.disallowed_token_ids.to(device)] = float("inf")

        # 3) 取每个位置的 top-k 候选 token id
        #    top_indices: [C, topk]
        top_indices = (-grad_for_topk).topk(topk, dim=1).indices  # [C, topk]

        # 4) 基础控制串：复制 batch_size 份
        #    base: [B, C]
        base = current_control.to(device).unsqueeze(0).expand(batch_size, -1).clone()

        # 5) 为每个 candidate 选择要修改的一个位置
        #    positions: [B]，范围在 [0, C-1]
        positions = torch.linspace(0, C - 1, steps=batch_size, device=device).long()

        # 6) 在该位置的 top-k 中随机选一个 token
        #    rand_k: [B] in [0, topk-1]
        rand_k = torch.randint(0, topk, (batch_size,), device=device)

        # 7) 从 top_indices 里取出新 token：new_vals: [B]
        new_vals = top_indices[positions, rand_k]  # [B]

        # 8) 写回 base，对应 (行, 列) = (0..B-1, positions[b])
        rows = torch.arange(batch_size, device=device)
        base[rows, positions] = new_vals

        return base  # [B, C]


    # ------------------------------------------------------------------
    # 退火接受函数（贴近 MultiPromptAttack.run 中的 P(e, e')）
    # ------------------------------------------------------------------
    def _accept(
        self,
        prev_loss: float,
        new_loss: float,
        step: int,
        n_steps: int,
    ) -> bool:
        if not self.cfg.anneal:
            return new_loss < prev_loss

        T = max(
            1.0 - float(step + 1 + self.cfg.anneal_from) / (n_steps + self.cfg.anneal_from),
            self.cfg.min_temperature,
        )
        if new_loss < prev_loss:
            return True
        # Metropolis-Hastings style
        prob = math.exp(-(new_loss - prev_loss) / T)
        return prob >= random.random()

    # ------------------------------------------------------------------
    # 主循环：vanilla GCG
    # ------------------------------------------------------------------
    def run(self) -> Tuple[Tensor, float]:
        """
        返回：
            best_control_ids: [C]
            best_loss: float
        """
        device = self.device

        # 初始控制串：用 control_init 的 token 化结果
        current_control = self.control_ids_init.clone().to(device)
        current_loss = self.compute_loss(current_control)

        best_control = current_control.clone()
        best_loss = current_loss

        print(f"[VanillaGCG] init loss={current_loss:.4f}")

        for step in range(self.cfg.max_steps):
            # 1) 计算 control 上的梯度
            grad = self.token_gradients(current_control)  # [C, V]

            # 2) 采样 candidates
            cand_controls = self.sample_controls(
                current_control=current_control,
                grad=grad,
                batch_size=self.cfg.batch_size,
                topk=self.cfg.topk,
            )  # [B, C]

            # 3) 评估每个 candidate 的 loss
            cand_losses: List[float] = []
            with torch.no_grad():
                for i in range(cand_controls.size(0)):
                    loss_i = self.compute_loss(cand_controls[i])
                    cand_losses.append(loss_i)

            # 4) 取最优 candidate
            min_idx = int(torch.tensor(cand_losses).argmin().item())
            cand_best_loss = cand_losses[min_idx]
            cand_best_control = cand_controls[min_idx]

            improved = cand_best_loss < best_loss

            # 5) 退火接受：是否把 current_control 更新为 cand_best
            if self._accept(current_loss, cand_best_loss, step, self.cfg.max_steps):
                current_control = cand_best_control.clone()
                current_loss = cand_best_loss

            # 6) 更新全局最优
            if improved:
                best_loss = cand_best_loss
                best_control = cand_best_control.clone()

            print(
                f"[VanillaGCG][step {step}] "
                f"cand_best_loss={cand_best_loss:.4f}, "
                f"current_loss={current_loss:.4f}, "
                f"best_loss={best_loss:.4f}"
            )

        return best_control, best_loss

    # ------------------------------------------------------------------
    # 将 control_ids decode 为文本
    # ------------------------------------------------------------------
    def decode_control(self, control_ids: Tensor) -> str:
        return self.tokenizer.decode(
            control_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    # 在 VanillaGCGTextAttacker 类里新增一个方法，方便外部调用
    def get_full_prompt(self, control_ids: Optional[Tensor] = None) -> str:
        """返回 goal + control + target 的完整文本（用于最终生成验证）"""
        if control_ids is None:
            control_ids = self.control_ids_init
        control_text = self.decode_control(control_ids)
        return f"{self.goal} {control_text}"