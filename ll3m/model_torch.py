"""
PyTorch GPT (v2): RMSNorm, RoPE (optional partial), SwiGLU, GQA, QK-Norm, no Linear biases.
Optional Gemma-style final logit softcap. Train may add PaLM z-loss externally.
Training may use full DRAM/GPU; L3 product claim is C inference WS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ll3m.config import GPTConfig
from ll3m.qat import quantize_wte_weight


@dataclass
class TrainBudget:
    n_params: int
    bytes_params: int
    bytes_grads: int
    bytes_opt: int
    bytes_acts: int
    total: int


@dataclass
class InferBudget:
    n_params: int
    weight_bytes: int
    activation_bytes: int
    total: int


def estimate_budget(
    cfg: GPTConfig,
    batch: int,
    optimizer: str = "lion",
    dtype_bytes: int = 4,
) -> TrainBudget:
    n = cfg.n_params()
    b = cfg.bytes_for_training(batch, optimizer=optimizer, checkpoint=True, dtype_bytes=dtype_bytes)
    mult = 3 if optimizer == "lion" else 4
    return TrainBudget(
        n_params=n,
        bytes_params=n * dtype_bytes,
        bytes_grads=n * dtype_bytes,
        bytes_opt=(mult - 2) * n * dtype_bytes,
        bytes_acts=b["activations"],
        total=b["total"],
    )


def estimate_infer_ws(
    cfg: GPTConfig,
    token_cap: int | None = None,
    page_slack: int = 8192,
    quant: str = "ternary",
    embed_quant: str | None = None,
    scale_bytes: int = 4,
) -> InferBudget:
    """C-infer WS estimate (norms + KV + quantized Linears / embeds).

    Product default is ternary Linears + int8 embeds (L3T2 serve pack).
    Pass quant="int8" or "f32" only for explicit ablations.

    quant: Linear packing "ternary" | "binary" | "int8" | "f32" (default: ternary).
    embed_quant: "f32" | "int8" | "ternary" | "binary"; default int8 when
                 quant is ternary/binary (densest practical product path), else f32.
    scale_bytes: bytes per Linear row-scale (4=f32, 2=f16 estimate).

    Footgun: omitting quant used to default to int8 and overstate product WS
    relative to the ternary freeze (~30.22 MiB). Always pass quant explicitly
    in research notes when comparing grids.

    Note: quant="binary" is a capacity-search / research estimate (1-bit pack).
    Product C serve is ternary (CLM_QUANT=2); binary has no ship pack yet.
    """
    V, T, L, H, D = cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd
    nkv = cfg.kv_head
    hd = cfg.head_dim
    mid = cfg.intermediate
    if token_cap is None:
        token_cap = T + 256
    if embed_quant is None:
        embed_quant = "int8" if quant in ("ternary", "binary") else "f32"
    if scale_bytes not in (2, 4):
        raise ValueError("scale_bytes must be 2 (f16) or 4 (f32)")
    if quant not in ("ternary", "binary", "int8", "f32"):
        raise ValueError(f"unknown linear quant: {quant}")
    if embed_quant not in ("f32", "int8", "ternary", "binary"):
        raise ValueError(f"unknown embed quant: {embed_quant}")

    # f32: norms (+ peri/ls) + final rms; embeds handled separately
    nf = L * (D + D + hd + hd) + D
    if cfg.peri_ln:
        nf += L * 2 * D
    if cfg.layer_scale_init > 0.0:
        nf += L * 2 * D
    n_scale_rows = L * (D + nkv * hd + nkv * hd + D + mid + mid + D)
    ni = L * (
        D * D
        + nkv * hd * D
        + nkv * hd * D
        + D * D
        + mid * D
        + mid * D
        + D * mid
    )
    floats = (
        T * D * 3
        + T * (D + 2 * nkv * hd)
        + H * T * T
        + T * mid
        + V
        + 2 * L * T * nkv * hd
    )
    n_embed = V * D

    if quant == "ternary":
        # 2-bit pack + ~5% pad to multiple-of-4 columns
        linear_bytes = int(ni * 0.25 * 1.05)
    elif quant == "binary":
        # 1-bit pack + ~5% pad to multiple-of-8 columns (capacity-search estimate)
        linear_bytes = int(ni * 0.125 * 1.05)
    elif quant == "f32":
        linear_bytes = ni * 4
    else:
        linear_bytes = ni  # int8

    # Tied LM head = wte. Compact path quantizes the embedding table.
    if embed_quant == "ternary":
        embed_bytes = int(n_embed * 0.25 * 1.05) + V * scale_bytes  # + per-row scale
    elif embed_quant == "binary":
        embed_bytes = int(n_embed * 0.125 * 1.05) + V * scale_bytes
    elif embed_quant == "int8":
        embed_bytes = n_embed + V * scale_bytes  # int8 codes + f32/f16 scale
    else:
        embed_bytes = n_embed * 4

    scale_payload = n_scale_rows * scale_bytes
    weight_bytes = (
        nf * 4
        + scale_payload
        + linear_bytes
        + embed_bytes
        + max(V, 4)
    )
    act_bytes = floats * 4 + token_cap * 4
    total = weight_bytes + act_bytes + page_slack
    return InferBudget(
        n_params=cfg.n_params(),
        weight_bytes=weight_bytes,
        activation_bytes=act_bytes,
        total=total,
    )


def choose_config_infer(
    vocab_size: int,
    infer_budget_bytes: int,
    block_size: int = 256,
) -> tuple[GPTConfig, InferBudget]:
    """Largest config whose C-infer WS fits the budget."""
    best: GPTConfig | None = None
    best_inf: InferBudget | None = None
    best_score = (-1.0, -1, -1)
    for n_layer in range(4, 25):
        for n_embd in (128, 160, 192, 224, 256, 288, 320, 384, 448, 512):
            for n_head in (4, 6, 8, 12, 16):
                if n_embd % n_head != 0:
                    continue
                hd = n_embd // n_head
                if hd not in (16, 24, 32, 40, 48, 64):
                    continue
                for n_kv in (1, 2, 4, n_head):
                    if n_head % n_kv != 0:
                        continue
                    cfg = GPTConfig(
                        vocab_size=vocab_size,
                        block_size=block_size,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_embd=n_embd,
                        n_kv_head=n_kv,
                    )
                    inf = estimate_infer_ws(cfg)
                    if inf.total > infer_budget_bytes:
                        continue
                    score = (float(cfg.n_params()), cfg.n_layer, cfg.n_embd)
                    if score > best_score:
                        best, best_inf, best_score = cfg, inf, score
    if best is None or best_inf is None:
        best = GPTConfig(
            vocab_size=vocab_size,
            block_size=min(128, block_size),
            n_layer=4,
            n_head=4,
            n_embd=128,
            n_kv_head=2,
        )
        best_inf = estimate_infer_ws(best)
    return best, best_inf


# ---- v2 building blocks ----


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        orig = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(orig)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dims: int | None = None,
):
    """Apply RoPE to the first rope_dims of each head (partial rotary if < head_dim).

    q,k: (B, H, T, hd); cos/sin: (T, rope_dims).
    """
    hd = q.size(-1)
    rd = hd if rope_dims is None else int(rope_dims)
    if rd <= 0:
        return q, k
    if rd >= hd:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_pos: int,
        theta: float = 10000.0,
        rope_dims: int | None = None,
    ):
        super().__init__()
        rd = head_dim if rope_dims is None else int(rope_dims)
        if rd % 2 != 0:
            raise ValueError("rope_dims must be even")
        self.rope_dims = rd
        inv = 1.0 / (theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_pos = max_pos
        self._build_cache(max_pos)

    def _build_cache(self, n: int) -> None:
        t = torch.arange(n, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, t: int, device: torch.device, *, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions [position_offset, position_offset + t)."""
        if position_offset < 0:
            raise ValueError("position_offset must be >= 0")
        end = position_offset + t
        if end > self.cos_cached.size(0):
            self._build_cache(max(end, self.max_pos))
        return (
            self.cos_cached[position_offset:end].to(device),
            self.sin_cached[position_offset:end].to(device),
        )



class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, rope: RotaryEmbedding):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.kv_head
        self.head_dim = cfg.head_dim
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.rope = rope
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Self-attention with optional KV cache for decode.

        past_kv: (k, v) each (B, n_kv_head, past_len, head_dim), pre-GQA-expand.
        When use_cache=True, returns (out, present_kv).
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        past_len = 0 if past_kv is None else int(past_kv[0].size(2))
        cos, sin = self.rope(T, x.device, position_offset=past_len)
        q, k = apply_rope(q, k, cos, sin, rope_dims=self.rope.rope_dims)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        present = (k, v) if use_cache else None
        # Expand KV for GQA only for the attention matmul (cache stays compact).
        k_att, v_att = k, v
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k_att = k.repeat_interleave(rep, dim=1)
            v_att = v.repeat_interleave(rep, dim=1)
        # Prefill (no past): square causal. Decode (T queries, longer K): full attend.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q,
            k_att,
            v_att,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(y))
        if use_cache:
            assert present is not None
            return out, present
        return out



class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        mid = cfg.intermediate
        self.gate = nn.Linear(cfg.n_embd, mid, bias=False)
        self.up = nn.Linear(cfg.n_embd, mid, bias=False)
        self.down = nn.Linear(mid, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))



class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, rope: RotaryEmbedding):
        super().__init__()
        self.rms_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, rope)
        self.rms_2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)
        # Gemma2 / OLMo2 peri-LN: normalize residual stream after each branch
        self.peri_ln = cfg.peri_ln
        self.post_attn = RMSNorm(cfg.n_embd) if cfg.peri_ln else None
        self.post_mlp = RMSNorm(cfg.n_embd) if cfg.peri_ln else None
        # CaiT-style LayerScale on residual branches (helps deep/thin stacks)
        if cfg.layer_scale_init > 0.0:
            self.ls_1 = nn.Parameter(torch.full((cfg.n_embd,), cfg.layer_scale_init))
            self.ls_2 = nn.Parameter(torch.full((cfg.n_embd,), cfg.layer_scale_init))
        else:
            self.ls_1 = None
            self.ls_2 = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            a, present = self.attn(self.rms_1(x), past_kv=past_kv, use_cache=True)
        else:
            a = self.attn(self.rms_1(x), past_kv=past_kv, use_cache=False)
            present = None
        if self.ls_1 is not None:
            a = a * self.ls_1
        x = x + a
        if self.post_attn is not None:
            x = self.post_attn(x)
        m = self.mlp(self.rms_2(x))
        if self.ls_2 is not None:
            m = m * self.ls_2
        x = x + m
        if self.post_mlp is not None:
            x = self.post_mlp(x)
        if use_cache:
            assert present is not None
            return x, present
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.rope = RotaryEmbedding(
            cfg.head_dim,
            cfg.block_size,
            cfg.rope_theta,
            rope_dims=cfg.rope_dims,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg, self.rope) for _ in range(cfg.n_layer)]),
                ln_f=RMSNorm(cfg.n_embd),
            )
        )
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        z_loss_coeff: float = 0.0,
        mtp_weight: float = 0.0,
        mtp_depth: int = 2,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            list[tuple[torch.Tensor, torch.Tensor]],
        ]
    ):
        """Train/eval forward. Optional past_key_values for incremental decode.

        When use_cache=True, returns (logits, loss, presents) where presents is a
        list of (k, v) per layer, each (B, n_kv_head, seq, head_dim).
        Train path: leave use_cache=False (default); API unchanged for loss.
        """
        B, T = idx.shape
        # Optional embed QAT (tied wte for lookup + LM head)
        embed_qat = getattr(self, "_ll3m_embed_qat", "none") or "none"
        wte_w = quantize_wte_weight(self.transformer.wte.weight, embed_qat)
        x = self.transformer.drop(F.embedding(idx, wte_w))
        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for li, block in enumerate(self.transformer.h):
            past = None if past_key_values is None else past_key_values[li]
            if use_cache:
                x, present = block(x, past_kv=past, use_cache=True)
                assert presents is not None
                presents.append(present)
            else:
                x = block(x, past_kv=past, use_cache=False)
        x = self.transformer.ln_f(x)
        logits = F.linear(x, wte_w)
        cap = float(self.config.logit_softcap)
        if cap > 0.0:
            # Gemma-style final softcap: cap * tanh(logits / cap)
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            flat = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(flat, targets.view(-1))
            if z_loss_coeff > 0.0:
                # PaLM z-loss: stabilize log-sum-exp of logits (train-only regularizer)
                log_z = torch.logsumexp(flat.float(), dim=-1)
                loss = loss + z_loss_coeff * (log_z**2).mean()
            # Shared-head multi-token prediction (Gloeckle/DeepSeek-style signal,
            # no extra modules; discarded cost at serve: zero).
            # At position t, also supervise token t+k for k=2..mtp_depth.
            if mtp_weight > 0.0 and mtp_depth >= 2 and T > 2:
                mtp_loss = 0.0
                n_terms = 0
                for k in range(2, min(mtp_depth, T - 1) + 1):
                    # logits[:, t] predicts targets[:, t+k-1] when targets is next-token
                    # of idx; with teacher targets aligned as next-token of idx,
                    # targets[:, t] == idx[:, t+1], so token at t+k is targets[:, t+k-1]
                    mtp_loss = mtp_loss + F.cross_entropy(
                        logits[:, : T - k + 1].reshape(-1, logits.size(-1)),
                        targets[:, k - 1 :].reshape(-1),
                    )
                    n_terms += 1
                if n_terms > 0:
                    # decay later horizons slightly (DeepSeek-like)
                    loss = loss + mtp_weight * (mtp_loss / n_terms)
        if use_cache:
            assert presents is not None
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
        *,
        use_cache: bool = True,
    ):
        """Yield each newly sampled token id (shape [1, 1]) as it is generated.

        use_cache=True (default): incremental decode with KV cache (fast on GPU).
        use_cache=False: legacy full-window recompute each step.
        """
        if idx.ndim != 2 or idx.size(0) < 1 or idx.size(1) < 1:
            raise ValueError("generate_stream expects idx of shape (B, T) with T>=1")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        # Drop prefix if longer than context; cache only keeps the window we feed.
        if idx.size(1) > self.config.block_size:
            idx = idx[:, -self.config.block_size :]
        for _ in range(max_new_tokens):
            if use_cache:
                if past is None:
                    idx_cond = idx[:, -self.config.block_size :]
                else:
                    # Only the new token; past holds prior K/V (truncated below).
                    idx_cond = idx[:, -1:]
                out = self(idx_cond, past_key_values=past, use_cache=True)
                assert isinstance(out, tuple) and len(out) == 3
                logits, _, past = out
                # Keep cache within block_size (drop oldest positions).
                max_past = self.config.block_size - 1
                if past and past[0][0].size(2) > max_past:
                    past = [(k[:, :, -max_past:, :], v[:, :, -max_past:, :]) for k, v in past]
            else:
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            # temperature <= 0 → greedy (argmax); avoid div-by-zero / nan probs
            if temperature is None or temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(float(temperature), 1e-6)
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), logits.size(-1))
                    if k > 0:
                        v, _ = torch.topk(logits, k)
                        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                # Guard all-nan/inf (e.g. broken softcap) → greedy fallback
                if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            yield idx_next

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
        *,
        use_cache: bool = True,
    ) -> torch.Tensor:
        for idx_next in self.generate_stream(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
        ):
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                c = b1 * m + (1.0 - b1) * g
                p.mul_(1.0 - lr * wd)
                p.add_(torch.sign(c), alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)
        return loss


def working_set_bytes(model: GPT, optimizer: str, batch: int) -> dict[str, int]:
    cfg = model.config
    b = estimate_budget(cfg, batch, optimizer=optimizer)
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "params": p_bytes,
        "grads_est": p_bytes,
        "opt_est": b.bytes_opt,
        "activations_est": b.bytes_acts,
        "total": b.total,
        "n_params": b.n_params,
    }
