"""GPT configuration (v2: Qwen3-dense-style)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """v2 GPT config.

    Library defaults are minimal (dropout 0, LayerScale off). The **product
    freeze** is not these dataclass defaults: see checkpoints/v75_freeze.json
    and AGENTS.md (L23×d576 MQA, mid 1536, peri_ln=False, layer_scale_init=1e-4,
    dropout=0.1, V=2048, T=256). Train CLI / freeze JSON set product values.
    """

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0  # product freeze uses 0.1
    n_kv_head: int | None = None  # product freeze: 1 (MQA)
    intermediate_size: int | None = None  # product freeze: 1536
    rope_theta: float = 10000.0
    # Fraction of head_dim that receives RoPE (LLaMA-3.1-style partial rotary).
    rope_partial: float = 1.0
    # Final-logit softcap (Gemma-style): logits = cap * tanh(logits/cap). 0 = off.
    logit_softcap: float = 0.0
    # Gemma2/OLMo2-style peri-LN: RMSNorm after each residual add (pre-norm kept).
    # Product freeze: False (peri blocked early learning on L23×d576).
    peri_ln: bool = False
    # Per-channel residual LayerScale (CaiT/ViT); 0 = off, else init value.
    # Product freeze: 1e-4 (not 0.0; 1e-2 was too aggressive on this shape).
    layer_scale_init: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.rope_partial <= 1.0:
            raise ValueError("rope_partial must be in (0, 1]")
        if self.logit_softcap < 0.0:
            raise ValueError("logit_softcap must be >= 0")
        if self.layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be >= 0")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.n_head < 1:
            raise ValueError("n_head must be >= 1")
        if self.n_embd < 1:
            raise ValueError("n_embd must be >= 1")
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError("intermediate_size must be >= 1 when set")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be > 0")
        nkv = self.n_kv_head if self.n_kv_head is not None else self.n_head
        if nkv < 1:
            raise ValueError("n_kv_head must be >= 1")
        if self.n_head % nkv != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.n_embd % nkv != 0:
            raise ValueError("n_embd must be divisible by n_kv_head")
        rd = self.rope_dims
        if rd % 2 != 0:
            raise ValueError(
                "rope dims (head_dim * rope_partial, even-floored) must be even"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def rope_dims(self) -> int:
        """Even number of head dims that receive rotary (partial RoPE)."""
        hd = self.head_dim
        raw = int(hd * self.rope_partial)
        even = raw - (raw % 2)
        if even < 2 and self.rope_partial > 0:
            even = 2
        return min(even, hd)

    @property
    def kv_head(self) -> int:
        return self.n_kv_head if self.n_kv_head is not None else self.n_head

    @property
    def intermediate(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        # ~ (8/3) d, rounded to multiple of 64 for GEMM friendliness
        raw = int(round(self.n_embd * 8 / 3))
        return max(64, (raw + 63) // 64 * 64)

    @property
    def arch(self) -> str:
        """Always v2 (legacy field kept for checkpoint dicts)."""
        return "v2"

    def n_params(self) -> int:
        d, L, V = self.n_embd, self.n_layer, self.vocab_size
        nkv = self.kv_head
        hd = self.head_dim
        mid = self.intermediate
        emb = V * d
        per = (
            d
            + d * d
            + nkv * hd * d
            + nkv * hd * d
            + d * d
            + hd
            + hd
            + d
            + mid * d
            + mid * d
            + d * mid
        )
        if self.peri_ln:
            per += 2 * d
        if self.layer_scale_init > 0.0:
            per += 2 * d
        return emb + L * per + d

    def bytes_for_training(
        self,
        batch: int,
        optimizer: str = "lion",
        checkpoint: bool = True,
        dtype_bytes: int = 4,
    ) -> dict[str, int]:
        n = self.n_params()
        B, T, d, L, nh = batch, self.block_size, self.n_embd, self.n_layer, self.n_head
        nkv = self.kv_head
        mid = self.intermediate
        if optimizer == "lion":
            state_mult = 3
        elif optimizer == "sgd":
            state_mult = 3
        else:
            state_mult = 4
        state = state_mult * n * dtype_bytes
        if checkpoint:
            acts = (
                (L + 2) * B * T * d
                + B * nh * T * T
                + B * T * mid
                + B * T * (d + 2 * nkv * (d // nh))
                + 3 * B * T * d
            ) * dtype_bytes
        else:
            acts = (
                (L + 1) * B * T * d
                + L * B * nh * T * T
                + L * B * T * mid
                + 2 * B * T * d
            ) * dtype_bytes
        return {
            "params_grads_opt": state,
            "activations": acts,
            "total": state + acts,
            "n_params": n,
            "optimizer": optimizer,
            "checkpoint": checkpoint,
        }

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "n_kv_head": self.kv_head,
            "intermediate_size": self.intermediate,
            "rope_theta": self.rope_theta,
            "rope_partial": self.rope_partial,
            "logit_softcap": self.logit_softcap,
            "peri_ln": self.peri_ln,
            "layer_scale_init": self.layer_scale_init,
            "arch": "v2",
        }

    @staticmethod
    def from_dict(c: dict) -> "GPTConfig":
        known = {
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "n_kv_head",
            "intermediate_size",
            "rope_theta",
            "rope_partial",
            "logit_softcap",
            "peri_ln",
            "layer_scale_init",
        }
        kw = {k: c[k] for k in known if k in c}
        kw.setdefault("dropout", 0.0)
        kw.setdefault("rope_partial", 1.0)
        kw.setdefault("logit_softcap", 0.0)
        kw.setdefault("peri_ln", False)
        kw.setdefault("layer_scale_init", 0.0)
        if "arch" not in c:
            raise ValueError(
                "checkpoint missing arch; only v2 is supported "
                "(legacy v1/CLM1 archives removed)"
            )
        if c.get("arch") != "v2":
            raise ValueError(
                f"unsupported arch={c.get('arch')!r}; only v2 remains "
                "(v1/CLM1 removed)"
            )
        return GPTConfig(**kw)
