"""LL3M: an LLM that lives in L3.

Small GPT trained from scratch; C inference fits one CPU L3 CCD.
v2: RMSNorm, RoPE, SwiGLU, GQA/MQA, small BPE. See README and docs/.
"""

from ll3m.cache_info import CacheHierarchy, probe_cache
from ll3m.config import GPTConfig

# Lazy: avoid importing torch at package import.
def __getattr__(name: str):
    if name == "LL3M":
        from ll3m.chat import LL3M

        return LL3M
    if name == "GPT":
        from ll3m.model_torch import GPT

        return GPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CacheHierarchy",
    "probe_cache",
    "LL3M",
    "GPT",
    "GPTConfig",
]
__version__ = "0.1.0"
