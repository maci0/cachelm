"""
Capability gates against a shipped v2 checkpoint (prefer v75).

Skip only if no v2 checkpoint exists yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    ROOT / "checkpoints" / "v75" / "best.pt",
    ROOT / "checkpoints" / "v2_dense" / "best.pt",
    ROOT / "checkpoints" / "best.pt",
)


def _pick_ckpt() -> Path | None:
    for p in CANDIDATES:
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="module")
def best_model():
    path = _pick_ckpt()
    if path is None:
        pytest.skip("no v2 checkpoint found")
    from ll3m.config import GPTConfig
    from ll3m.train import load_model

    # Reject legacy v1 archives early
    import torch as _t

    ck = _t.load(path, map_location="cpu", weights_only=False)
    arch = dict(ck.get("config", {})).get("arch", "v1")
    if arch != "v2":
        pytest.skip(f"{path.name} is arch={arch}; v1 removed")
    model, stoi, itos, bpe = load_model(path, device="cpu")
    model.eval()
    return model, stoi, itos, bpe, path


def test_checkpoint_is_v2_transformer(best_model) -> None:
    model, stoi, itos, bpe, _path = best_model
    cfg = model.config
    assert cfg.arch == "v2"
    assert cfg.n_layer >= 1
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.vocab_size == len(stoi) == len(itos)
    assert bpe is not None
    w = next(model.parameters())
    assert w.numel() > 1000
    assert float(w.detach().std()) > 1e-4


def test_val_nll_finite(best_model) -> None:
    """Smoke: NLL on a short BPE-encoded sample is finite."""
    from ll3m.eval import nll_over_data

    model, _stoi, _itos, bpe, _path = best_model
    text = "Once upon a time there was a little girl named Lily.\n" * 40
    ids = torch.tensor(bpe.encode(text), dtype=torch.long)
    if len(ids) < model.config.block_size + 2:
        pytest.skip("sample too short for block")
    val_nll, n_tok = nll_over_data(
        model, ids, model.config.block_size, batch_size=4, max_batches=8
    )
    assert n_tok > 0
    assert val_nll == val_nll  # not NaN
    assert 0.0 < val_nll < 20.0


def test_cache_claim_for_checkpoint_config(best_model) -> None:
    """Product gate: ternary C-infer WS under 95% of one CCD L3."""
    from ll3m.cache_info import probe_cache
    from ll3m.model_torch import estimate_infer_ws

    model, _, _, _, _ = best_model
    hier = probe_cache()
    gate = int(hier.l3_bytes * 0.95)
    inf = estimate_infer_ws(model.config, quant="ternary")
    assert inf.total <= gate


def test_generate_printable(best_model) -> None:
    """Real generate path: non-empty sample via torch fallback."""
    model, _stoi, _itos, bpe, _ = best_model
    prompt = "Once upon a time"
    ids = torch.tensor([bpe.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, temperature=0.8, top_k=40)
    text = bpe.decode(out[0].tolist())
    assert len(text) > len(prompt)
