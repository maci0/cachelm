"""C inference engine backend for LL3M.

Python owns BPE encode/decode and UI. Generation runs in `ll3m_c`.
Weight packs: *.ll3m (magic L3T2 ternary / L3B2 binary / L3M2 f32).
Legacy *.clm + CLT2/CLM2 OK.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ll3m.tokenizer import BPETokenizer

WEIGHTS_EXT = ".ll3m"
_LEGACY_EXT = ".clm"
# New + legacy magics
_MAGICS = {
    b"L3T2": "L3T2",
    b"L3B2": "L3B2",  # research binary Linears + int8 embeds
    b"L3M2": "L3M2",
    b"CLT2": "CLT2",  # legacy ternary
    b"CLM2": "CLM2",  # legacy f32
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_ll3m_c() -> Path:
    root = _repo_root()
    for name in ("ll3m_c", "cachelm_c"):
        p = root / "c_infer" / name
        if p.is_file() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError("ll3m_c not found; build with: make -C c_infer -j")


@dataclass(frozen=True)
class WeightsMeta:
    magic: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_kv_head: int
    intermediate: int
    rope_theta: float = 10000.0
    has_ls: bool = False
    has_peri: bool = False

    @property
    def is_ternary(self) -> bool:
        return self.magic in ("L3T2", "CLT2")

    @property
    def is_binary(self) -> bool:
        return self.magic == "L3B2"


# Back-compat alias
ClmMeta = WeightsMeta


# Header: magic(4) + 7×u32 + f32 rope + u32 flags + u32 tok_len  (44 bytes)
_HEADER_FMT = "<4s7IfII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _read_pack_header(path: Path) -> tuple[bytes, int, int, int, int, int, int, int, float, int, int]:
    """Return (magic, V, T, L, H, nkv, D, mid, theta, flags, tok_len)."""
    with open(path, "rb") as f:
        raw = f.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"truncated header: {path}")
    magic_b, V, T, L, H, nkv, D, mid, theta, flags, tok_len = struct.unpack(
        _HEADER_FMT, raw
    )
    if magic_b not in _MAGICS:
        raise ValueError(
            f"not an ll3m weight file: {path} magic={magic_b!r} "
            f"(want L3T2/L3B2/L3M2 or legacy CLT2/CLM2)"
        )
    if min(V, T, L, H, nkv, D, mid) < 1:
        raise ValueError(
            f"invalid weight header dims in {path}: "
            f"V={V} T={T} L={L} H={H} nkv={nkv} D={D} mid={mid}"
        )
    if H % nkv != 0:
        raise ValueError(
            f"invalid weight header: n_head={H} not divisible by n_kv_head={nkv}"
        )
    if D % H != 0:
        raise ValueError(
            f"invalid weight header: n_embd={D} not divisible by n_head={H}"
        )
    if not (theta > 0.0) or theta != theta:  # NaN check
        raise ValueError(f"invalid rope_theta={theta} in {path}")
    if tok_len < 0 or tok_len > 64 * 1024 * 1024:
        raise ValueError(f"invalid tok_len={tok_len} in {path}")
    return magic_b, V, T, L, H, nkv, D, mid, theta, flags, tok_len


def read_weights_meta(path: Path) -> WeightsMeta:
    """Parse L3T2/L3B2/L3M2 (or legacy CLT2/CLM2) header without loading weights."""
    magic_b, V, T, L, H, nkv, D, mid, theta, flags, _tok_len = _read_pack_header(path)
    return WeightsMeta(
        magic=_MAGICS[magic_b],
        vocab_size=V,
        block_size=T,
        n_layer=L,
        n_head=H,
        n_embd=D,
        n_kv_head=nkv,
        intermediate=mid,
        rope_theta=theta,
        has_ls=bool(flags & 1),
        has_peri=bool(flags & 2),
    )


def extract_pack_tokenizer_bytes(path: Path) -> bytes | None:
    """Return embedded tokenizer.json blob from an L3T2/L3M2 pack, or None."""
    path = Path(path)
    if path.suffix not in (WEIGHTS_EXT, _LEGACY_EXT):
        return None
    _magic_b, _V, _T, _L, _H, _nkv, _D, _mid, _theta, _flags, tok_len = (
        _read_pack_header(path)
    )
    if tok_len <= 0:
        return None
    with open(path, "rb") as f:
        f.seek(_HEADER_SIZE)
        blob = f.read(tok_len)
    if len(blob) != tok_len:
        raise ValueError(
            f"truncated tokenizer blob in {path}: want {tok_len} got {len(blob)}"
        )
    # Must look like our BPE JSON
    if b"stoi" not in blob or b"merges" not in blob:
        raise ValueError(f"pack tokenizer blob in {path} missing stoi/merges")
    return blob


def resolve_tokenizer_path(
    weights: Path,
    *,
    tokenizer_path: str | Path | None = None,
) -> Path:
    """Pick BPE file matching the weight pack.

    Order:
      1. explicit tokenizer_path
      2. embedded blob in .ll3m/.clm → written beside pack as ``.tokenizer.json``
      3. ``<weights_dir>/tokenizer.json``
      4. repo ``data/tokenizer.json`` (legacy story default)
    """
    if tokenizer_path is not None:
        p = Path(tokenizer_path)
        if not p.is_file():
            raise FileNotFoundError(f"tokenizer not found: {p}")
        return p

    weights = Path(weights)
    if weights.suffix in (WEIGHTS_EXT, _LEGACY_EXT):
        blob = extract_pack_tokenizer_bytes(weights)
        if blob is not None:
            # Sidecar next to pack so reloads are cheap and inspectable
            side = weights.with_suffix(weights.suffix + ".tokenizer.json")
            if (
                not side.is_file()
                or side.stat().st_size != len(blob)
                or side.read_bytes() != blob
            ):
                side.write_bytes(blob)
            return side

    side2 = weights.parent / "tokenizer.json"
    if side2.is_file():
        return side2

    legacy = _repo_root() / "data" / "tokenizer.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"BPE tokenizer not found for {weights} "
        f"(no pack blob, no sidecar, no data/tokenizer.json)"
    )


# Back-compat
read_clm_meta = read_weights_meta


def ensure_weights(
    checkpoint: Path,
    *,
    out_path: Path | None = None,
    tokenizer: Path | None = None,
    quant: str = "ternary",
) -> Path:
    """Export .pt → .ll3m if needed. Default quant=ternary (L3T2 product)."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix in (WEIGHTS_EXT, _LEGACY_EXT):
        return checkpoint
    out = out_path or checkpoint.with_suffix(WEIGHTS_EXT)
    tok = tokenizer or (_repo_root() / "data" / "tokenizer.json")
    if not out.is_file() or out.stat().st_mtime < checkpoint.stat().st_mtime:
        root = _repo_root()
        cmd = [
            sys.executable,
            str(root / "scripts" / "export_weights.py"),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out),
            "--quant",
            quant,
        ]
        if tok.is_file():
            cmd.extend(["--tokenizer", str(tok)])
        subprocess.check_call(cmd, cwd=str(root))
    return out


# Back-compat
def ensure_clm(
    checkpoint: Path,
    *,
    clm_path: Path | None = None,
    tokenizer: Path | None = None,
    quant: str = "ternary",
) -> Path:
    return ensure_weights(
        checkpoint, out_path=clm_path, tokenizer=tokenizer, quant=quant
    )


class CEngine:
    """Generate tokens via ll3m_c subprocess (streaming token ids)."""

    def __init__(
        self,
        model: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        binary: str | Path | None = None,
        use_hard: bool = False,
        seed: int | None = None,
    ):
        self.binary = Path(binary) if binary else find_ll3m_c()
        model = Path(model)
        if model.suffix == ".pt":
            # Prefer explicit tok; else path recorded in ckpt; else pack embed later
            tok_for_export = (
                Path(tokenizer_path)
                if tokenizer_path
                else _repo_root() / "data" / "tokenizer.json"
            )
            self.weights = ensure_weights(
                model, tokenizer=tok_for_export if tok_for_export.is_file() else None
            )
        else:
            self.weights = model
        if not self.weights.is_file():
            raise FileNotFoundError(f"model not found: {self.weights}")
        self.meta = read_weights_meta(self.weights)
        self.use_hard = use_hard
        self.seed = seed
        tok = resolve_tokenizer_path(self.weights, tokenizer_path=tokenizer_path)
        self.bpe = BPETokenizer.load(tok)
        self.tokenizer_path = tok
        # legacy attr names
        self.clm = self.weights

    @property
    def block_size(self) -> int:
        return self.meta.block_size

    @property
    def n_layer(self) -> int:
        return self.meta.n_layer

    @property
    def n_embd(self) -> int:
        return self.meta.n_embd

    def encode(self, text: str) -> list[int]:
        return self.bpe.encode(text)

    def decode_token(self, tid: int) -> str:
        return self.bpe.decode([tid])

    def decode_ids(self, ids: list[int]) -> str:
        return self.bpe.decode(ids)

    def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        stop: str | None = None,
    ) -> Iterator[str]:
        """Yield text pieces from C engine as token ids stream in."""
        ids = self.encode(prompt)
        if not ids:
            ids = [0]
        if len(ids) > self.meta.block_size:
            ids = ids[-self.meta.block_size :]

        cmd = [
            str(self.binary),
            "-m",
            str(self.weights),
            "--quiet",
            "--stream",
            "--token-ids",
            ",".join(str(i) for i in ids),
            "-n",
            str(max_tokens),
            "-t",
            str(temperature),
            "-k",
            str(top_k),
            "--warm",
            "0",
            "--steady-warm",
            "0",
            "--no-cache-stats",
        ]
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])
        if temperature <= 0:
            cmd.append("--greedy")
        if self.use_hard:
            cmd.append("--residency-hard")

        # Drain stderr on a side thread so a chatty C process cannot fill the
        # OS pipe and deadlock while we only read stdout.
        err_chunks: list[str] = []

        def _drain_err(stream) -> None:
            try:
                if stream is None:
                    return
                for chunk in stream:
                    err_chunks.append(chunk)
            except Exception:
                pass

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        import threading

        err_thread = threading.Thread(
            target=_drain_err, args=(proc.stderr,), daemon=True
        )
        err_thread.start()
        gen = ""
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    tid = int(line)
                except ValueError:
                    continue
                # OOB token ids: decode as empty / skip rather than crash BPE
                if tid < 0 or tid >= self.meta.vocab_size:
                    continue
                piece = self.decode_token(tid)
                if not piece:
                    continue
                gen += piece
                if stop:
                    cut = gen.find(stop)
                    if cut >= 0:
                        prev = len(gen) - len(piece)
                        if cut > prev:
                            yield gen[prev:cut]
                        proc.terminate()
                        return
                yield piece
        finally:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            err_thread.join(timeout=2)
            err_buf = "".join(err_chunks)
            if proc.returncode not in (0, None, -15) and err_buf:
                sys.stderr.write(err_buf)
                raise RuntimeError(
                    f"ll3m_c exited {proc.returncode}: {err_buf.strip()[:400]}"
                )


def export_and_check(checkpoint: str | Path) -> Path:
    return ensure_weights(Path(checkpoint))
