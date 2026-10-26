"""Generate text via the C inference engine (ll3m_c).

Default: ll3m_c. Python only tokenizes and decodes streamed token ids.
Pass --python to fall back to in-process PyTorch generate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ll3m.c_engine import CEngine


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Generate from ll3m checkpoint via C engine"
    )
    p.add_argument("--checkpoint", type=str, required=True, help=".pt or .ll3m path")
    p.add_argument("--prompt", type=str, default="ROMEO:\n")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="default 0.7 under ternary (high temp samples quant noise)",
    )
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="print only the continuation",
    )
    p.add_argument(
        "--python",
        action="store_true",
        dest="python_infer",
        help="fallback: run model.generate in PyTorch instead of ll3m_c",
    )
    args = p.parse_args(argv)

    if args.python_infer:
        _torch_generate(args)
        return

    eng = CEngine(args.checkpoint, seed=args.seed)
    if not args.no_prompt:
        sys.stdout.write(args.prompt)
        sys.stdout.flush()
    for piece in eng.stream_generate(
        args.prompt,
        max_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")


def _torch_generate(args: argparse.Namespace) -> None:
    import torch
    from ll3m.train import decode, load_model

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, stoi, itos, bpe = load_model(Path(args.checkpoint))
    if bpe is not None:
        ids = torch.tensor([bpe.encode(args.prompt)], dtype=torch.long)
    else:
        ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    if bpe is not None:
        text = bpe.decode(out[0])
    else:
        text = decode(out[0], itos)
    if args.no_prompt and text.startswith(args.prompt):
        text = text[len(args.prompt) :]
    print(text)


if __name__ == "__main__":
    main()
