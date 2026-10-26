"""
Chat / complete UI for LL3M.

Inference runs in **ll3m_c** by default. Python only does BPE encode/decode,
streaming display, multi-turn formatting, and the tool agent loop.

  # Product agent (assist off, When2Call gate on):
  uv run python -m ll3m.chat --checkpoint checkpoints/v76_dense_c_pure_v2/best.ll3m chat --mode agent

  # story continuation
  uv run python -m ll3m.chat --checkpoint checkpoints/v76_dense_c_pure_v2/best.ll3m chat --mode story

  # pure-policy stress: --pure-policy   |  gold inject: --assist-tools
  uv run python -m ll3m.chat --python --checkpoint ... chat   # PyTorch fallback
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from ll3m.agent import AgentConfig, run_agent, stream_agent_text
from ll3m.c_engine import CEngine
from ll3m.constrained import quant_sample_defaults


class LL3M:
    """Chat wrapper over C inference; optional `--python` torch fallback."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        python_infer: bool = False,
        device: str = "cpu",
        seed: int | None = None,
        threads: int | None = None,
    ):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        self.checkpoint = path
        self.engine_name = "python" if python_infer else "c"
        self.history: list[str] = []
        self._torch = None
        self._c: CEngine | None = None
        self._block_size = 256
        self._n_layer = 0
        self._n_embd = 0

        if not python_infer:
            self._c = CEngine(path, seed=seed)
            self._block_size = self._c.block_size
            self._n_layer = self._c.n_layer
            self._n_embd = self._c.n_embd
        else:
            import torch
            from ll3m.train import decode, load_model

            if threads is not None:
                torch.set_num_threads(max(1, threads))
            if seed is not None:
                torch.manual_seed(seed)
            self.device = torch.device(device)
            model, stoi, itos, bpe = load_model(path, device=str(self.device))
            model.eval()
            self._torch = {
                "model": model,
                "stoi": stoi,
                "itos": itos,
                "bpe": bpe,
                "decode": decode,
            }
            self._block_size = int(model.config.block_size)
            self._n_layer = int(model.config.n_layer)
            self._n_embd = int(model.config.n_embd)

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def config(self):
        class _C:
            pass

        c = _C()
        c.n_layer = self._n_layer
        c.n_embd = self._n_embd
        c.block_size = self._block_size
        return c

    def stream_complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop: str | None = None,
        include_prompt: bool = False,
    ) -> Iterator[str]:
        if include_prompt and prompt:
            yield prompt
        if self._c is not None:
            yield from self._c.stream_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=40 if top_k is None else top_k,
                stop=stop,
            )
            return
        # torch fallback (only with --python)
        import torch

        assert self._torch is not None
        model = self._torch["model"]
        bpe = self._torch["bpe"]
        stoi = self._torch["stoi"]
        itos = self._torch["itos"]
        decode = self._torch["decode"]
        if bpe is not None:
            ids_list = bpe.encode(prompt)
        else:
            ids_list = [stoi.get(c, 0) for c in prompt]
        if not ids_list:
            ids_list = [0]
        ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        gen = ""
        with torch.no_grad():
            for idx_next in model.generate_stream(
                ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                tid = int(idx_next.item())
                ids = torch.cat([ids, idx_next], dim=1)
                if bpe is not None:
                    piece = bpe.decode([tid])
                else:
                    piece = decode(torch.tensor([tid]), itos)
                if not piece:
                    continue
                gen += piece
                if stop:
                    cut = gen.find(stop)
                    if cut >= 0:
                        prev = len(gen) - len(piece)
                        if cut > prev:
                            yield gen[prev:cut]
                        return
                yield piece

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop: str | None = None,
        include_prompt: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        parts: list[str] = []
        if include_prompt:
            parts.append(prompt)
        for piece in self.stream_complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            include_prompt=False,
        ):
            parts.append(piece)
            if on_token is not None:
                on_token(piece)
        return "".join(parts)

    def reset_chat(self) -> None:
        self.history.clear()
        self._story_buf = ""
        # Agent multi-turn buffer must clear too; battery/eval call reset_chat
        # between cases and must not leak prior tool turns into the next prompt.
        if hasattr(self, "_agent_history"):
            self._agent_history.clear()

    def _ensure_story_buf(self) -> None:
        if not hasattr(self, "_story_buf"):
            self._story_buf = ""

    def _story_seed(self, user_message: str) -> str:
        """Map a short user line into a stem the story model can continue.

        The product model is trained on children's-story / mix text, not
        instruction chat. Bare names or topics become a simple story opener.
        Long lines that already look like prose are used as-is.
        """
        msg = user_message.strip()
        if not msg:
            return "Once upon a time"
        # Already a story / prose stem
        low = msg.lower()
        if low.startswith("once upon") or low.startswith("one day") or "\n" in msg:
            return msg if msg.endswith((" ", "\n", ".", "!", "?", ",")) else msg + " "
        # Ends mid-sentence: continue as-is
        if msg[-1] not in ".!?" and len(msg.split()) >= 6:
            return msg if msg.endswith(" ") else msg + " "
        # Short topic / name → story frame (keeps domain distribution)
        topic = msg.rstrip(".!?")
        # Prefer "a man/girl named …" over bare "someone" for readability
        if topic[:1].isupper() and len(topic.split()) <= 4:
            return f"Once upon a time, there was a person named {topic}. "
        return f"Once upon a time, there was a story about {topic}. "

    def _chat_prompt(self, user_message: str, system: str | None = None) -> str:
        """Build a continuation prompt (story LM), not User/Assistant tags.

        `system` is optional prefix prose (ignored if empty). Multi-turn keeps
        the growing story buffer plus the new user seed.
        """
        self._ensure_story_buf()
        seed = self._story_seed(user_message)
        parts: list[str] = []
        if system:
            parts.append(system.rstrip() + "\n\n")
        if self._story_buf:
            # Continue the shared story; user line steers the next beat
            buf = self._story_buf.rstrip()
            if not buf.endswith((".", "!", "?", "\n")):
                buf += "."
            parts.append(buf + " ")
            # Short steer: weave user text as next story beat
            steer = user_message.strip()
            if steer and steer.lower() not in buf.lower():
                parts.append(f"Then, about {steer}: ")
            else:
                parts.append(seed if not self._story_buf else "")
        else:
            parts.append(seed)
        return "".join(parts)

    def stream_chat_reply(
        self,
        user_message: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        system: str | None = None,
    ) -> Iterator[str]:
        """Yield story continuation pieces; updates shared story when done."""
        self._ensure_story_buf()
        prompt = self._chat_prompt(user_message, system=system)
        pieces: list[str] = []
        for piece in self.stream_complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop="\n\n\n",
            include_prompt=False,
        ):
            pieces.append(piece)
            yield piece
        cont = "".join(pieces).rstrip()
        # Store full text so next turn continues the same narrative
        self._story_buf = (prompt + cont).strip()
        # Keep history list for /reset compatibility / trimming by char budget
        self.history.append(user_message)
        self._trim_history()

    def chat_reply(
        self,
        user_message: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        system: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """One interactive turn: story continuation. Streams via on_token if set."""
        parts: list[str] = []
        for piece in self.stream_chat_reply(
            user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            system=system,
        ):
            parts.append(piece)
            if on_token is not None:
                on_token(piece)
        return "".join(parts).lstrip()

    def _trim_history(self) -> None:
        """Trim story buffer to roughly fit context (char proxy for tokens)."""
        self._ensure_story_buf()
        budget = max(128, self.block_size * 3)
        if len(self._story_buf) > budget:
            self._story_buf = self._story_buf[-budget:]
        while self.history and len(self.history) > 16:
            self.history.pop(0)

    def _generate_complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stop: str | None = None,
        include_prompt: bool = False,
    ) -> str:
        return self.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
            include_prompt=include_prompt,
        )

    def agent_reply(
        self,
        user_message: str,
        *,
        max_tokens: int = 120,
        temperature: float = 0.4,
        top_k: int | None = 40,
        assist_tools: bool = False,
        gate_tools: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Q&A turn with tool loop. Product default: assist off, gate on."""
        if not hasattr(self, "_agent_history"):
            self._agent_history: list[str] = []
        cfg = AgentConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=40 if top_k is None else top_k,
            assist_tools=assist_tools,
            gate_tools=gate_tools,
        )

        def gen(prompt: str, **kw) -> str:
            return self._generate_complete(prompt, **kw)

        turn = run_agent(
            user_message,
            gen,
            history=self._agent_history,
            cfg=cfg,
            on_token=on_token,
        )
        # keep last few turns only (T=256)
        while len(self._agent_history) > 4:
            self._agent_history.pop(0)
        return turn.final

    def stream_agent_reply(
        self,
        user_message: str,
        *,
        max_tokens: int = 120,
        temperature: float = 0.4,
        top_k: int | None = 40,
        assist_tools: bool = False,
        gate_tools: bool = True,
    ) -> Iterator[str]:
        if not hasattr(self, "_agent_history"):
            self._agent_history = []
        cfg = AgentConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=40 if top_k is None else top_k,
            assist_tools=assist_tools,
            gate_tools=gate_tools,
        )

        def gen(prompt: str, **kw) -> str:
            return self._generate_complete(prompt, **kw)

        def gen_stream(prompt: str, **kw) -> Iterator[str]:
            yield from self.stream_complete(prompt, **kw)

        yield from stream_agent_text(
            user_message,
            gen_stream,
            gen,
            history=self._agent_history,
            cfg=cfg,
        )
        while len(self._agent_history) > 4:
            self._agent_history.pop(0)

    def interactive_chat(
        self,
        *,
        mode: str = "story",
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        stream: bool = True,
        assist_tools: bool = False,
        gate_tools: bool = True,
    ) -> None:
        mode = mode.lower().strip()
        if mode not in ("story", "agent"):
            raise ValueError("mode must be story or agent")
        self._ensure_story_buf()
        if mode == "agent":
            self._agent_history = []
            print(
                f"ll3m agent  ({self.checkpoint.name}, engine={self.engine_name}, "
                f"L{self.config.n_layer} d{self.config.n_embd} T{self.block_size})"
                f"{'  [streaming]' if stream else ''}"
            )
            print(
                "Q&A + tools: calc, time, lookup, wiki, note. "
                "Host runs tools; model emits <call>…</call>."
            )
            print(
                "Product default: no gold inject, When2Call gate on "
                f"(assist={assist_tools}, gate={gate_tools}). "
                "Pure stress: --pure-policy. Gold inject: --assist-tools."
            )
            print("Commands: /reset  /temp N  /tokens N  /quit  /mode story|agent")
            # Apply quant-aware agent defaults unless user overrode CLI defaults
            qd = quant_sample_defaults("agent")
            if temperature in (0.7, 0.8):  # stock complete defaults
                temperature = qd.temperature
            if top_k in (30, 40, None):
                top_k = qd.top_k
            max_tokens = min(max_tokens, 120)
        else:
            print(
                f"ll3m story  ({self.checkpoint.name}, engine={self.engine_name}, "
                f"L{self.config.n_layer} d{self.config.n_embd} T{self.block_size})"
                f"{'  [streaming]' if stream else ''}"
            )
            print(
                "Story LM: type a seed or topic "
                "(e.g. 'a brave mouse', 'Once upon a time…')."
            )
            print("Commands: /reset  /temp N  /tokens N  /quit  /mode story|agent")

        while True:
            try:
                line = input("You> ").rstrip("\n")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line.strip():
                continue
            if line.strip() in ("/quit", "/exit", "/q"):
                break
            if line.strip() == "/reset":
                self.reset_chat()
                if mode == "agent":
                    self._agent_history = []
                    print("(agent history cleared)")
                else:
                    print("(story cleared)")
                continue
            if line.startswith("/mode "):
                new_mode = line.split(None, 1)[1].strip().lower()
                if new_mode in ("story", "agent"):
                    mode = new_mode
                    self.reset_chat()
                    self._agent_history = []
                    print(f"(mode={mode})")
                    if mode == "agent":
                        temperature = 0.4
                        max_tokens = min(max_tokens, 120)
                else:
                    print("(usage: /mode story|agent)")
                continue
            if line.startswith("/temp "):
                try:
                    temperature = float(line.split(None, 1)[1])
                    print(f"(temperature={temperature})")
                except ValueError:
                    print("(usage: /temp 0.8)")
                continue
            if line.startswith("/tokens "):
                try:
                    max_tokens = int(line.split(None, 1)[1])
                    print(f"(max_tokens={max_tokens})")
                except ValueError:
                    print("(usage: /tokens 200)")
                continue

            label = "Bot> " if mode == "agent" else "Story> "
            print(label, end="", flush=True)
            try:
                if mode == "agent":
                    if stream:
                        for piece in self.stream_agent_reply(
                            line,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            assist_tools=assist_tools,
                            gate_tools=gate_tools,
                        ):
                            print(piece, end="", flush=True)
                    else:
                        print(
                            self.agent_reply(
                                line,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                assist_tools=assist_tools,
                                gate_tools=gate_tools,
                            ),
                            end="",
                            flush=True,
                        )
                else:
                    if stream:
                        for piece in self.stream_chat_reply(
                            line,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                        ):
                            print(piece, end="", flush=True)
                    else:
                        print(
                            self.chat_reply(
                                line,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                            ),
                            end="",
                            flush=True,
                        )
            except KeyboardInterrupt:
                print("\n(interrupted)", flush=True)
                continue
            print(flush=True)


def _default_checkpoint() -> Path:
    root = Path(__file__).resolve().parent.parent
    for cand in (
        root / "checkpoints" / "v75" / "best.pt",
        root / "checkpoints" / "best.pt",
    ):
        if cand.is_file():
            return cand
    return root / "checkpoints" / "best.pt"


def _add_gen_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    # Default 0.7 under ternary (Colibrì: high temp samples quant noise). Agent
    # mode still forces 0.4 unless the user set a non-default temp.
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (default 0.7 for ternary; agent mode uses 0.4)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="top-k sampling (default 30 under ternary; agent uses 20)",
    )
    ap.add_argument("--tokens", type=int, default=200, help="max new tokens")
    ap.add_argument(
        "--no-stream",
        action="store_true",
        help="disable token streaming (print full reply at once)",
    )


def _stream_write(piece: str) -> None:
    sys.stdout.write(piece)
    sys.stdout.flush()


def build_argparser() -> argparse.ArgumentParser:
    # Shared gen flags via parent so subparsers do not clobber defaults.
    gen = argparse.ArgumentParser(add_help=False)
    _add_gen_flags(gen)

    p = argparse.ArgumentParser(description="ll3m completion / chat wrapper")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_default_checkpoint()),
        help="path to .pt or .ll3m (default: v75/best.pt or best.pt)",
    )
    p.add_argument(
        "--python",
        action="store_true",
        dest="python_infer",
        help="fallback: run model.generate in PyTorch instead of ll3m_c",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("complete", help="one-shot completion", parents=[gen])
    c.add_argument("--prompt", type=str, required=True)
    c.add_argument(
        "--stop",
        type=str,
        default=None,
        help="stop string in generated suffix (e.g. '\\n\\n')",
    )
    c.add_argument(
        "--no-prompt",
        action="store_true",
        help="print only the generated continuation",
    )

    ch = sub.add_parser("chat", help="interactive multi-turn chat", parents=[gen])
    ch.add_argument(
        "--message",
        type=str,
        default=None,
        help="single message then exit (non-interactive)",
    )
    ch.add_argument(
        "--mode",
        type=str,
        default="agent",
        choices=["agent", "story"],
        help="agent=Q&A+tools (default); story=narrative continuation",
    )
    ch.add_argument(
        "--assist-tools",
        action="store_true",
        help="agent: enable host gold inject / calc repair (default: off)",
    )
    ch.add_argument(
        "--no-assist-tools",
        action="store_true",
        help="deprecated alias of default; use --pure-policy for gate-off stress test",
    )
    ch.add_argument(
        "--pure-policy",
        action="store_true",
        help=(
            "agent: assist off and gate off (pure model policy stress; "
            "not product default)"
        ),
    )
    ch.add_argument(
        "--no-gate",
        action="store_true",
        help="agent: disable When2Call gate only (keep assist off unless --assist-tools)",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    try:
        lm = LL3M(
            args.checkpoint,
            python_infer=args.python_infer,
            seed=args.seed,
            threads=args.threads,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"engine init failed: {e}", file=sys.stderr)
        if not args.python_infer:
            print(
                "hint: make -C c_infer -j  && retry, or pass --python for torch fallback",
                file=sys.stderr,
            )
        sys.exit(1)

    stream = not args.no_stream

    if args.cmd == "complete":
        if not args.no_prompt:
            if stream:
                _stream_write(args.prompt)
            else:
                pass
        if stream:
            for piece in lm.stream_complete(
                args.prompt,
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop=args.stop,
                include_prompt=False,
            ):
                _stream_write(piece)
            _stream_write("\n")
        else:
            text = lm.complete(
                args.prompt,
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop=args.stop,
                include_prompt=not args.no_prompt,
            )
            print(text)
        return

    if args.cmd == "chat":
        mode = getattr(args, "mode", "agent")
        # Product default: assist off, When2Call gate on.
        # --pure-policy: both off. --assist-tools: gold inject. --no-gate: gate off only.
        assist = bool(getattr(args, "assist_tools", False))
        pure = bool(getattr(args, "pure_policy", False))
        if pure:
            assist = False
            gate = False
        else:
            gate = not bool(getattr(args, "no_gate", False))
        if args.message is not None:
            if mode == "agent":
                if stream:
                    for piece in lm.stream_agent_reply(
                        args.message,
                        max_tokens=min(args.tokens, 120),
                        temperature=(
                            quant_sample_defaults("agent").temperature
                            if args.temperature in (0.7, 0.8)
                            else args.temperature
                        ),
                        top_k=(
                            quant_sample_defaults("agent").top_k
                            if args.top_k in (30, 40)
                            else args.top_k
                        ),
                        assist_tools=assist,
                        gate_tools=gate,
                    ):
                        _stream_write(piece)
                    _stream_write("\n")
                else:
                    print(
                        lm.agent_reply(
                            args.message,
                            max_tokens=min(args.tokens, 120),
                            temperature=(
                                quant_sample_defaults("agent").temperature
                                if args.temperature in (0.7, 0.8)
                                else args.temperature
                            ),
                            top_k=(
                                quant_sample_defaults("agent").top_k
                                if args.top_k in (30, 40)
                                else args.top_k
                            ),
                            assist_tools=assist,
                            gate_tools=gate,
                        )
                    )
            else:
                if stream:
                    for piece in lm.stream_chat_reply(
                        args.message,
                        max_tokens=args.tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    ):
                        _stream_write(piece)
                    _stream_write("\n")
                else:
                    print(
                        lm.chat_reply(
                            args.message,
                            max_tokens=args.tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                        )
                    )
            return
        lm.interactive_chat(
            mode=mode,
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=stream,
            assist_tools=assist,
            gate_tools=gate,
        )
        return


if __name__ == "__main__":
    main()
