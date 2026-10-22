"""lpspell FILE [--model SPEC]: print spelling issues, one per line."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .checker import Checker, Config, load_words
from .text import LineIndex


def _quiet():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lpspell", description="A spell checker that reads the whole sentence.")
    p.add_argument("file", help="text or Markdown file, or - for stdin")
    p.add_argument(
        "--model",
        default=os.environ.get("LPSPELL_MODEL", "hf:Qwen/Qwen2.5-0.5B"),
        help="hf:NAME, llama:FILE.gguf or py:MODULE:NAME (see lpspell/models.py)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--dict", action="append", default=[], metavar="PATH", help="extra word list (repeatable)")
    p.add_argument("-c", "--config", action="append", default=[], metavar="KEY=VALUE", help="override a Config field")
    args = p.parse_args(argv)
    try:
        cfg = Config().update(dict(kv.split("=", 1) for kv in args.config))
    except (KeyError, ValueError) as err:
        p.error(f"bad -c override: {err}")

    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    _quiet()
    from .models import load

    issues = Checker(load(args.model), cfg, load_words(args.dict)).check(text)
    lines = LineIndex(text)
    if args.json:
        json.dump([i.to_dict(lines) for i in issues], sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for iss in issues:
            line, col = lines.line_col(iss.start)
            print(f"{line}:{col}  {iss.word} → {', '.join(iss.suggestions) or '?'}  ({iss.kind}, {iss.gain:+.1f})")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
