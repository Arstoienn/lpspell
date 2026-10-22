"""A word-level stand-in for a language model, so tests run without one."""

from __future__ import annotations

import re


class FakeLM:
    """Each whitespace-separated word is one token.

    A word scores ``bigram`` if (previous word, word) is in ``good``, ``known``
    if it's in ``vocab``, else ``unk``. Punctuation is ignored for lookup.
    """

    def __init__(
        self, good: set[tuple[str, str]], vocab: set[str], unk=-12.0, known=-4.0, bigram=-0.5, cased=False
    ):
        self.good, self.vocab, self.cased = good, vocab, cased
        self.unk, self.known, self.bigram = unk, known, bigram
        self.calls = 0

    @staticmethod
    def _strip(w: str) -> str:
        return w.strip(".,!?;:\"'()")

    @staticmethod
    def _norm(w: str) -> str:
        return w.lower().strip(".,!?;:\"'()")

    def logprobs(self, texts: list[str]) -> list[list[tuple[int, int, float]]]:
        self.calls += 1
        norm = self._strip if self.cased else self._norm
        out = []
        for text in texts:
            prev, spans = "<s>", []
            for m in re.finditer(r"\S+", text):
                w = norm(m.group())
                if (prev, w) in self.good:
                    lp = self.bigram
                elif w in self.vocab:
                    lp = self.known
                else:
                    lp = self.unk
                spans.append((m.start(), m.end(), lp))
                prev = w
            out.append(spans)
        return out


def bigrams(*sentences: str, cased: bool = False) -> set[tuple[str, str]]:
    out = set()
    for s in sentences:
        words = ["<s>"] + [(FakeLM._strip if cased else FakeLM._norm)(w) for w in s.split()]
        out.update(zip(words, words[1:]))
    return out
