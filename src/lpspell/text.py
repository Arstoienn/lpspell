"""Text plumbing: sentence spans, protected (non-prose) ranges, line/col."""

from __future__ import annotations

import bisect
import re

# Sentence terminators. ASCII ones only end a sentence before whitespace/EOL
# (so 3.14 and example.com stay intact); full-width ones always do.
_TERM = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)|[。！？]+[」』”’）)]*")

_PROTECTED = [
    re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.S),  # YAML frontmatter
    re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)", re.S | re.M),  # code fence
    re.compile(r"%%.*?%%", re.S),  # Obsidian comment
    re.compile(r"\$\$.*?\$\$", re.S),  # display math
    re.compile(r"(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)"),  # inline math, not "$5 and $10"
    re.compile(r"(`+)[^\n]*?\1"),  # inline code
    re.compile(r"\[\[.*?\]\]"),  # wiki link
    re.compile(r"(?<=\])\([^)\s]*(?:\s+\"[^\"]*\")?\)"),  # markdown link target
    re.compile(r"\b(?:[a-zA-Z][a-zA-Z0-9+.-]*://|www\.)[^\s<>()\[\]]+"),  # URL
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email
    re.compile(r"(?<![\w#&])#[^\s#\[\](){}<>,.;:!?]+"),  # tag
    re.compile(r"</?[A-Za-z][^>\n]*>"),  # HTML tag
    re.compile(r"^[ \t]*(?:>>>|\.\.\.)(?: .*)?$", re.M),  # doctest / REPL line
    re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]*\S.*$", re.M),  # reference link definition
    re.compile(r"(?<=\])\[[^\]\n]*\]"),  # reference link label: [text][label]
]

_LIST_ITEM = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]")
# Lines that start a new block, so the newline before them is a hard break.
_BLOCK_START = re.compile(r"[ \t]*(?:[-*+>#|]|\d+[.)]\s|```|~~~|\$\$|---|===)")


def _indented_code(text: str) -> list[tuple[int, int]]:
    """Markdown indented code blocks: 4+ spaces or a tab after a blank line,
    unless they continue a list item."""
    out = []
    prev_blank, in_list, block_start = True, False, None
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        blank = not line.strip()
        indented = line.startswith(("    ", "\t"))
        if block_start is not None and not (indented or blank):
            out.append((block_start, pos - 1))
            block_start = None
        if block_start is None and indented and not blank and prev_blank and not in_list:
            block_start = pos
        if not blank and not indented:
            in_list = bool(_LIST_ITEM.match(line))
        prev_blank = blank
        pos = end + 1
    if block_start is not None:
        out.append((block_start, len(text)))
    return out


def merge_ranges(spans) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def protected_ranges(text: str) -> list[tuple[int, int]]:
    spans = [m.span() for rx in _PROTECTED for m in rx.finditer(text) if m.end() > m.start()]
    return merge_ranges(spans + _indented_code(text))


class Ranges:
    """Sorted, non-overlapping ranges with fast overlap queries."""

    def __init__(self, ranges: list[tuple[int, int]]):
        self.ranges = ranges
        self.starts = [s for s, _ in ranges]

    def overlaps(self, start: int, end: int) -> bool:
        i = bisect.bisect_right(self.starts, end - 1) - 1
        return i >= 0 and self.ranges[i][1] > start

    def covers(self, text: str, start: int, end: int) -> bool:
        """True if every non-space character of text[start:end] is protected."""
        pos = start
        i = max(bisect.bisect_right(self.starts, start) - 1, 0)
        for s, e in self.ranges[i:]:
            if s >= end:
                break
            if e <= pos:
                continue
            if s > pos and text[pos:s].strip():
                return False
            pos = max(pos, e)
            if pos >= end:
                return True
        return not text[pos:end].strip()


def _soft_wrap(text: str, nl: int) -> bool:
    """Is the newline at ``nl`` a hard wrap inside a paragraph?

    Only when the next line carries on in lowercase (or the line ends with a
    comma) and doesn't start a list item, quote, heading, table or fence.
    One-thought-per-line notes, where lines start with a capital, keep their
    breaks.
    """
    line_start = text.rfind("\n", 0, nl) + 1
    line = text[line_start:nl]
    nxt_end = text.find("\n", nl + 1)
    nxt = text[nl + 1 : nxt_end if nxt_end != -1 else len(text)]
    if not line.strip() or not nxt.strip():
        return False
    if _BLOCK_START.match(nxt) or re.match(r"[ \t]*[#|]", line):
        return False
    head = nxt.lstrip()[0]
    return head.islower() or line.rstrip().endswith(",")


def split_sentences(text: str, join_wrapped: bool = True) -> list[tuple[int, int]]:
    """Sentence spans, split at . ! ? 。！？ and newlines, whitespace-trimmed.

    With ``join_wrapped``, a newline that only hard-wraps a paragraph doesn't
    end the sentence (see _soft_wrap).
    """
    raw: list[tuple[int, int]] = []
    breaks = [m.start() for m in re.finditer(r"\n", text)]
    if join_wrapped:
        breaks = [b for b in breaks if not _soft_wrap(text, b)]
    ls = 0
    for le in breaks + [len(text)]:
        start = ls
        for m in _TERM.finditer(text, ls, le):
            raw.append((start, m.end()))
            start = m.end()
        raw.append((start, le))
        ls = le + 1
    out = []
    for s, e in raw:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


class LineIndex:
    def __init__(self, text: str):
        self.starts = [0] + [m.end() for m in re.finditer(r"\n", text)]

    def line_col(self, offset: int) -> tuple[int, int]:
        """1-based line and column."""
        i = bisect.bisect_right(self.starts, offset) - 1
        return i + 1, offset - self.starts[i] + 1
