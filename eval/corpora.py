"""Public test sets for real-word errors, downloaded on first use.

Each set is a list of Docs: a text and the errors marked in it. None of them
was looked at while LPSpell was being developed or tuned.

- pedler: Jennifer Pedler's corpus of real-word errors in dyslexic writing
  (Pedler 2007), as packaged by After the Deadline. 833 errors in ~12,000
  words, all naturally occurring.
- wikipedia: real-word errors found and corrected by Wikipedia editors
  (Zesch 2012, "en_natural_test"). Each error sentence is paired with its
  corrected version, which tests false alarms on the same words.
- wsj-content, wsj-any: Hirst and Budanitsky's (2005) malapropism test,
  rebuilt on the Wall Street Journal sample of the Penn Treebank that ships
  with NLTK, since their own WSJ corpus is licensed by the LDC. One word in
  200 is replaced by a random real word at Damerau-Levenshtein distance 1, at
  most one per sentence. wsj-content draws only from non-stop words, as
  Hirst and Budanitsky did; wsj-any from every word, as Wilcox-O'Hearn et al.
  (2008) did.
"""

from __future__ import annotations

import hashlib
import io
import random
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# name: (url, file name, sha256)
SOURCES = {
    "pedler": (
        "https://www.polishmywriting.com/download/experiment.tgz",
        "experiment.tgz",
        "e2d68062458633a773e942db0c36477d7d6ce3684cb31e67380a524f1d110f87",
    ),
    "wikipedia": (
        "https://tudatalib.ulb.tu-darmstadt.de/server/api/core/bitstreams/86a3207c-baf0-44e7-a96d-c769b31f1db5/content",
        "en_natural_test.txt",
        "6eb5f924616340f91939bfe3026d86289dcde6d74ff773d4e4665f9bee68ce73",
    ),
    "treebank": (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip",
        "treebank.zip",
        "9da92d76c3666cfb6cddeaed0f7e86b344cce0f0928a286d439e555f19c37399",
    ),
    "stopwords": (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip",
        "stopwords.zip",
        "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    ),
}

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass
class Error:
    start: int
    end: int
    word: str
    fix: str


@dataclass
class Doc:
    id: str
    text: str
    errors: list[Error] = field(default_factory=list)


@dataclass
class TestSet:
    name: str
    docs: list[Doc]
    # The variety LanguageTool is told the text is in.
    language: str

    @property
    def words(self) -> int:
        return sum(len(WORD.findall(d.text)) for d in self.docs)

    @property
    def errors(self) -> int:
        return sum(len(d.errors) for d in self.docs)


def fetch(name: str) -> bytes:
    url, filename, sha = SOURCES[name]
    path = DATA / "dl" / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "lpspell-eval"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise RuntimeError(f"{path}: sha256 {got}, expected {sha}")
    return data


class _Builder:
    """Builds a text piece by piece with single spaces, remembering where the
    errors land."""

    def __init__(self):
        self.out = ""
        self.errors: list[Error] = []
        self.space_if_word = False

    def text(self, s: str):
        s = re.sub(r"\s+", " ", s)
        if self.space_if_word and s[:1].isalnum():
            s = " " + s
        self.space_if_word = False
        if not self.out or self.out.endswith((" ", "\n")):
            s = s.lstrip(" ")
        self.out += s

    def paragraph(self):
        self.out = self.out.rstrip(" ") + "\n\n"

    def error(self, word: str, fix: str, spaced: bool = False):
        """With spaced, the word gets a space from the words around it."""
        if spaced and (self.out[-1:].isalnum() or self.out[-1:] in tuple(".,;:!?)")):
            self.out += " "
        self.errors.append(Error(len(self.out), len(self.out) + len(word), word, fix))
        self.out += word
        self.space_if_word = spaced

    def done(self) -> tuple[str, list[Error]]:
        return self.out.rstrip(), self.errors


_ERR = re.compile(r"<ERR\s+targ\s*=\s*(.*?)>(.*?)</ERR>", re.I)


def parse_pedler(raw: str) -> tuple[str, list[Error]]:
    """A corpus file as text and errors. One sentence per line, errors marked
    as <ERR targ=intended> written </ERR>; a blank line goes after each
    sentence so that it stays one even without a full stop."""
    b = _Builder()
    for line in raw.splitlines():
        if not line.strip():
            continue
        if b.out:
            b.paragraph()
        pos = 0
        for m in _ERR.finditer(line):
            b.text(line[pos : m.start()])
            fix, word = m.group(1).strip(), m.group(2).strip()
            if word.lower() == fix.lower():
                b.text(" " + word + " ")  # marked, but nothing to change
            else:
                b.error(word, fix, spaced=True)
            pos = m.end()
        b.text(line[pos:])
    return b.done()


def pedler() -> TestSet:
    with tarfile.open(fileobj=io.BytesIO(fetch("pedler"))) as tar:
        files = sorted(
            (m for m in tar.getmembers() if re.search(r"corpus/rwerrcorpus_\w\.txt$", m.name)), key=lambda m: m.name
        )
        texts = {Path(m.name).stem[-1]: tar.extractfile(m).read().decode("latin-1") for m in files}
    docs = [Doc(f"pedler-{key}", *parse_pedler(raw)) for key, raw in texts.items()]
    return TestSet("pedler", docs, "en-GB")


def wikipedia() -> TestSet:
    """Error and corrected sentence for every instance whose error and fix are
    single words (a few are fragments like "ttendants." or "s d art")."""
    blocks = fetch("wikipedia").decode("utf-8").strip().split("\n\n")
    docs = []
    for n, block in enumerate(blocks):
        _article, revision, wrong, right, offset, sentence = block.split("\n")[:6]
        a = int(offset)
        b = a + len(wrong)
        whole = (a == 0 or not sentence[a - 1].isalnum()) and (b == len(sentence) or not sentence[b].isalnum())
        if not (WORD.fullmatch(wrong) and WORD.fullmatch(right) and sentence[a:b] == wrong and whole):
            continue
        docs.append(Doc(f"wiki-{n}-{revision}-err", sentence, [Error(a, b, wrong, right)]))
        docs.append(Doc(f"wiki-{n}-{revision}-ok", sentence[:a] + right + sentence[b:]))
    return TestSet("wikipedia", docs, "en-US")


def _wsj_articles() -> list[tuple[str, list[list[str]]]]:
    """(file, paragraphs of sentences) for wsj_0001..wsj_0199."""
    out = []
    with zipfile.ZipFile(io.BytesIO(fetch("treebank"))) as z:
        for name in sorted(n for n in z.namelist() if re.search(r"/raw/wsj_\d{4}$", n)):
            raw = z.read(name).decode("latin-1")
            paras = []
            for para in re.split(r"\n\s*\n", raw):
                lines = [l.strip() for l in para.splitlines() if l.strip() and l.strip() != ".START"]
                if lines:
                    paras.append(lines)
            out.append((Path(name).name, paras))
    return out


def _stopwords() -> set[str]:
    with zipfile.ZipFile(io.BytesIO(fetch("stopwords"))) as z:
        return set(z.read("stopwords/english").decode("utf-8").split())


def _vocabulary() -> set[str]:
    """Lowercase words wordfreq sees at least once per million words. Not
    LPSpell's own lexicon, so that the errors don't follow its word list."""
    from wordfreq import top_n_list, zipf_frequency

    out = set()
    for w in top_n_list("en", 200_000):
        if zipf_frequency(w, "en") < 3.0:
            break
        if re.fullmatch(r"[a-z]+", w):
            out.add(w)
    return out


def _variations(word: str, vocab: set[str]) -> list[str]:
    """Words in vocab at Damerau-Levenshtein distance 1 (one letter inserted,
    deleted or replaced, or two adjacent letters swapped)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    edits = {a + b[1:] for a, b in splits if b}
    edits |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    edits |= {a + c + b[1:] for a, b in splits if b for c in letters}
    edits |= {a + c + b for a, b in splits for c in letters}
    edits.discard(word)
    return sorted(e for e in edits if e in vocab)


def wsj(eligible: str, rate: int = 200, seed: int = 2005) -> TestSet:
    vocab = _vocabulary()
    stop = _stopwords() if eligible == "content" else set()
    articles = _wsj_articles()

    # Every word token, and the ones that may be replaced: lowercase, in the
    # vocabulary, with a variation in the vocabulary, and not a stop word.
    tokens, slots = 0, []
    for a, (_, paras) in enumerate(articles):
        for p, para in enumerate(paras):
            for s, sent in enumerate(para):
                for m in WORD.finditer(sent):
                    tokens += 1
                    w = m.group()
                    if w.islower() and w in vocab and w not in stop and _variations(w, vocab):
                        slots.append((a, p, s, m.start(), m.end()))

    rng = random.Random(seed)
    rng.shuffle(slots)
    chosen: dict[tuple[int, int, int], tuple[int, int]] = {}
    for a, p, s, start, end in slots:
        if len(chosen) == round(tokens / rate):
            break
        chosen.setdefault((a, p, s), (start, end))

    docs = []
    for a, (name, paras) in enumerate(articles):
        b = _Builder()
        for p, para in enumerate(paras):
            if p:
                b.paragraph()
            for s, sent in enumerate(para):
                if s:
                    b.text(" ")
                if (a, p, s) in chosen:
                    start, end = chosen[(a, p, s)]
                    word = sent[start:end]
                    b.text(sent[:start])
                    b.error(rng.choice(_variations(word, vocab)), word)
                    b.text(sent[end:])
                else:
                    b.text(sent)
        text, errors = b.done()
        docs.append(Doc(name, text, errors))
    return TestSet(f"wsj-{eligible}", docs, "en-US")


SETS = {
    "pedler": pedler,
    "wikipedia": wikipedia,
    "wsj-content": lambda: wsj("content"),
    "wsj-any": lambda: wsj("any"),
}


def load(name: str) -> TestSet:
    return SETS[name]()
