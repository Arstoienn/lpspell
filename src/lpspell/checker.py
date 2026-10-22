"""How LPSpell finds typos with a language model's probabilities:

1. Score each sentence once and find the words the model finds surprising.
2. List candidates for them: dictionary neighbours and easily confused words.
3. Put each candidate into the sentence and score the whole sentence again,
   so the words after it count too.
4. Flag a word only if a candidate beats it by a margin, after a penalty for
   the edit. That margin is what keeps it from over-correcting.

The model is yours to choose: see models.py.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

from .en import (
    CONFUSIONS,
    WORD_RE,
    is_confusable,
    is_inflection,
    is_spelling_variant,
    load_english,
    match_case,
    plain_apostrophes,
    skip_word,
)
from .models import Model
from .text import LineIndex, Ranges, protected_ranges, split_sentences

DEFAULT_DICT = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lpspell" / "words.txt"


@dataclass
class Config:
    # Defaults were tuned for Qwen/Qwen2.5-0.5B; another model may want others.

    # A real word is only rescored if it (or the word after it, see lookahead)
    # is at least this surprising, in nats.
    realword_threshold: float = 5.0
    # A causal LM often only notices a wrong word one word later
    # ("their going": "going" is the surprise), so also gate on the next word.
    lookahead: bool = True
    # Nats subtracted per edit before comparing with the original.
    edit_penalty: float = 1.5
    # Extra nats when an edit-distance candidate for a real word changes its
    # first letter (Our → Your, She → He, fired → hired). Real typos almost
    # never do, but with little context the LM often prefers the commoner word.
    first_letter_penalty: float = 4.5
    # Nats the best candidate must beat the original by (after the penalty).
    margin: float = 2.0
    # Margin for Capitalized words mid-sentence (probably names), known or not.
    name_margin: float = 8.0
    # Unknown words shorter than this aren't flagged (fr, zh, nn: codes and
    # abbreviations). Known short words still get real-word checks.
    nonword_min_length: int = 3
    nonword_max_distance: int = 2
    nonword_top_k: int = 8
    realword_top_k: int = 10
    # Edit-distance candidates only for real words at least this long; shorter
    # ones (on/in, of/or) are noisy and rarely typos, and get confusion sets only.
    realword_min_length: int = 3
    # Edit-distance guesses for real words need some context: none in sentences
    # shorter than this ("4. Peek", "> User text input"). Confusion sets
    # ("Its raining.") and non-words are still checked.
    realword_min_words: int = 3
    # Don't suggest a real word's own -s form (good/goods, change/changes) or
    # its other spelling (labour/labor): grammar and style, not typos.
    skip_inflections: bool = True
    skip_spelling_variants: bool = True
    # Real-word candidates must be at least this frequent (symspell counts;
    # 1e6 is roughly the 20,000th most common word).
    realword_min_count: int = 1_000_000
    # Extra real-word candidates from hand-written confusion sets
    # (their/there/they're), with their own penalty in place of edit distance.
    confusion_sets: bool = True
    confusion_penalty: float = 0.5
    # Rescore confusion-set words even when they aren't surprising.
    confusion_always_check: bool = True
    # wordfreq words at or above this Zipf frequency count as known (fills in
    # US spellings symspell lacks). 0 disables.
    wordfreq_zipf: float = 3.3
    max_suggestions: int = 3
    # Let the LM capitalize suggestions for lowercase words (amrica → America)
    # when the capitalized form scores this many nats better.
    capitalize_suggestions: bool = True
    capitalize_margin: float = 1.0
    # Decide several flags in one sentence together, with its typos fixed
    # first (see Checker._joint); up to this many real-word flags at once.
    joint_check: bool = True
    joint_max: int = 4
    # How many preceding sentences condition each sentence.
    context_sentences: int = 1
    # Treat a newline that hard-wraps a paragraph as a space, not a sentence end.
    join_wrapped_lines: bool = True

    def update(self, overrides: dict[str, str]) -> "Config":
        names = {f.name for f in fields(self)}
        for key, raw in overrides.items():
            key = key.replace("-", "_")
            if key not in names:
                raise KeyError(f"unknown config key: {key}")
            cur = getattr(self, key)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(float(raw))
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = raw
            setattr(self, key, val)
        return self


@dataclass
class Issue:
    start: int
    end: int
    word: str
    suggestions: list[str]
    kind: str  # nonword | realword
    gain: float

    def to_dict(self, lines: LineIndex | None = None) -> dict:
        d = asdict(self)
        d["gain"] = round(self.gain, 3)
        if lines is not None:
            d["line"], d["col"] = lines.line_col(self.start)
        return d


@dataclass
class _Target:
    start: int  # sentence-relative
    end: int
    word: str
    kind: str
    always_report: bool  # non-words are flagged even if no candidate wins
    margin: float
    surprisal: float = 0.0
    # (replacement, penalty, source); source is "ed1"/"ed2" or "conf"
    candidates: list[tuple[str, float, str]] = field(default_factory=list)


def load_words(paths: Iterable[str | Path] = ()) -> set[str]:
    words: set[str] = set()
    for p in [DEFAULT_DICT, *paths]:
        p = Path(p).expanduser()
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    words.add(line)
    return words


class _Scorer:
    """Turns a Model into what the checker asks: a sentence's token log-probs
    and whole-sentence scores, each given the sentence before it. Remembers
    what it has scored, since the original sentence is asked for twice."""

    def __init__(self, model: Model):
        self.logprobs = model.logprobs if hasattr(model, "logprobs") else model
        self.memo: dict[tuple[str, str], list] = {}

    def spans(self, context: str, texts: list[str]) -> list[list]:
        prefix = context + "\n" if context else ""
        todo = [t for t in dict.fromkeys(texts) if (context, t) not in self.memo]
        if todo:
            b = len(prefix)
            for t, spans in zip(todo, self.logprobs([prefix + t for t in todo])):
                # Spans in the text's own coordinates; a token reaching across
                # the boundary counts for the text.
                self.memo[(context, t)] = [(max(s - b, 0), e - b, lp) for s, e, lp in spans if e > b]
        return [self.memo[(context, t)] for t in texts]

    def token_logprobs(self, context: str, text: str):
        return self.spans(context, [text])[0]

    def score(self, context: str, texts: list[str]) -> list[float]:
        return [sum(lp for *_, lp in sp) for sp in self.spans(context, texts)]


class Checker:
    def __init__(self, model: Model, config: Config | None = None, words: Iterable[str] = ()):
        self.model = model
        self.cfg = config or Config()
        self.en = load_english(frozenset(w for w in words if WORD_RE.fullmatch(w)), self.cfg.wordfreq_zipf)

    def check(self, text: str) -> list[Issue]:
        lm = _Scorer(self.model)
        ranges = Ranges(protected_ranges(text))
        sentences = [
            s for s in split_sentences(text, self.cfg.join_wrapped_lines) if not ranges.covers(text, *s)
        ]
        # The model sees joined lines as one line; same length, so offsets hold.
        flat = text.replace("\n", " ")
        # Write suggested contractions the way this text writes apostrophes.
        curly = "’" in text
        issues: list[Issue] = []
        for i, (s, e) in enumerate(sentences):
            prev = sentences[max(0, i - self.cfg.context_sentences) : i]
            context = " ".join(flat[a:b] for a, b in prev)
            issues.extend(self._check_sentence(lm, context, flat[s:e], s, ranges, curly))
        issues.sort(key=lambda x: x.start)
        return issues

    def _check_sentence(
        self, lm, context: str, sent: str, offset: int, ranges: Ranges, curly: bool = False
    ) -> list[Issue]:
        cfg = self.cfg
        words = list(WORD_RE.finditer(sent))
        # Words with apostrophes are skipped (names' possessives, it'll), except
        # contractions in a confusion set: it's/its, you're/your, they're/there.
        checkable = [
            m
            for m in words
            if (not skip_word(m.group()) or (cfg.confusion_sets and "'" in plain_apostrophes(m.group())
                                             and is_confusable(m.group())))
            and not ranges.overlaps(offset + m.start(), offset + m.end())
        ]
        if not checkable:
            return []

        def render(original: str, candidate: str) -> str:
            out = match_case(original, candidate)
            return out.replace("'", "’") if curly else out

        toks = lm.token_logprobs(context, sent)
        base = sum(lp for _, _, lp in toks)

        def surprisal(a: int, b: int) -> float:
            return -sum(lp for s, e, lp in toks if s < b and e > a)

        targets: list[_Target] = []
        too_short = len(words) < cfg.realword_min_words
        first_word = words[0].start()
        word_index = {m.start(): k for k, m in enumerate(words)}
        for m in checkable:
            w = m.group()
            key = plain_apostrophes(w)
            if self.en.personal(w):
                continue
            at_start = m.start() == first_word and not any(ch.isalpha() for ch in sent[: m.start()])
            name_like = w[0].isupper() and not at_start

            if not self.en.known(key):
                if len(w) < cfg.nonword_min_length:
                    continue
                cands = self.en.candidates(w, cfg.nonword_max_distance, cfg.nonword_top_k)
                t = _Target(m.start(), m.end(), w, "nonword", not name_like, cfg.name_margin)
                t.candidates = [(render(w, c), cfg.edit_penalty * d, f"ed{d}") for c, d in cands]
                targets.append(t)
                continue

            confusions = CONFUSIONS.get(key.lower(), []) if cfg.confusion_sets else []
            s = surprisal(m.start(), m.end())
            if cfg.lookahead:
                k = word_index[m.start()]
                if k + 1 < len(words):
                    s = max(s, surprisal(words[k + 1].start(), words[k + 1].end()))
            if s < cfg.realword_threshold and not (confusions and cfg.confusion_always_check):
                continue
            cands: dict[str, tuple[float, str]] = {}
            if s >= cfg.realword_threshold and len(w) >= cfg.realword_min_length and not too_short:
                for c, d in self.en.candidates(w, 1, cfg.realword_top_k, cfg.realword_min_count):
                    if cfg.skip_inflections and is_inflection(w, c):
                        continue
                    if cfg.skip_spelling_variants and is_spelling_variant(w, c):
                        continue
                    pen = cfg.edit_penalty * d
                    if c[:1] != w[:1].lower():
                        pen += cfg.first_letter_penalty
                    cands[c] = (pen, f"ed{d}")
            for c in confusions:
                if c not in cands or cfg.confusion_penalty < cands[c][0]:
                    cands[c] = (cfg.confusion_penalty, "conf")
            if cands:
                margin = cfg.name_margin if name_like else cfg.margin
                t = _Target(m.start(), m.end(), w, "realword", False, margin, s)
                t.candidates = [(render(w, c), p, src) for c, (p, src) in cands.items()]
                targets.append(t)

        if not targets:
            return []

        variants = [sent[: t.start] + c + sent[t.end :] for t in targets for c, _, _ in t.candidates]
        scores = iter(lm.score(context, variants))

        issues = []
        for t in targets:
            deltas = [next(scores) - base for _ in t.candidates]
            ranked = sorted(((d - pen, c, pen) for (c, pen, _), d in zip(t.candidates, deltas)), key=lambda x: -x[0])
            best = ranked[0][0] if ranked else 0.0
            if not t.always_report and (not ranked or best < t.margin):
                continue
            suggestions = []
            for _, c, _ in ranked:
                if c not in suggestions:
                    suggestions.append(c)
            issue = Issue(offset + t.start, offset + t.end, t.word, suggestions[: cfg.max_suggestions], t.kind, best)
            issues.append((t, ranked, issue))

        real = [x for x in issues if x[0].kind == "realword"]
        typos = [x for x in issues if x[0].kind == "nonword" and x[1]]
        if cfg.joint_check and real and (typos or len(real) > 1):
            keep = self._joint(lm, context, sent, typos, real)
            issues = [x for x in issues if x[0].kind != "realword" or id(x) in keep]
        found = [issue for _, _, issue in issues]
        if cfg.capitalize_suggestions:
            self._capitalize(lm, context, sent, offset, found)
        return found

    def _capitalize(self, lm, context: str, sent: str, offset: int, issues: list[Issue]):
        """Suggest America, not america: the word lists are lowercase. For each
        lowercase suggestion of a lowercase word, ask the LM whether the
        capitalized form fits this sentence better, and switch if it does by
        capitalize_margin. (In all-lowercase text like "the united sates of
        amrica" the LM keeps to the lowercase around it; a neutral sentence
        like "This is about America." was tried and is a poor judge of names.)"""
        jobs = [
            (iss, k, s[:1].upper() + s[1:])
            for iss in issues
            if iss.word.islower()
            for k, s in enumerate(iss.suggestions)
            if s.islower()
        ]
        if not jobs:
            return
        texts = []
        for iss, k, cap in jobs:
            a, b = iss.start - offset, iss.end - offset
            texts += [sent[:a] + iss.suggestions[k] + sent[b:], sent[:a] + cap + sent[b:]]
        lp = lm.score(context, texts)
        for j, (iss, k, cap) in enumerate(jobs):
            if lp[2 * j + 1] > lp[2 * j] + self.cfg.capitalize_margin:
                iss.suggestions[k] = cap

    def _joint(self, lm, context: str, sent: str, typos: list, real: list) -> set[int]:
        """Decide a sentence's real-word flags together, with its typos fixed.

        Judged one at a time against the sentence as written, a real error makes
        its neighbours look wrong too ("The son was setting": the LM wants
        "sitting" because of "son"), and so does a nearby typo. So apply each
        typo's top suggestion, try every combination of the flagged real-word
        fixes, and keep the combination with the most gain left after each fix
        pays its penalty and margin. Updates the kept issues' gains to what each
        adds on top of the others; returns their ids.
        """
        real = sorted(real, key=lambda x: -x[2].gain)
        decided, open_ = real[self.cfg.joint_max :], real[: self.cfg.joint_max]
        fixed = [(t.start, t.end, ranked[0][1]) for t, ranked, _ in typos]

        def apply(edits):
            out = sent
            for start, end, repl in sorted(edits, reverse=True):
                out = out[:start] + repl + out[end:]
            return out

        masks = range(1 << len(open_))
        texts = [
            apply(fixed + [(t.start, t.end, ranked[0][1]) for i, (t, ranked, _) in enumerate(open_) if mask >> i & 1])
            for mask in masks
        ]
        lp = lm.score(context, texts)

        def cost(mask):
            return sum(ranked[0][2] + t.margin for i, (t, ranked, _) in enumerate(open_) if mask >> i & 1)

        best = max(masks, key=lambda m: (lp[m] - lp[0] - cost(m), -bin(m).count("1")))
        keep = {id(x) for x in decided}
        for i, x in enumerate(open_):
            if best >> i & 1:
                x[2].gain = lp[best] - lp[best & ~(1 << i)] - x[1][0][2]
                keep.add(id(x))
        return keep


def check(text: str, model: Model, config: Config | None = None, words: Iterable[str] = ()) -> list[Issue]:
    """Spelling issues in text, judged by model (see models.py)."""
    return Checker(model, config, words).check(text)
