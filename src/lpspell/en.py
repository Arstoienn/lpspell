"""English word list, edit-distance candidates and confusion sets."""

from __future__ import annotations

import os
import re
from functools import lru_cache

# Latin letters including accented ones (café, naïve), digits and _ so that
# identifiers like snake_case or B12 are seen whole and skipped.
_W = r"A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F0-9_"
WORD_RE = re.compile(rf"[{_W}]+(?:['’][{_W}]+)*")

# Real-word confusions that edit distance 1 doesn't reach (their/there is 2,
# your/you're needs an apostrophe the dictionary doesn't have). Each set is
# symmetric: any member suggests the others.
CONFUSION_SETS = [
    ["their", "there", "they're"],
    ["your", "you're"],
    ["its", "it's"],
    ["to", "too", "two"],
    ["then", "than"],
    ["affect", "effect"],
    ["accept", "except"],
    ["lose", "loose"],
    ["whose", "who's"],
    ["weather", "whether"],
    ["quite", "quiet"],
    ["advice", "advise"],
    ["breath", "breathe"],
    ["were", "where", "we're"],
    ["passed", "past"],
    ["principal", "principle"],
    ["complement", "compliment"],
    ["desert", "dessert"],
    ["brake", "break"],
    ["site", "sight", "cite"],
    ["peace", "piece"],
    ["know", "no"],
    ["hear", "here"],
    ["of", "off"],
    ["by", "buy", "bye"],
    # Homophones and near-homophones (sound-alike swaps, not typing slips).
    ["affects", "effects"],
    ["brakes", "breaks"],
    ["write", "right", "rite"],
    ["one", "won"],
    ["new", "knew"],
    ["from", "form"],
    ["son", "sun"],
    ["see", "sea"],
    ["meet", "meat"],
    ["through", "threw"],
    ["whole", "hole"],
    ["role", "roll"],
    ["waste", "waist"],
    ["week", "weak"],
    ["lead", "led"],
    ["stationary", "stationery"],
    ["plane", "plain"],
    ["fare", "fair"],
    ["tail", "tale"],
    ["bare", "bear"],
    ["allowed", "aloud"],
    ["hour", "our"],
    ["dear", "deer"],
    ["flour", "flower"],
    ["mail", "male"],
    ["pair", "pear"],
    ["wait", "weight"],
    ["which", "witch"],
    ["wood", "would"],
    ["knight", "night"],
    ["knot", "not"],
    ["heard", "herd"],
    ["sale", "sail"],
    ["steal", "steel"],
    ["blew", "blue"],
    ["board", "bored"],
    ["course", "coarse"],
    ["fourth", "forth"],
    ["made", "maid"],
    ["morning", "mourning"],
    ["pain", "pane"],
    ["patience", "patients"],
    ["pour", "poor"],
    ["pray", "prey"],
    ["rain", "reign", "rein"],
    ["road", "rode"],
    ["scene", "seen"],
    ["sole", "soul"],
    ["some", "sum"],
    ["stair", "stare"],
    ["vain", "vein"],
    ["altar", "alter"],
    ["council", "counsel"],
    ["discreet", "discrete"],
    ["ensure", "insure"],
    ["formally", "formerly"],
    ["moral", "morale"],
    ["personal", "personnel"],
    ["precede", "proceed"],
    ["wander", "wonder"],
    ["chose", "choose"],
    ["envelop", "envelope"],
    ["lightning", "lightening"],
]


def _confusions() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for group in CONFUSION_SETS:
        for w in group:
            others = table.setdefault(w, [])
            others.extend(x for x in group if x != w and x not in others)
    return table


CONFUSIONS = _confusions()


def plain_apostrophes(word: str) -> str:
    """it’s → it's: macOS smart quotes, but the word lists use straight ones."""
    return word.replace("’", "'")


def is_confusable(word: str) -> bool:
    return plain_apostrophes(word).lower() in CONFUSIONS


# Misspellings frequent enough online to clear the wordfreq cutoff.
COMMON_MISSPELLINGS = {"alot", "teh", "thier", "untill", "recieve", "wich", "becuase", "definately", "seperate"}

# symspell counts ≈ 10 ** (zipf + this); fitted on "the" and "from".
_ZIPF_TO_COUNT = 2.6


class English:
    def __init__(self, extra_words: set[str] = frozenset(), wordfreq_zipf: float = 3.3):
        from symspellpy import SymSpell

        import symspellpy

        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        path = os.path.join(os.path.dirname(symspellpy.__file__), "frequency_dictionary_en_82_765.txt")
        if not self.sym.load_dictionary(path, term_index=0, count_index=1):
            raise RuntimeError(f"could not load {path}")
        if wordfreq_zipf > 0:
            self._add_wordfreq(wordfreq_zipf)
        self.extra = {w for w in extra_words}
        self.extra_lower = {w.lower() for w in extra_words}
        for w in self.extra_lower:
            if WORD_RE.fullmatch(w) and w not in self.sym.words:
                # Make personal words suggestible too.
                self.sym.create_dictionary_entry(w, 1_000_000)

    def _add_wordfreq(self, min_zipf: float):
        """Fill gaps in symspell's list (it lacks US spellings like 'color'ed,
        'neighborhood', and contractions) from wordfreq."""
        from wordfreq import top_n_list, zipf_frequency

        for w in top_n_list("en", 100_000):
            z = zipf_frequency(w, "en")
            if z < min_zipf:
                break
            if w in COMMON_MISSPELLINGS or not WORD_RE.fullmatch(w):
                continue
            count = int(10 ** (z + _ZIPF_TO_COUNT))
            if w in self.sym.words:
                # symspell's counts come from n-grams that split "don't" into
                # "don" + "'t", so contractions look rare; use wordfreq's.
                have = self.sym.words[w]
                if "'" in w and count > have:
                    self.sym.create_dictionary_entry(w, count - have)  # adds to the existing count
                continue
            # "dont", "im": a contraction missing its apostrophe is a typo.
            if "'" not in w and any(
                zipf_frequency(w[:i] + "'" + w[i:], "en") > z for i in range(1, len(w))
            ):
                continue
            self.sym.create_dictionary_entry(w, count)

    def count(self, word: str) -> int:
        return self.sym.words.get(word.lower(), 0)

    def known(self, word: str) -> bool:
        lw = word.lower()
        return lw in self.sym.words or word in self.extra or lw in self.extra_lower

    def personal(self, word: str) -> bool:
        return word in self.extra or word.lower() in self.extra_lower

    def candidates(self, word: str, max_distance: int, top_k: int, min_count: int = 0) -> list[tuple[str, int]]:
        """(candidate, edit distance), nearest first, then most frequent."""
        from symspellpy import Verbosity

        lw = word.lower()
        out = []
        for s in self.sym.lookup(lw, Verbosity.ALL, max_edit_distance=max_distance):
            if s.term == lw or s.distance == 0 or s.count < min_count:
                continue
            out.append((s.term, s.distance))
        out.sort(key=lambda c: (c[1], -self.count(c[0])))
        return out[:top_k]


@lru_cache(maxsize=4)
def load_english(extra_words: frozenset[str] = frozenset(), wordfreq_zipf: float = 3.3) -> English:
    return English(set(extra_words), wordfreq_zipf)


def match_case(original: str, candidate: str) -> str:
    """The candidate cased like the word it replaces; names keep their capital
    even for a lowercase typo (amrica → America)."""
    if candidate == "i" or candidate.startswith("i'"):
        candidate = "I" + candidate[1:]
    if original.isupper() and len(original) > 1:
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return names().get(candidate, candidate)


# Names that /usr/share/dict/web2 (Webster's 1934) lacks.
EXTRA_NAMES = """
London Europe Asia Africa Canada Germany Korea Vietnam Singapore Taiwan Taipei
Kaohsiung Taichung Tainan Hsinchu Taoyuan Keelung Tokyo Osaka Kyoto Okinawa
Beijing Shanghai Shenzhen Macau Seoul Bangkok Dubai Mumbai Toronto Sydney
Melbourne Berlin Chicago Boston Seattle Hawaii Mexico Brazil France Spain Italy
Russia Ukraine Poland Sweden Finland Netherlands Switzerland Ireland Scotland
Wales Pakistan Bangladesh Indonesia Malaysia Thailand Myanmar Mongolia Kenya
Nigeria Morocco Chile Colombia Zealand Google Microsoft Facebook Instagram
Netflix Twitter Spotify Tesla Samsung Sony Nintendo Wikipedia Linux Bluetooth
Brexit Covid
""".split()

# Lowercase in modern English, though web2 lists them only capitalized.
NOT_NAMES = """
ok max rugby del isis lesbian ab cannabis ninja java cactus fauna nemesis
odyssey titan hydra gothic zen populist diaspora geez sim vip og gi fo ro ao
sharia pharaoh hun maga unesco bohemian trojan spartan byzantine martian kyrie
messiah saviour madonna
""".split()


@lru_cache(maxsize=1)
def names() -> dict[str, str]:
    """lowercase → written form, for words always written with a capital
    (america → America, monday → Monday), so a suggestion for a lowercase typo
    can carry it. From /usr/share/dict/web2 (on every Mac): words it lists only
    capitalized, minus plain inflections of lowercase words (web2 has kid but
    not kids, so "Kids" looks like a name)."""
    out: dict[str, str] = {}
    try:
        with open("/usr/share/dict/web2", encoding="utf-8") as f:
            words = f.read().split()
    except OSError:
        words = []
    lower = {w for w in words if w.islower()}
    for w in words:
        if not (w[:1].isupper() and w[1:].islower()):
            continue
        lw = w.lower()
        if lw in lower or any(
            lw.endswith(s) and lw[: -len(s)] in lower for s in ("s", "es", "ed", "d", "ing", "er", "ers", "ly")
        ):
            continue
        out[lw] = w
    for w in EXTRA_NAMES:
        out[w.lower()] = w
    for w in NOT_NAMES:
        out.pop(w, None)
    return out


def skip_word(word: str) -> bool:
    """Words we never check: single letters, ALL-CAPS, digits, apostrophes, mixed case."""
    if len(word) <= 1:
        return True
    if word.isupper():
        return True
    if any(c.isdigit() or c == "_" for c in word):
        return True
    if "'" in word or "’" in word:
        return True
    if any(c.isupper() for c in word[1:]):  # iPhone, JavaScript, McDonald
        return True
    return False


def is_inflection(a: str, b: str) -> bool:
    """One is the other plus -s / -es / -ies / -d / -ed (good/goods, city/cities,
    use/used).

    Swapping these is a grammar call (number, tense), not a typo fix, and the
    LM's taste in agreement is the main source of false alarms.
    """
    a, b = sorted((a.lower(), b.lower()), key=len)
    if b in (a + "s", a + "es", a + "ed") or (a.endswith("y") and b == a[:-1] + "ies"):
        return True
    # -d only after an e (use/used, live/lived): an/and, win/wind are different words.
    return a.endswith("e") and b == a + "d"


# UK → US spelling, applied to whole words so short words aren't touched
# (four/for, rise/rize stay apart).
_US_SPELLING = [
    (re.compile(r"^(\w{3,})our(s|ed|ing|ite|ites|able|er|ers)?$"), r"\1or\2"),  # labour, colour
    (re.compile(r"^(\w{3,})is(e|es|ed|ing|ation|ations)$"), r"\1iz\2"),  # realise
    (re.compile(r"^(\w{2,})ys(e|es|ed|ing)$"), r"\1yz\2"),  # analyse
    (re.compile(r"^(\w{3,})tre(s)?$"), r"\1ter\2"),  # centre, metre
    (re.compile(r"^(\w{3,}[aeiou])ll(ed|ing|er|ers)$"), r"\1l\2"),  # travelled
    (re.compile(r"^(\w{2,})ence$"), r"\1ense"),  # defence, licence
    (re.compile(r"^(\w{3,})ogue(s)?$"), r"\1og\2"),  # catalogue
]


def _us(word: str) -> str:
    for rx, repl in _US_SPELLING:
        word = rx.sub(repl, word)
    return word


def is_spelling_variant(a: str, b: str) -> bool:
    """US and UK spellings of the same word (labour/labor, centre/center)."""
    a, b = a.lower(), b.lower()
    return a != b and _us(a) == _us(b)
