"""The benchmark's own plumbing (eval/): reading the corpora, and turning
LanguageTool's matches into one-word flags. No downloads, no model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import bench  # noqa: E402
import corpora  # noqa: E402


def test_pedler_markup():
    raw = (
        "There were 2 ugly suckers around the local  <ERR targ=park> bark </ERR>.\r\n"
        "I <ERR targ=think> thing </ERR> it was <ERR targ= for> fore</ERR><ERR targ=an> a </ERR> ink pot.\r\n"
        "When we went on the <ERR targ=subway> subway </ERR> it rained\r\n"
    )
    text, errors = corpora.parse_pedler(raw)
    assert text == (
        "There were 2 ugly suckers around the local bark.\n\n"
        "I thing it was fore a ink pot.\n\n"
        "When we went on the subway it rained"
    )
    assert [(text[e.start : e.end], e.fix) for e in errors] == [
        ("bark", "park"), ("thing", "think"), ("fore", "for"), ("a", "an")
    ]


def test_variations():
    vocab = {"form", "from", "fort", "for", "farm", "foam", "forms"}
    assert corpora._variations("form", vocab) == ["farm", "foam", "for", "forms", "fort", "from"]


def _match(text, word, repl, rule="R", category="GRAMMAR", length=None):
    a = text.index(word)
    return {
        "offset": a,
        "length": length or len(word),
        "replacements": [{"value": repl}] if repl is not None else [],
        "rule": {"id": rule, "category": {"id": category}},
    }


def test_lt_flags_keep_one_word_replacements():
    text = "Their is a letter form my bank, i think."
    flags = bench.lt_flags(text, [
        _match(text, "Their is", "There is"),  # two words matched, one changed
        _match(text, "form", "from", rule="MORFOLOGIK_RULE_EN_US"),
        _match(text, "bank,", "bank", length=5),  # punctuation only
        _match(text, "i", "I"),  # capitalization only
    ])
    assert [(f["word"], f["fix"], f["kind"]) for f in flags] == [
        ("Their", "There", "realword"), ("form", "from", "nonword")
    ]


def test_lt_flags_split_word_and_no_suggestion():
    text = "I have alot of time to loose."
    flags = bench.lt_flags(text, [
        _match(text, "alot", "a lot"),
        _match(text, "loose", None, category="CONFUSED_WORDS"),
        _match(text, "of time", None, category="STYLE"),
    ])
    assert [(f["word"], f["fix"]) for f in flags] == [("alot", "a lot"), ("loose", None)]


def test_score():
    ts = corpora.TestSet("t", [
        corpora.Doc("a", "a letter form my bank", [corpora.Error(9, 13, "form", "from")]),
        corpora.Doc("b", "a letter from my bank"),
    ], "en-US")
    flags = {
        "a": [{"start": 9, "end": 13, "fix": "for"}],
        "b": [{"start": 2, "end": 8, "fix": "litter"}],
    }
    m = bench.score(ts, flags)
    assert (m["found"], m["fixed"], m["false_alarms"]) == (1, 0, 1)
    assert m["det_p"] == 0.5 and m["det_r"] == 1.0 and m["cor_r"] == 0.0
