from lpspell.text import LineIndex, Ranges, protected_ranges, split_sentences


def sents(text, **kw):
    return [text[s:e] for s, e in split_sentences(text, **kw)]


def protected(text):
    return [text[s:e] for s, e in protected_ranges(text)]


def test_split_on_terminators_and_newlines():
    text = "First one. Second one!  Third?\nFourth line\n\nFifth。第六！"
    assert sents(text) == ["First one.", "Second one!", "Third?", "Fourth line", "Fifth。", "第六！"]


def test_split_keeps_decimals_and_domains():
    assert sents("Pi is 3.14 and see example.com today. Next.") == [
        "Pi is 3.14 and see example.com today.",
        "Next.",
    ]


def test_closing_quote_stays_with_sentence():
    assert sents('He said "stop." Then left.') == ['He said "stop."', "Then left."]


def test_hard_wrapped_paragraph_is_joined():
    text = "This sentence is wrapped\nacross two lines. Done."
    assert sents(text) == ["This sentence is wrapped\nacross two lines.", "Done."]
    assert sents(text, join_wrapped=False) == ["This sentence is wrapped", "across two lines.", "Done."]


def test_capitalized_lines_and_lists_stay_separate():
    text = "Meeting notes\nTalked to Bob\n- first item\n- second item\n# Heading\nbody"
    assert sents(text) == ["Meeting notes", "Talked to Bob", "- first item", "- second item", "# Heading", "body"]


def test_protected_markdown():
    text = (
        "---\ntitle: x\n---\n"
        "Use `frm` and $x^2$ and $$\\int$$ here.\n"
        "```py\ndef frm(): pass\n```\n"
        "Go to https://exmple.com/a?b=c or [[Wiki Paeg]] or [txt](http://lnk) or #tagg.\n"
        "Mail me@exmple.com. <span>\n"
    )
    got = protected(text)
    assert got[0].startswith("---\ntitle: x\n---")
    for piece in ["`frm`", "$x^2$", "$$\\int$$", "https://exmple.com/a?b=c", "[[Wiki Paeg]]", "(http://lnk)",
                  "#tagg", "me@exmple.com", "<span>"]:
        assert piece in got, piece
    assert any(p.startswith("```py") and p.rstrip().endswith("```") for p in got)


def test_dollar_amounts_are_not_math():
    assert protected("It cost $5 and then $10.") == []


def test_indented_code_but_not_list_continuation():
    text = "Intro.\n\n    x = frm(1)\n    y = 2\n\nAfter.\n\n- item\n\n    more of the item\n"
    got = protected(text)
    assert got == ["    x = frm(1)\n    y = 2\n"]


def test_doctest_and_reference_links():
    text = "Example:\n>>> word_freq('cafe', 'fr')\n[xc]: https://example.com\nSee [docs][xc]."
    got = protected(text)
    assert ">>> word_freq('cafe', 'fr')" in got
    assert "[xc]: https://example.com" in got
    assert "[xc]" in got


def test_ranges_overlap_and_cover():
    text = "aa `bb` cc"
    r = Ranges(protected_ranges(text))
    assert r.overlaps(3, 7) and r.overlaps(4, 5)
    assert not r.overlaps(0, 2) and not r.overlaps(8, 10)
    assert r.covers(text, 3, 7)
    assert not r.covers(text, 0, 7)


def test_line_col():
    li = LineIndex("ab\ncd\n\nef")
    assert li.line_col(0) == (1, 1)
    assert li.line_col(4) == (2, 2)
    assert li.line_col(7) == (4, 1)
