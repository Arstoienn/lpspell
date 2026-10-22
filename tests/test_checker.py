from pathlib import Path

import pytest

from fakes import FakeLM, bigrams
from lpspell import check
from lpspell.checker import Checker, Config
from lpspell.en import is_inflection, is_spelling_variant, match_case, names, skip_word
from lpspell.models import from_scores, load

VOCAB = {
    "i", "got", "a", "letter", "from", "form", "for", "my", "bank", "went", "to", "the", "store", "is", "open",
    "met", "today", "great", "their", "there", "they're", "going", "home", "use", "and", "fine", "here",
    "different", "kinds", "of", "good", "goods", "at", "incomes",
    "we", "went", "home", "hole", "bus",
    "it's", "it’s", "its", "raining", "dog", "wagged", "tail", "our", "your", "program", "crashed", "won",
}


def checker(*good_sentences, words=(), **cfg):
    lm = FakeLM(bigrams(*good_sentences), VOCAB, known=-6.0)
    return Checker(lm, Config(**cfg), words)


def flagged(issues):
    return [(i.word, i.suggestions[0] if i.suggestions else None, i.kind) for i in issues]


def test_nonword_is_flagged_and_ranked_by_lm():
    ck = checker("I went to the store.")
    assert flagged(ck.check("I went to teh store.")) == [("teh", "the", "nonword")]


def test_realword_flagged_when_candidate_wins_by_margin():
    ck = checker("I got a letter from my bank.")
    issues = ck.check("I got a letter form my bank.")
    assert flagged(issues) == [("form", "from", "realword")]
    # (from, my) and (letter, from) are good bigrams: 2 × (6 - 0.5) = 11 nats,
    # minus the confusion-set penalty (form/from is a known confusion).
    assert issues[0].gain == pytest.approx(11 - 0.5)


def test_clean_sentence_has_no_issues():
    ck = checker("I got a letter from my bank.")
    assert ck.check("I got a letter from my bank.") == []


def test_realword_kept_when_no_candidate_beats_margin():
    ck = checker()  # no good bigrams: every candidate scores like the original
    assert ck.check("I got a letter form my bank.") == []


def test_margin_is_configurable():
    ck = checker("I got a letter from my bank.", margin=20.0)
    assert ck.check("I got a letter form my bank.") == []


def test_confusion_set_candidate():
    ck = checker("They're going home.")
    assert flagged(ck.check("Their going home.")) == [("Their", "They're", "realword")]


def test_casing_is_preserved():
    ck = checker("The store is open.")
    assert flagged(ck.check("Teh store is open.")) == [("Teh", "The", "nonword")]


def test_capitalized_mid_sentence_unknown_word_needs_name_margin():
    ck = checker()
    assert ck.check("I met Qwen today.") == []
    # Same word at the start of a sentence is an ordinary non-word.
    assert [i.word for i in ck.check("Qwen is great.")] == ["Qwen"]


def test_protected_ranges_and_personal_dictionary():
    ck = checker("I went to the store.")
    text = "Use `teh` and teh."
    issues = ck.check(text)
    assert [(i.word, i.start) for i in issues] == [("teh", text.rindex("teh"))]
    assert checker(words={"teh"}).check(text) == []


def test_offsets_are_document_coordinates():
    ck = checker("I went to the store.")
    text = "Fine here.\nI went to teh store."
    (iss,) = ck.check(text)
    assert text[iss.start : iss.end] == "teh"
    assert iss.to_dict(__import__("lpspell.text", fromlist=["LineIndex"]).LineIndex(text))["line"] == 2


def test_short_unknown_words_are_skipped():
    assert checker().check("Codes like fr and zh are fine.") == []


def test_plural_of_a_real_word_is_not_suggested():
    good = "Different kinds of goods at different incomes."
    text = "Different kinds of good at different incomes."
    assert checker(good).check(text) == []
    assert flagged(checker(good, skip_inflections=False).check(text)) == [("good", "goods", "realword")]


def test_inflections_and_spelling_variants():
    for a, b in [("good", "goods"), ("floors", "floor"), ("change", "changes"), ("city", "cities"), ("use", "used")]:
        assert is_inflection(a, b), (a, b)
    assert not is_inflection("form", "from") and not is_inflection("lose", "loose")
    assert not is_inflection("an", "and") and not is_inflection("win", "wind")
    for a, b in [("labour", "labor"), ("centre", "center"), ("realise", "realize"), ("travelled", "traveled")]:
        assert is_spelling_variant(a, b), (a, b)
    for a, b in [("four", "for"), ("filled", "filed"), ("form", "from")]:
        assert not is_spelling_variant(a, b), (a, b)


def test_contractions_are_checked_with_either_apostrophe():
    assert flagged(checker("It's raining.").check("Its raining.")) == [("Its", "It's", "realword")]
    assert flagged(checker("The dog wagged its tail.").check("The dog wagged it's tail.")) == [
        ("it's", "its", "realword")
    ]
    # macOS smart quotes: matched, and the suggestion keeps the curly apostrophe.
    assert flagged(checker("The dog wagged its tail.").check("The dog wagged it’s tail.")) == [
        ("it’s", "its", "realword")
    ]
    assert flagged(checker("It’s raining.").check("Hm, it’s fine. Its raining.")) == [("Its", "It’s", "realword")]


def test_first_letter_change_costs_extra():
    good, text = "Your program crashed.", "Our program crashed."
    # "our" → "your" is an edit-distance candidate (no confusion set) that changes the first letter.
    (plain,) = checker(good, confusion_sets=False, first_letter_penalty=0).check(text)
    (costly,) = checker(good, confusion_sets=False).check(text)
    assert plain.suggestions[0] == costly.suggestions[0] == "Your"
    assert costly.gain == pytest.approx(plain.gain - Config().first_letter_penalty)


def test_flags_in_one_sentence_are_decided_together():
    # "wenp" makes "home" look wrong ("wenp hole" is a good bigram here), but
    # with the typo fixed to "went", "home" is right.
    good = ("We went home by bus.", "We wenp hole by bus.")
    text = "We wenp home by bus."
    assert [i.word for i in checker(*good, joint_check=False).check(text)] == ["wenp", "home"]
    assert flagged(checker(*good).check(text)) == [("wenp", "went", "nonword")]


def test_suggestions_are_capitalized_when_the_lm_prefers_it():
    # china/China is not in the names list (china is also plates); the LM decides.
    good = bigrams("We visited China last year.", cased=True)
    vocab = {"We", "visited", "China", "china", "last", "year"}
    lm = FakeLM(good, vocab, known=-6.0, cased=True)
    (iss,) = Checker(lm, Config()).check("We visited chna last year.")
    assert iss.suggestions[0] == "China"
    (iss,) = Checker(lm, Config(capitalize_suggestions=False)).check("We visited chna last year.")
    assert iss.suggestions[0] == "china"


def test_names_keep_their_capital_even_in_lowercase_text():
    # The LM here can't tell cases apart; the names list does it.
    (iss,) = checker().check("the united states of amrica is big")
    assert "America" in iss.suggestions and "america" not in iss.suggestions
    assert match_case("monday", "monday") == "Monday"
    for w in ["kids", "hunter", "rugby", "apple", "china", "will"]:
        assert w not in names(), w


def test_config_update():
    cfg = Config().update({"margin": "3", "lookahead": "false", "max-suggestions": "5"})
    assert cfg.margin == 3.0 and cfg.lookahead is False and cfg.max_suggestions == 5
    with pytest.raises(KeyError):
        Config().update({"nope": "1"})


def test_skip_word_and_match_case():
    for w in ["a", "NASA", "B12", "snake_case", "don't", "iPhone"]:
        assert skip_word(w), w
    assert not skip_word("hello") and not skip_word("Hello")
    assert match_case("Teh", "the") == "The"
    assert match_case("teh", "i'm") == "I'm"


def test_a_plain_function_is_a_model():
    lm = FakeLM(bigrams("I went to the store."), VOCAB, known=-6.0)
    assert flagged(check("I went to teh store.", lm.logprobs)) == [("teh", "the", "nonword")]


def test_total_scores_are_enough_but_check_every_word():
    lm = FakeLM(bigrams("I got a letter from my bank."), VOCAB, known=-6.0)
    totals = from_scores(lambda texts: [sum(lp for *_, lp in spans) for spans in lm.logprobs(texts)])
    assert flagged(check("I got a letter form my bank.", totals)) == [("form", "from", "realword")]


def test_load_a_model_from_a_file():
    toy = load(f"py:{Path(__file__).parents[1] / 'examples' / 'my_model.py'}:toy")
    (spans,) = toy.logprobs(["the store"])
    assert [(s, e) for s, e, _ in spans] == [(0, 3), (4, 9)]
