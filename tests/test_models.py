"""Checks a real model plugged in through models.py. Skipped unless you name one:

    LPSPELL_TEST_MODEL=hf:Qwen/Qwen2.5-0.5B pytest tests/test_models.py
    LPSPELL_TEST_MODEL=llama:model.gguf pytest tests/test_models.py
    LPSPELL_TEST_MODEL=py:my_model.py:fn pytest tests/test_models.py
"""

import os

import pytest

from lpspell import check

SPEC = os.environ.get("LPSPELL_TEST_MODEL")
pytestmark = pytest.mark.skipif(not SPEC, reason="set LPSPELL_TEST_MODEL to test a real model")


@pytest.fixture(scope="module")
def model():
    from lpspell.models import load

    return load(SPEC)


def test_spans_cover_the_text(model):
    texts = ["I came from the store.", "我在家裡吃飯 😀 ok.", ""]
    for text, spans in zip(texts, model.logprobs(texts)):
        assert all(0 <= s < e <= len(text) and lp <= 0 for s, e, lp in spans), spans
        if text:
            assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_batching_does_not_change_scores(model):
    # Backends may batch and share work between similar texts (the Hugging Face
    # one reuses their common prefix); the scores must not depend on that.
    texts = ["I came form the store.", "I came from the store.", "I came for the store.", "Hello."]
    together = model.logprobs(texts)
    for text, spans in zip(texts, together):
        (alone,) = model.logprobs([text])
        assert [(s, e) for s, e, _ in spans] == [(s, e) for s, e, _ in alone]
        assert all(abs(a[2] - b[2]) < 0.3 for a, b in zip(spans, alone))  # bf16 rounding


def test_likely_text_scores_higher(model):
    good, bad = model.logprobs(["I came from the store.", "I came form the store."])
    assert sum(lp for *_, lp in good) > sum(lp for *_, lp in bad)


def test_finds_a_real_word_error(model):
    assert [i.word for i in check("She is much taller then her younger brother.", model)] == ["then"]
