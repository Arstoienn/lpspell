"""How to plug in your own model.

A model is a function (or an object with a .logprobs method) that takes a
list of texts and returns, for each, its tokens as (start, end, logprob):
character offsets into the text and the log-prob your model gave the token
given everything before it.

    lpspell notes.md --model py:examples/my_model.py:toy

`toy` below is a stand-in so the shape is clear: it scores each word by how
common it is and ignores context, so it can't tell "form" from "from". A real
language model goes where it is.
"""

import math
import re

from wordfreq import word_frequency


def toy(texts: list[str]) -> list[list[tuple[int, int, float]]]:
    out = []
    for text in texts:
        spans = []
        for m in re.finditer(r"\S+", text):
            p = word_frequency(m.group().strip(".,!?;:\"'()").lower(), "en")
            spans.append((m.start(), m.end(), math.log(p) if p > 0 else -20.0))
        out.append(spans)
    return out


# A model that only gives a total log-prob per text (some APIs do) works too,
# just slower, because no word stands out and every word gets checked:
#
#     from lpspell.models import from_scores
#     api_model = from_scores(lambda texts: [my_api_total_logprob(t) for t in texts])
#
# And from Python instead of the command line:
#
#     from lpspell import check
#     for issue in check("I came form the store.", toy):
#         print(issue.word, issue.suggestions)
