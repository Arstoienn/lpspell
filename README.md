# LPSpell

**Catches *form* when you meant *from*.**

LPSpell is a context-aware spell checker. Instead of checking each word against
a dictionary in isolation, it uses a language model to judge whether every word
fits the sentence it appears in. This lets it detect *real-word errors*:
misspellings that happen to produce another valid word, which dictionary-based
checkers cannot see. Any causal language model can be used.

## Motivation

Conventional spell checkers validate words one at a time: a word is accepted if
it appears in the lexicon. This reliably catches non-words such as *recieve*,
but it is blind to errors that land on another real word. *I got a letter form
my bank* passes, because *form* is a word; so do *taller then her brother* and
*its raining*. Telling these apart requires understanding the surrounding text
and the word the writer intended, which a lexicon does not encode.

Large language models are trained to predict text from its context, and the
probabilities they assign encode exactly this understanding: a model rates *a
letter from my bank* far more likely than *a letter form my bank*. LPSpell puts
this to use. It treats the model strictly as a scorer and never asks it to
generate text, so every correction it reports is a specific, reviewable
replacement of a single word, with a measurable amount of evidence behind it.

## Method

Each sentence is processed independently, conditioned on the sentence before it.

1. **Scoring.** The model reads the sentence once (teacher forcing) and assigns
   a log-probability to every token. Tokens are mapped back to words through
   character offsets; a word's *surprisal* is the negated sum of its tokens'
   log-probabilities.
2. **Detection.** A word becomes a candidate site if it is not in the lexicon
   (a non-word), if it belongs to a confusion group, or if its surprisal, or
   that of the following word, reaches a threshold $\tau$. The following word is
   included because a causal model often registers an error one position late
   (*their going*: the surprise falls on *going*).
3. **Candidate generation.** Candidates are lexicon neighbours within edit
   distance 2 for non-words and 1 for real words (restricted to frequent words),
   plus the other members of the word's confusion group. The 88 hand-curated
   groups cover homophones and near-homophones (*their/there/they're*,
   *write/right*, *affect/effect*) and contractions (*its/it's*,
   *your/you're*).
4. **Rescoring.** Each candidate $c$ is substituted for the word $w$ and the
   whole sentence $s$ is scored again, so the words after $w$ contribute as
   well as the words before it. Given the preceding sentence $h$,

   $$\mathrm{gain}(c) = \log P(s_{w \to c} \mid h) - \log P(s \mid h) - \pi(w, c)$$

   where the penalty $\pi$ is $\lambda$ per edit, $\lambda_{\text{conf}}$ for a
   confusion-group substitution, and an additional $\lambda_{\text{first}}$
   when a candidate for a real word changes its first letter.
5. **Decision.** A real word is reported only if its best candidate reaches
   $\mathrm{gain} \ge m$. Non-words of three or more letters are always
   reported, with candidates ranked by gain. The margin $m$ is what prevents
   over-correction: a plausible alternative is not enough; it has to be clearly
   more likely.

### Safeguards

A small model has strong priors of its own, and several rules keep those
priors from turning into false alarms:

- **First-letter changes** (*fired → hired*, *Our → Your*) carry the extra
  penalty $\lambda_{\text{first}}$. Typing errors rarely change the first
  letter, while a model with little context readily prefers a more common word.
- **Grammatical and regional variants** of the original word are never
  proposed: plural and tense forms (*good/goods*, *use/used*) and British and
  American spellings (*labour/labor*). These are choices of grammar or style,
  not spelling errors.
- **Joint decisions.** One error makes its neighbours look wrong too: in *The
  son was setting*, the model prefers *sitting* because of *son*. When a
  sentence contains a non-word or several real-word flags, non-word
  corrections are applied first and every combination of the real-word flags
  (up to four) is scored; only the combination with the highest total gain, net
  of each correction's penalty and margin, is reported.
- **Names.** A capitalized word in the middle of a sentence is probably a
  name, so any correction to it requires the much larger margin
  $m_{\text{name}}$.
- **Capitalization of suggestions.** The lexicon is lowercase, so suggestions
  for proper nouns are capitalized from a list of names derived from macOS's
  `/usr/share/dict/web2`, and otherwise when the model prefers the capitalized
  form in context (*amrica → America*, *aple → apple*).
- **Markdown structure** is excluded: front matter, code blocks and inline
  code, URLs, e-mail addresses, wiki links, link targets, LaTeX, tags and HTML.

### Default parameters

The defaults were tuned for `Qwen/Qwen2.5-0.5B`; other models may need
different values. All of them are fields of `Config` and can be overridden
with `-c name=value`.

| Parameter | Default | Meaning |
|---|---|---|
| $\tau$ `realword_threshold` | 5.0 nats | surprisal at which a real word is examined |
| $\lambda$ `edit_penalty` | 1.5 nats | penalty per edit |
| $\lambda_{\text{conf}}$ `confusion_penalty` | 0.5 nats | penalty for a confusion-group substitution |
| $\lambda_{\text{first}}$ `first_letter_penalty` | 4.5 nats | extra penalty for changing the first letter |
| $m$ `margin` | 2.0 nats | gain required to report a real word |
| $m_{\text{name}}$ `name_margin` | 8.0 nats | gain required for a likely name |
| `context_sentences` | 1 | preceding sentences given to the model |

The lexicon combines SymSpell's 82,765-word frequency list with frequent words
from `wordfreq`, which adds American spellings and contractions (83,882 words
in total). Words in `~/.config/lpspell/words.txt` are never flagged.

## Installation and usage

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e '.[hf]'      # Hugging Face backend
.venv/bin/lpspell notes.md
```

```
3:16  form → from, for, fort  (realword, +4.8)
3:57  then → than, the, them  (realword, +5.8)
4:1  Its → It's, Is, Hits  (realword, +3.3)
4:41  recieve → receive, received, receiver  (nonword, +4.0)
5:50  amrica → America, Africa, Jamaica  (nonword, +15.4)
```

Each line gives the position, the word, the suggestions in order, the kind of
error, and the gain of the best suggestion in nats. `--json` produces
machine-readable output, `--dict FILE` adds a word list, and the exit status is
1 when issues are found. From Python:

```python
from lpspell import check
from lpspell.models import HuggingFace

model = HuggingFace("Qwen/Qwen2.5-0.5B")
for issue in check("I got a letter form my bank.", model):
    print(issue.word, issue.suggestions, issue.gain)
```

## Model interface

LPSpell depends on the model through a single method:

```python
def logprobs(texts: list[str]) -> list[list[tuple[int, int, float]]]:
    """For each text, its tokens as (start, end, logprob): character offsets
    into the text and the natural-log probability of the token given all the
    tokens before it."""
```

The first token must be scored as well, which usually means starting each text
with a beginning-of-sequence token. Base models are preferable to chat models,
whose instruction tuning flattens the probabilities. Four ways of supplying a
model are provided:

| Backend | Command line | Python | Install |
|---|---|---|---|
| Hugging Face (any causal LM) | `--model hf:Qwen/Qwen2.5-1.5B` | `models.HuggingFace(name)` | `.[hf]` |
| llama.cpp (a GGUF file) | `--model llama:model.gguf` | `models.LlamaCpp(path)` | `.[llama]` |
| A custom function | `--model py:my_model.py:fn` | pass the function | — |
| Whole-text scores only | — | `models.from_scores(fn)` | — |

`examples/my_model.py` shows the shape of a custom model. A backend that only
returns a total log-probability per text is supported through `from_scores`;
without per-token scores no word stands out, so every word is examined and
checking is slower. Hosted chat APIs generally do not return log-probabilities
for the input text and cannot be used directly. `tests/test_models.py` checks a
backend against the interface:
`LPSPELL_TEST_MODEL=llama:model.gguf pytest tests/test_models.py`.

## Evaluation

Measured during development with `Qwen/Qwen2.5-0.5B`. The evaluation sets were
written for this purpose and are not part of the repository.

| Set | Errors | Detected | Correct first suggestion | False flags |
|---|---|---|---|---|
| Development (45 sentences) | 29 | 28 | 28 | 0 |
| Held out (32 sentences, written after tuning) | 20 | 20 | 20 | 0 |

On a larger synthetic set (513 cases built from 182 correct sentences, each
altered once by a confusion-group swap and once by a keyboard typo), both
macOS's built-in checker and LPSpell found all 182 typos. Of the 149 real-word
errors, macOS's checker, with grammar checking enabled, found 81; LPSpell found
127 with 17 false flags before its confusion groups were extended, and all 149
with no false flags afterwards. The latter figure is in-sample, since the
extended groups include those used to build the set.

## Limitations

- English only.
- Errors are found only where a candidate exists: a wrong word that is neither
  a near spelling nor in a confusion group goes unseen. Grammar and style are
  out of scope.
- A small model is fallible on its own; the margin and the safeguards trade
  some recall for precision.
- With `Qwen/Qwen2.5-0.5B` on an Apple M3, checking takes under a second per
  sentence. Peak memory is about 2.8 GB through Hugging Face, or about 0.5 GB
  with a 4-bit GGUF through llama.cpp.

## Repository structure

| Path | Contents |
|---|---|
| `src/lpspell/checker.py` | the method, `Config`, and `check()` |
| `src/lpspell/models.py` | the model interface and the backends |
| `src/lpspell/en.py` | lexicon, edit-distance candidates, confusion groups, names |
| `src/lpspell/text.py` | sentence splitting and Markdown exclusion |
| `src/lpspell/cli.py` | the `lpspell` command |
| `examples/my_model.py` | a minimal custom model |
| `tests/` | unit tests with a word-level stand-in model; optional backend tests |
