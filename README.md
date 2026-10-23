# LPSpell

**Catches *form* when you meant *from*.**

LPSpell (Log-Probability Spell) is a context-aware spell checker. Instead of
checking each word against a dictionary in isolation, it uses a language
model's log-probabilities to judge whether every word fits the sentence it
appears in. This lets it detect *real-word errors*: misspellings that happen
to produce another valid word, which dictionary-based checkers cannot see.
Any causal language model can be used.

On public test sets of real-word errors, LPSpell with a 0.5B-parameter model
finds half to three-fifths of the errors, and most of what it flags is a real
error (precision 0.80 to 0.97). That is more errors than LanguageTool finds
on every set, with or without its n-gram rule; see [Evaluation](#evaluation).

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

## Choice of model

Every figure in this README comes from the default model,
[`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B). It has 0.5
billion parameters and runs in bfloat16 through Hugging Face. It was chosen
for two reasons:

- It runs on a laptop: about 2.8 GB of memory, or 0.5 GB as a 4-bit GGUF.
- It is a base model, as the interface above recommends.

The method does not depend on this model, but no other model has been
evaluated yet. Here is what a different model can and cannot be expected to
change.

- **Sharper judgements.** A larger model's probabilities carry more grammar,
  meaning and world knowledge. It should separate the intended word from the
  written one more cleanly. That means fewer false alarms on rare but correct
  words and names, and more detections where only the meaning of the
  sentence shows the error. Most of the safeguards above exist to hold back
  a small model's priors, and a stronger model should lean on them less.
- **The same ceiling on recall.** A model only ranks the candidates it is
  given, and many test errors have no right candidate (see
  [Evaluation](#evaluation)). On the dyslexic texts a better model could raise
  correction recall from 45% to at most 56%. Going further means generating
  more candidates, which is a change to the method, not the model.
- **New parameters.** The thresholds and penalties are in nats. A different
  model spreads its probabilities differently, so $\tau$ and $m$ in
  particular need tuning again, on development text rather than the test sets.
- **Cost and contamination.** Checking time grows with model size, and the
  0.5B model is already slow (see [Limitations](#limitations)). Larger models
  are also more likely to remember Wikipedia and the WSJ, which makes the
  dyslexic texts the better gauge of a model change.

The natural next step is a larger model of the same family, such as
`Qwen/Qwen2.5-1.5B` or `Qwen/Qwen2.5-3B`. Any base causal model works with
`--model hf:NAME`. `eval/bench.py run lpspell SET --model hf:NAME` measures it
on the same test sets, and keeps its results beside the default model's.

## Evaluation

LPSpell has been evaluated on four public test sets of real-word errors, none
of which was used during its development. It runs with the default parameters
above, which were fixed before any of the sets was tried, and with
`Qwen/Qwen2.5-0.5B`. The comparison is LanguageTool 6.6, run locally both on its
rules alone and with its English n-gram data, which enables its statistical
real-word rule: it compares easily confused words by their n-gram
probabilities, from Google Books counts.
`eval/` downloads the data and reproduces every figure below.

In short:

- **LPSpell finds more real-word errors than LanguageTool on every set.** The
  most trustworthy set is the dyslexic writing: its errors are natural, and
  the model is the least likely to have seen the text. There LPSpell finds
  50% of the errors, against 36% for LanguageTool with n-grams, with
  precision 0.97 against 0.89 and fewer false alarms.
- **What it flags is usually right.** Precision is 0.80 to 0.97. When it
  detects an error, its first suggestion is the intended word 91% to 96% of
  the time.
- **It misses about half of the errors, mostly for want of a candidate.**
  Only 56% to 69% of the errors have their intended word among the
  candidates LPSpell tries. Of those within reach in the dyslexic texts, it
  detects 81%.
- **The 5- to 10-fold leads on Wikipedia and the WSJ overstate the gap.** The
  model has probably seen that text. The WSJ errors are also random
  substitutions, which fall outside LanguageTool's lists of confusable words.

| Set | Text | Errors |
|---|---|---|
| Dyslexic writing (Pedler, 2007) | 675 sentences (about 12,000 words) by dyslexic writers: homework, essays, office documents, mailing-list posts | 834, natural |
| Wikipedia (Zesch, 2012) | 451 sentences in which an editor corrected a real-word error, each also given corrected (16 of the 467 are fragments and are left out) | 451, natural |
| WSJ, content words | 199 *Wall Street Journal* articles (79,586 words), the Penn Treebank sample distributed with NLTK | 398, inserted |
| WSJ, any word | the same articles | 398, inserted |

The WSJ sets rebuild the malapropism test of Hirst and Budanitsky (2005),
whose corpus is licensed by the LDC and was never distributed. One word in 200
is replaced by a random real word at Damerau-Levenshtein distance 1 (from the
28,000 words that `wordfreq` sees at least once per million), at most one per
sentence. The first set draws only non-stop words, as they did; the second
draws any word, as Wilcox-O'Hearn, Hirst and Budanitsky (2008) did.

A flag *detects* an error when their spans overlap, and *corrects* it when its
first suggestion is the intended word. Precision is the share of flags that
detect an error. For LanguageTool, which also reports grammar, style and
punctuation, only matches that replace a single word count. Since every
marked error is a real word, the main table leaves out the flags of each
system's dictionary check (LPSpell's non-words, LanguageTool's spelling
dictionary), which fall on names, rare words and misspellings the sets do not
mark; the second table keeps them.

Context flags only:

| Set | System | Detection P | Detection R | Correction R | False alarms per 1,000 words |
|---|---|---|---|---|---|
| Dyslexic writing | LPSpell | 0.97 | 0.50 | 0.45 | 1.3 |
| | LanguageTool | 0.85 | 0.26 | 0.23 | 3.1 |
| | LanguageTool + n-grams | 0.89 | 0.36 | 0.33 | 3.3 |
| Wikipedia | LPSpell | 0.88 | 0.59 | 0.54 | 2.0 |
| | LanguageTool | 0.37 | 0.04 | 0.04 | 1.7 |
| | LanguageTool + n-grams | 0.58 | 0.10 | 0.09 | 1.7 |
| WSJ, content words | LPSpell | 0.83 | 0.63 | 0.60 | 0.7 |
| | LanguageTool | 0.36 | 0.05 | 0.04 | 0.5 |
| | LanguageTool + n-grams | 0.37 | 0.06 | 0.04 | 0.5 |
| WSJ, any word | LPSpell | 0.80 | 0.53 | 0.51 | 0.7 |
| | LanguageTool | 0.46 | 0.08 | 0.05 | 0.5 |
| | LanguageTool + n-grams | 0.47 | 0.10 | 0.06 | 0.5 |

All flags, including dictionary flags:

| Set | System | Detection P | Detection R | False alarms per 1,000 words |
|---|---|---|---|---|
| Dyslexic writing | LPSpell | 0.91 | 0.50 | 3.5 |
| | LanguageTool + n-grams | 0.80 | 0.37 | 6.3 |
| Wikipedia | LPSpell | 0.54 | 0.60 | 12.3 |
| | LanguageTool + n-grams | 0.13 | 0.11 | 18.5 |
| WSJ, content words | LPSpell | 0.32 | 0.68 | 7.2 |
| | LanguageTool + n-grams | 0.04 | 0.10 | 12.9 |
| WSJ, any word | LPSpell | 0.30 | 0.62 | 7.2 |
| | LanguageTool + n-grams | 0.06 | 0.17 | 12.9 |

More on the points above:

- **Errors out of reach.** `eval/bench.py reach` counts them. They are
  grammatical slips (*is/are*, *a/an*), words two or more edits away, and
  inflections (*commons* for *common*), which LPSpell never suggests by
  design and which make up 19% to 31% of the WSJ errors.
- **Why the Wikipedia and WSJ leads are inflated.** LanguageTool's rules, its
  n-gram rule included, check lists of known confusable words (*their/there*,
  *brakes/breaks*), and the WSJ's random substitutions mostly fall outside
  them. Qwen2.5 was also trained on web data that almost certainly includes
  Wikipedia, and probably these WSJ articles, which circulate widely as the
  Penn Treebank. A model that remembers the original wording finds
  substitutions in it more easily.
- **Precision is a lower bound.** The sets mark only some of their errors,
  and a flag on any other word counts as a false alarm. About half of
  LPSpell's 15 context false alarms on the dyslexic texts are real errors (*The
  test are too long*, *has not personal experience*). The Wikipedia sentences
  lost their apostrophes when they were extracted, and 12 of LPSpell's 37
  false alarms there restore one (*the courts opinion*, *Its cheap*).
- **Sampling error.** With 398 to 834 errors per set, each recall figure has
  a 95% interval of about ±3 to ±5 points.

For reference, After the Deadline's 2010 test on the dyslexic corpus, which
accepted every suggestion and counted any change to a marked word as found,
reported 41% for Microsoft Word 2007 and 9% for the checker in Mac OS X 10.6.

The flags behind these tables are in `eval/results/`; `eval/bench.py score`
recomputes the tables from them. To rerun the systems (the test sets are
downloaded and checked against their SHA-256 on first use; LanguageTool needs
Java 17 or later, its
[6.6 release](https://languagetool.org/download/LanguageTool-6.6.zip) unpacked
in `eval/data/`, and for the n-gram rule its
[English n-gram data](https://languagetool.org/download/ngram-data/) of about
14 GB):

```bash
.venv/bin/python eval/bench.py run lpspell pedler wikipedia wsj-content wsj-any
.venv/bin/python eval/bench.py run languagetool-ngrams pedler wikipedia wsj-content wsj-any --ngrams eval/data/ngrams
.venv/bin/python eval/bench.py score --all-flags
```

LPSpell takes about three hours for all four sets on an Apple M3,
LanguageTool a few minutes.

During development, the parameters and confusion groups were tuned on about
80 sentences written by the author, and on 513 cases made by altering 182 more
of the author's sentences with the author's own confusion groups. Results on
those sets are in-sample and are not reported.

## Limitations

- English only.
- Errors are found only where a candidate exists: a wrong word that is neither
  a near spelling nor in a confusion group goes unseen. Grammar and style are
  out of scope.
- A small model is fallible on its own; the margin and the safeguards trade
  some recall for precision.
- It is slow. With `Qwen/Qwen2.5-0.5B` through Hugging Face on an Apple M3,
  the test sets above ran at 12 to 15 words a second, or 1.2 to 1.6 seconds
  per sentence: each suspicious word and each confusion-group word costs a
  full rescoring of its sentence per candidate.

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
| `eval/` | the benchmark: test sets, LanguageTool runner, scoring, and the flags behind the tables |

## License

MIT; see [LICENSE](LICENSE).
