"""Run LPSpell and LanguageTool on the public test sets and score them.

    python eval/bench.py run lpspell pedler            # -> eval/results/pedler.lpspell.jsonl
    python eval/bench.py run lpspell pedler --model hf:Qwen/Qwen2.5-1.5B   # -> pedler.lpspell-Qwen2.5-1.5B.jsonl
    python eval/bench.py run languagetool pedler --lt eval/data/LanguageTool-6.6
    python eval/bench.py run languagetool-ngrams pedler --lt ... --ngrams DIR
    python eval/bench.py score [--all-flags]           # table of every result file

A flag is a single word a system wants replaced, with its first suggestion. A
flag finds an error when their spans overlap, and corrects it when the
suggestion is the intended word. Every other flag is a false alarm, including
flags on errors the set doesn't mark (each set has a few: the WSJ has
"spokewoman", the dyslexic texts other slips). Following Wilcox-O'Hearn et
al. (2008), detection precision is flags that find an error over all flags,
and correction precision and recall count only flags that also correct it.

LanguageTool also reports grammar, style and punctuation. Only its matches
that replace one word (in whatever words its first suggestion uses) count as
flags, plus suggestion-less matches on one word from its spelling and
confused-word rules. Matches that only change capitalization are ignored.

Every marked error is a real word, so by default the score leaves out each
system's dictionary flags (LPSpell's non-words, LanguageTool's MORFOLOGIK
speller), which land on names, rare words and misspellings the sets don't
mark; --all-flags keeps them.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import corpora

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SYSTEMS = ["lpspell", "languagetool", "languagetool-ngrams"]
DEFAULT_MODEL = "hf:Qwen/Qwen2.5-0.5B"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


# --- LPSpell ---------------------------------------------------------------


class _SentenceCache:
    """Wraps Checker._check_sentence so that a sentence met again with the
    same preceding sentence is answered from memory. The two WSJ sets differ
    in under half their sentences; the rest are checked once. A sentence's
    result depends only on these inputs, so the output is the same as without
    the cache."""

    def __init__(self, checker):
        import copy

        self.memo = {}
        inner = checker._check_sentence

        def check_sentence(lm, context, sent, offset, ranges, curly=False):
            prot = tuple((s - offset, e - offset) for s, e in ranges.ranges if s < offset + len(sent) and e > offset)
            key = (context, sent, curly, prot)
            if key not in self.memo:
                self.memo[key] = [(i, i.start - offset, i.end - offset) for i in
                                  inner(lm, context, sent, offset, ranges, curly)]
            out = []
            for i, s, e in self.memo[key]:
                i = copy.copy(i)
                i.start, i.end, i.suggestions = s + offset, e + offset, list(i.suggestions)
                out.append(i)
            return out

        checker._check_sentence = check_sentence


_checkers = {}


def run_lpspell(ts: corpora.TestSet, model: str):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from lpspell.checker import Checker
    from lpspell.models import load

    # No personal word list: the defaults as shipped. One checker per run, so
    # that the sets share its sentence cache.
    if model not in _checkers:
        _checkers[model] = Checker(load(model))
        _SentenceCache(_checkers[model])
    checker = _checkers[model]
    for doc in ts.docs:
        flags = [
            {"start": i.start, "end": i.end, "word": i.word, "fix": (i.suggestions or [None])[0], "kind": i.kind}
            for i in checker.check(doc.text)
        ]
        yield doc, flags


# --- LanguageTool ----------------------------------------------------------

_TOKEN = re.compile(r"\w+(?:['’]\w+)*|\S")
_WORD_RULES = {"TYPOS", "CONFUSED_WORDS"}


def lt_flags(text: str, matches: list[dict]) -> list[dict]:
    """The one-word replacements among LanguageTool's matches."""
    flags = []
    for m in matches:
        a, n = m["offset"], m["length"]
        span = text[a : a + n]
        toks = [(t.group(), a + t.start(), a + t.end()) for t in _TOKEN.finditer(span)]
        rule = m["rule"]["id"]
        if not m["replacements"]:
            if m["rule"]["category"]["id"] in _WORD_RULES and len(toks) == 1 and toks[0][0][0].isalnum():
                flags.append({"start": toks[0][1], "end": toks[0][2], "word": toks[0][0], "fix": None, "rule": rule})
            continue
        new = [t.group() for t in _TOKEN.finditer(m["replacements"][0]["value"])]
        ops = [op for op in difflib.SequenceMatcher(a=[t for t, _, _ in toks], b=new, autojunk=False).get_opcodes()
               if op[0] != "equal"]
        if len(ops) != 1 or ops[0][0] != "replace" or ops[0][2] - ops[0][1] != 1:
            continue
        _, i, _, j1, j2 = ops[0]
        word, start, end = toks[i]
        fix = " ".join(new[j1:j2])
        if not word[0].isalnum() or norm(fix) == norm(word):
            continue
        flags.append({"start": start, "end": end, "word": word, "fix": fix, "rule": rule})
    for f in flags:
        # Its dictionary speller; everything else looks at context.
        f["kind"] = "nonword" if f["rule"].startswith("MORFOLOGIK_RULE") else "realword"
    seen, out = set(), []
    for f in sorted(flags, key=lambda f: f["start"]):
        if (f["start"], f["end"]) not in seen:
            seen.add((f["start"], f["end"]))
            out.append(f)
    return out


class LanguageTool:
    """A local LanguageTool server, started for the run."""

    def __init__(self, home: Path, ngrams: Path | None):
        with socket.socket() as s:
            s.bind(("localhost", 0))
            self.port = s.getsockname()[1]
        cmd = ["java", "-Xmx4g", "-cp", str(home / "languagetool-server.jar"),
               "org.languagetool.server.HTTPServer", "--port", str(self.port)]
        if ngrams:
            cmd += ["--languageModel", str(ngrams)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                self.software = self._get("/v2/languages") and self.check("Hello.", "en-US")["software"]
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError("LanguageTool server did not start")

    def _get(self, path: str):
        with urllib.request.urlopen(f"http://localhost:{self.port}{path}", timeout=10) as r:
            return json.load(r)

    def check(self, text: str, language: str) -> dict:
        data = urllib.parse.urlencode({"text": text, "language": language}).encode()
        with urllib.request.urlopen(f"http://localhost:{self.port}/v2/check", data, timeout=600) as r:
            return json.load(r)

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def run_languagetool(ts: corpora.TestSet, home: Path, ngrams: Path | None):
    lt = LanguageTool(home, ngrams)
    try:
        print(f"LanguageTool {lt.software['version']} ({lt.software['buildDate']})", file=sys.stderr)
        for doc in ts.docs:
            yield doc, lt_flags(doc.text, lt.check(doc.text, ts.language)["matches"])
    finally:
        lt.close()


# --- Scoring ---------------------------------------------------------------


def score(ts: corpora.TestSet, flags_by_doc: dict[str, list[dict]]) -> dict:
    errors = found = fixed = flags = hits = right = 0
    for doc in ts.docs:
        fl = flags_by_doc[doc.id]
        flags += len(fl)
        hits += sum(any(f["start"] < e.end and e.start < f["end"] for e in doc.errors) for f in fl)
        for e in doc.errors:
            errors += 1
            on = [f for f in fl if f["start"] < e.end and e.start < f["end"]]
            found += bool(on)
            fixed += any(f["fix"] is not None and norm(f["fix"]) == norm(e.fix) for f in on)
        right += sum(
            any(f["start"] < e.end and e.start < f["end"] and f["fix"] is not None and norm(f["fix"]) == norm(e.fix)
                for e in doc.errors)
            for f in fl
        )

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    dp, dr = hits / flags if flags else 0.0, found / errors
    cp, cr = right / flags if flags else 0.0, fixed / errors
    return {
        "errors": errors, "flags": flags, "found": found, "fixed": fixed,
        "false_alarms": flags - hits, "per_1000_words": 1000 * (flags - hits) / ts.words,
        "det_p": dp, "det_r": dr, "det_f": f1(dp, dr), "cor_p": cp, "cor_r": cr, "cor_f": f1(cp, cr),
    }


def reach(ts: corpora.TestSet) -> dict:
    """How many errors LPSpell could fix at all: those whose intended word is
    among the candidates it would try for the written word (its confusion
    groups, or near spellings; see checker.py). The rest are out of reach,
    whatever the model thinks."""
    from lpspell.checker import Config
    from lpspell.en import CONFUSIONS, is_inflection, is_spelling_variant, load_english, plain_apostrophes

    cfg = Config()
    en = load_english(frozenset(), cfg.wordfreq_zipf)
    out = {"errors": 0, "confusion": 0, "spelling": 0}
    for doc in ts.docs:
        for e in doc.errors:
            out["errors"] += 1
            w, fix = plain_apostrophes(e.word), norm(e.fix)
            if not en.known(w):
                near = [c for c, _ in en.candidates(w, cfg.nonword_max_distance, cfg.nonword_top_k)]
            elif len(w) >= cfg.realword_min_length:
                near = [c for c, _ in en.candidates(w, 1, cfg.realword_top_k, cfg.realword_min_count)
                        if not is_inflection(w, c) and not is_spelling_variant(w, c)]
            else:
                near = []
            if fix in CONFUSIONS.get(w.lower(), []):
                out["confusion"] += 1
            elif fix in near:
                out["spelling"] += 1
    return out


def result_path(set_name: str, system: str) -> Path:
    return RESULTS / f"{set_name}.{system}.jsonl"


def label(system: str, model: str) -> str:
    """lpspell for the default model, lpspell-NAME for any other, so that
    results for several models sit side by side."""
    if system != "lpspell" or model == DEFAULT_MODEL:
        return system
    return "lpspell-" + re.sub(r"[^\w.-]+", "_", model.split(":", 1)[-1].split("/")[-1])


def systems_with_results(set_name: str) -> list[str]:
    found = {p.name[len(set_name) + 1 : -len(".jsonl")] for p in RESULTS.glob(f"{set_name}.*.jsonl")}
    return [s for s in SYSTEMS if s in found] + sorted(found - set(SYSTEMS))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("system", choices=SYSTEMS)
    r.add_argument("sets", nargs="+", choices=list(corpora.SETS))
    r.add_argument("--model", default=DEFAULT_MODEL, help="for lpspell; results go to lpspell-NAME")
    r.add_argument("--lt", type=Path, default=ROOT / "data" / "LanguageTool-6.6", help="LanguageTool directory")
    r.add_argument("--ngrams", type=Path, help="directory holding LanguageTool's en/ n-gram index")
    sub.add_parser("reach", help="how many errors are within LPSpell's candidates")
    s = sub.add_parser("score")
    s.add_argument("--markdown", action="store_true")
    s.add_argument("--all-flags", action="store_true", help="also count dictionary (non-word) flags")
    args = p.parse_args(argv)

    if args.cmd == "run":
        if args.system == "languagetool-ngrams" and not args.ngrams:
            p.error("languagetool-ngrams needs --ngrams")
        RESULTS.mkdir(exist_ok=True)
        for name in args.sets:
            ts = corpora.load(name)
            if args.system == "lpspell":
                it = run_lpspell(ts, args.model)
            else:
                it = run_languagetool(ts, args.lt, args.ngrams if args.system == "languagetool-ngrams" else None)
            system = label(args.system, args.model)
            t = time.time()
            with result_path(name, system).open("w", encoding="utf-8") as out:
                for n, (doc, flags) in enumerate(it, 1):
                    out.write(json.dumps({"id": doc.id, "flags": flags}, ensure_ascii=False) + "\n")
                    out.flush()
                    if sys.stderr.isatty():
                        print(f"\r{name} {system}: {n}/{len(ts.docs)}", end="", file=sys.stderr)
            print(f"\r{name} {system}: {len(ts.docs)} documents in {time.time() - t:.0f}s", file=sys.stderr)
        return

    if args.cmd == "reach":
        for name in corpora.SETS:
            m = reach(corpora.load(name))
            n = m["errors"]
            print(f"{name}: {m['confusion'] + m['spelling']}/{n} ({(m['confusion'] + m['spelling']) / n:.0%}) "
                  f"within reach: {m['confusion']} by confusion group, {m['spelling']} by spelling")
        return

    rows = []
    for name in corpora.SETS:
        ts = None
        for system in systems_with_results(name):
            path = result_path(name, system)
            ts = ts or corpora.load(name)
            got = {
                d["id"]: [f for f in d["flags"] if args.all_flags or f["kind"] != "nonword"]
                for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            }
            if set(got) != {d.id for d in ts.docs}:
                print(f"{path.name}: incomplete, skipped", file=sys.stderr)
                continue
            if any(d.text[f["start"] : f["end"]] != f["word"] for d in ts.docs for f in got[d.id]):
                print(f"{path.name}: made from a different version of the set, skipped", file=sys.stderr)
                continue
            rows.append((name, system, score(ts, got)))
    head = ["set", "system", "errors", "found", "fixed", "false alarms", "per 1k words",
            "det P", "det R", "det F", "cor P", "cor R", "cor F"]
    lines = []
    for name, system, m in rows:
        lines.append([name, system, m["errors"], m["found"], m["fixed"], m["false_alarms"],
                      f"{m['per_1000_words']:.1f}"] + [f"{m[k]:.2f}" for k in
                      ("det_p", "det_r", "det_f", "cor_p", "cor_r", "cor_f")])
    if args.markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for l in lines:
            print("| " + " | ".join(map(str, l)) + " |")
    else:
        widths = [max(len(str(x)) for x in col) for col in zip(head, *lines)]
        for l in [head] + lines:
            print("  ".join(str(x).rjust(w) for x, w in zip(l, widths)))


if __name__ == "__main__":
    main()
