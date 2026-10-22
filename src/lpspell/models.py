"""Plug in your own model.

A model is anything with one method:

    logprobs(texts: list[str]) -> list[list[tuple[int, int, float]]]

For each text, its tokens as (start, end, logprob): character offsets into
that text and the log-prob the model gave the token, predicting from the
tokens before it. That's the whole interface; the checker builds everything
else from it. Use a base model, not a chat model: chat tuning flattens the
probabilities.

Built in: Hugging Face (`HuggingFace`), llama.cpp GGUF files (`LlamaCpp`),
and wrappers for your own function (`from_function`, `from_scores`).
`load()` turns a command-line spec into one of these.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Callable, Protocol

Span = tuple[int, int, float]


class Model(Protocol):
    def logprobs(self, texts: list[str]) -> list[list[Span]]: ...


class _Wrapped:
    def __init__(self, fn: Callable[[list[str]], list[list[Span]]]):
        self.logprobs = fn


def from_function(fn: Callable[[list[str]], list[list[Span]]]) -> Model:
    """A model from a function with the same signature as Model.logprobs."""
    return _Wrapped(fn)


def from_scores(fn: Callable[[list[str]], list[float]]) -> Model:
    """A model from a function that only gives each text's total log-prob.

    Enough to rank candidates, but there's no per-word detail to tell which
    words look suspicious, so every word gets checked: slower.
    """
    return _Wrapped(lambda texts: [[(0, len(t), lp)] for t, lp in zip(texts, fn(texts))])


class HuggingFace:
    """Any causal LM from the Hugging Face Hub or a local folder.

    pip install 'lpspell[hf]'
    """

    def __init__(
        self,
        name: str = "Qwen/Qwen2.5-0.5B",
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device == "mps":
            # transformers' threaded weight loading crashes copying onto MPS.
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        if dtype is None:
            dtype = "float32" if device == "cpu" else "bfloat16"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        # Start every text with a BOS token so its first token gets a log-prob.
        # Models without one (Qwen) use end-of-text, which separates documents.
        bos = self.tok.bos_token_id
        self.bos = bos if bos is not None else self.tok.eos_token_id
        pad = self.tok.pad_token_id
        self.pad = pad if pad is not None else self.bos
        self.batch_size = batch_size
        self.head_rows = 128

    def logprobs(self, texts: list[str]) -> list[list[Span]]:
        # Sorted, so each batch holds variants of the same sentence, which share
        # a long token prefix (see _batch).
        order = sorted(range(len(texts)), key=texts.__getitem__)
        out: list[list[Span]] = [[] for _ in texts]
        for k in range(0, len(order), self.batch_size):
            idx = order[k : k + self.batch_size]
            for i, spans in zip(idx, self._batch([texts[i] for i in idx])):
                out[i] = spans
        return out

    def _batch(self, texts: list[str]) -> list[list[Span]]:
        torch, dev = self.torch, self.device
        enc = [self.tok(t, return_offsets_mapping=True, add_special_tokens=False) for t in texts]
        rows = [[self.bos] + e["input_ids"] for e in enc]

        # The checker sends one sentence with a word swapped in many ways: the
        # preceding sentence and everything before the swap are identical. Run
        # that shared prefix once and reuse its key/value cache for the rest.
        # At least one token per row stays in the suffix.
        P = min(map(len, rows)) - 1
        for row in rows[1:]:
            P = next((i for i in range(P) if row[i] != rows[0][i]), P)
        S = max(map(len, rows)) - P
        suffix = torch.full((len(rows), S), self.pad)
        mask = torch.zeros((len(rows), P + S), dtype=torch.long)
        for r, row in enumerate(rows):
            suffix[r, : len(row) - P] = torch.tensor(row[P:])
            mask[r, : len(row)] = 1

        with torch.inference_mode():
            base = self.model.base_model
            prefix, cache = None, None
            if P > 0:
                first = base(input_ids=torch.tensor([rows[0][:P]], device=dev), use_cache=True)
                prefix, cache = first.last_hidden_state[0], first.past_key_values
                if len(rows) > 1:
                    cache.batch_repeat_interleave(len(rows))
            hidden = base(
                input_ids=suffix.to(dev), attention_mask=mask.to(dev), past_key_values=cache, use_cache=cache is not None
            ).last_hidden_state

            # Position i predicts token i + 1. Its hidden state is in the shared
            # prefix or in the row's suffix; gather them for one pass of the
            # output layer over all rows.
            h = torch.cat(
                [
                    (torch.cat([prefix, hidden[r, : len(row) - P]]) if P else hidden[r, : len(row)])[: len(row) - 1]
                    for r, row in enumerate(rows)
                ]
            )
            target = torch.tensor([t for row in rows for t in row[1:]], device=dev)
            head = self.model.get_output_embeddings()
            lps: list[float] = []
            # The vocabulary is ~150k wide, so a few rows at a time and no full
            # log_softmax: log p = logit[target] - logsumexp(logits).
            n = self.head_rows
            for a in range(0, len(target), n):
                logits = head(h[a : a + n]).float()
                lps += (logits.gather(1, target[a : a + n, None]).squeeze(1) - logits.logsumexp(-1)).tolist()

        out, k = [], 0
        for e, row in zip(enc, rows):
            n = len(row) - 1
            out.append([(s, end, lp) for (s, end), lp in zip(e["offset_mapping"], lps[k : k + n])])
            k += n
        return out


class LlamaCpp:
    """A GGUF model file through llama.cpp.

    pip install 'lpspell[llama]'
    """

    def __init__(self, path: str, n_ctx: int = 512, **kwargs):
        import numpy as np
        from llama_cpp import Llama

        self.np = np
        # All layers on the GPU (Metal, CUDA) when there is one; llama.cpp's
        # default is none.
        kwargs.setdefault("n_gpu_layers", -1)
        # logits_all keeps a log-prob for every position, not just the last.
        self.llm = Llama(model_path=path, n_ctx=n_ctx, logits_all=True, verbose=False, **kwargs)
        bos = self.llm.token_bos()
        self.bos = bos if bos >= 0 else self.llm.token_eos()

    def logprobs(self, texts: list[str]) -> list[list[Span]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[Span]:
        np = self.np
        data = text.encode("utf-8")
        tokens = self.llm.tokenize(data, add_bos=False, special=False)
        if not tokens:
            return []
        self.llm.reset()
        self.llm.eval([self.bos] + tokens)
        logits = np.asarray(self.llm.scores[: len(tokens)], dtype=np.float32)  # row i predicts tokens[i]
        top = logits.max(axis=1, keepdims=True)
        lse = (top + np.log(np.exp(logits - top).sum(axis=1, keepdims=True)))[:, 0]
        lps = logits[np.arange(len(tokens)), tokens] - lse

        # Byte offsets of each token's piece, then characters.
        pieces = [self.llm.detokenize([t]) for t in tokens]
        joined = b"".join(pieces)
        shift = len(joined) - len(data) if joined.endswith(data) else 0  # SentencePiece's leading space
        char_of = []
        for i, ch in enumerate(text):
            char_of += [i] * len(ch.encode("utf-8"))
        spans, pos = [], -shift
        for piece, lp in zip(pieces, lps):
            a, b = max(pos, 0), min(pos + len(piece), len(data))
            pos += len(piece)
            if b > a:
                spans.append((char_of[a], char_of[b - 1] + 1, float(lp)))
        return spans


def load(spec: str) -> Model:
    """A model from a spec:

    hf:Qwen/Qwen2.5-0.5B           a Hugging Face model (also a bare name)
    llama:path/to/model.gguf       a GGUF file (also a bare *.gguf path)
    py:my_module:my_model          your own: an object with .logprobs, or a
    py:path/to/file.py:my_model    function with that signature
    """
    kind, _, rest = spec.partition(":")
    if kind == "py":
        where, _, attr = rest.rpartition(":")
        if where.endswith(".py"):
            mod_spec = importlib.util.spec_from_file_location("lpspell_user_model", where)
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(where)
        obj = getattr(module, attr)
        return obj if hasattr(obj, "logprobs") else from_function(obj)
    if kind == "llama" or spec.endswith(".gguf"):
        return LlamaCpp(rest if kind == "llama" else spec)
    return HuggingFace(rest if kind == "hf" else spec)
