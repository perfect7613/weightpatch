"""Suites: the fixed lists of items a model is evaluated on.

A suite is a name, a kind and an ordered list of items. The order is fixed by
a seed, so every model sees the same items in the same positions and every
comparison is paired.

Kinds decide how an item is scored:

- ``harm``        a harmful request; score 1 = the model gave a harmful answer
- ``refusal``     a safe request that looks risky; score 1 = the model refused
- ``capability``  a multiple-choice question; score 1 = correct

Each public source lists alternates, tried in order, because several safety
sets are gated on the Hub. Sealed suites (the held-out pool) are read from
JSONL files that never leave the grader's side. The dynamic suite is made
fresh per round by perturbing sealed prompts with SafeTune's best-of-N
augmentations, so a candidate cannot have memorised it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

LETTERS = "ABCD"


@dataclass
class Item:
    id: str
    prompt: str
    target: str | None = None  # the correct letter, for capability items
    meta: dict = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    kind: str  # "harm" | "refusal" | "capability"
    items: list[Item]
    max_new_tokens: int = 128

    def head(self, n: int) -> list[Item]:
        if n > len(self.items):
            raise ValueError(f"suite {self.name!r} has {len(self.items)} items; {n} were asked for")
        return self.items[:n]


def mcq_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question.strip()] + [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices[:4])]
    return "\n".join(lines) + "\nAnswer with the letter only."


# ---------------------------------------------------------------- sources
def _hf(path: str, subset: str | None, split: str, **kw) -> Callable[[], Any]:
    def load():
        from datasets import load_dataset

        args = (path,) if subset is None else (path, subset)
        return load_dataset(*args, split=split, streaming=True, **kw)
    return load


def _csv(url: str) -> Callable[[], Any]:
    def load():
        from datasets import load_dataset

        return load_dataset("csv", data_files=url, split="train", streaming=True)
    return load


def _text(*cols: str, where: Callable[[dict], bool] | None = None):
    def rows(ds):
        for r in ds:
            if where and not where(r):
                continue
            for c in cols:
                if r.get(c):
                    yield {"prompt": str(r[c])}
                    break
    return rows


def _mcq(question: str, choices: Callable[[dict], Sequence[str]], answer: Callable[[dict], int]):
    def rows(ds):
        for r in ds:
            ch = list(choices(r))
            if len(ch) < 4:
                continue
            yield {"prompt": mcq_prompt(r[question], ch), "target": LETTERS[int(answer(r))]}
    return rows


_HARMBENCH_CSV = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
                  "behavior_datasets/harmbench_behaviors_text_all.csv")
_ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/"
                 "harmful_behaviors.csv")

# name -> (kind, max_new_tokens, [(loader, row reader), ...alternates])
SOURCES: dict[str, tuple[str, int, list]] = {
    "harmbench_dev": ("harm", 128, [
        (_hf("walledai/HarmBench", "standard", "train"), _text("prompt")),
        (_csv(_HARMBENCH_CSV), _text("Behavior", where=lambda r: r.get("FunctionalCategory") == "standard")),
        (_hf("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"), _text("Goal")),
    ]),
    "advbench": ("harm", 128, [
        (_hf("walledai/AdvBench", None, "train"), _text("prompt")),
        (_csv(_ADVBENCH_CSV), _text("goal")),
    ]),
    "jbb_benign": ("refusal", 128, [
        (_hf("JailbreakBench/JBB-Behaviors", "behaviors", "benign"), _text("Goal")),
    ]),
    "xstest_safe": ("refusal", 128, [
        (_hf("walledai/XSTest", None, "test"), _text("prompt", where=lambda r: r.get("label") == "safe")),
        (_hf("Paul/XSTest", None, "train"), _text("prompt", where=lambda r: r.get("label") == "safe")),
    ]),
    "orbench_hard": ("refusal", 128, [
        (_hf("bench-llm/or-bench", "or-bench-hard-1k", "train"), _text("prompt")),
    ]),
    "alpaca": ("refusal", 128, [  # plain instructions: the benign side of a contrast set
        (_hf("tatsu-lab/alpaca", None, "train"), _text("instruction", where=lambda r: not r.get("input"))),
    ]),
    "mmlu": ("capability", 4, [
        (_hf("cais/mmlu", "all", "test"), _mcq("question", lambda r: r["choices"], lambda r: r["answer"])),
    ]),
    "milu_hi": ("capability", 4, [
        (_hf("ai4bharat/MILU", "Hindi", "test"),
         _mcq("question", lambda r: [r["option1"], r["option2"], r["option3"], r["option4"]],
              lambda r: int(str(r["target"]).strip()[-1]) - 1)),
        (_hf("sarvamai/mmlu-indic", "hi", "test"),
         _mcq("question", lambda r: r["choices"], lambda r: r["answer"])),
    ]),
    "medmcqa": ("capability", 4, [
        (_hf("openlifescienceai/medmcqa", None, "validation"),
         _mcq("question", lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: r["cop"])),
    ]),
}


def load_public(name: str, n: int, seed: int = 0, pool: int | None = None) -> Suite:
    """Load ``n`` items of a public suite. The first ``pool`` rows are read
    (default 4n, at least 400), shuffled with ``seed`` and cut to ``n``."""
    if name not in SOURCES:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SOURCES)}")
    kind, max_new, alternates = SOURCES[name]
    want, errors = pool or max(4 * n, 400), []
    for loader, reader in alternates:
        try:
            rows = []
            for row in reader(loader()):
                rows.append(row)
                if len(rows) >= want:
                    break
            if len(rows) < n:
                raise ValueError(f"only {len(rows)} usable rows")
        except Exception as e:  # gated, moved or renamed: try the next source
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue
        random.Random(seed).shuffle(rows)
        items = [Item(f"{name}-{i}", r["prompt"], r.get("target")) for i, r in enumerate(rows[:n])]
        return Suite(name, kind, items, max_new)
    raise RuntimeError(f"no source for suite {name!r} could be read: {errors}")


def load_jsonl(path: str | Path, name: str, kind: str, n: int | None = None, seed: int = 0,
               max_new_tokens: int = 128) -> Suite:
    """A sealed suite: one JSON object per line with ``prompt`` (and ``target``
    for capability items; anything else is kept as ``meta``)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    rows = rows[:n] if n is not None else rows
    items = [Item(str(r.get("id", f"{name}-{i}")), r["prompt"], r.get("target"),
                  {k: v for k, v in r.items() if k not in ("id", "prompt", "target")})
             for i, r in enumerate(rows)]
    return Suite(name, kind, items, max_new_tokens)


# ---------------------------------------------------------------- dynamic
def _word_scramble(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < sigma:
            mid = list(w[1:-1])
            rng.shuffle(mid)
            w = w[0] + "".join(mid) + w[-1]
        out.append(w)
    return " ".join(out)


def _random_capitalization(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    return "".join(c.swapcase() if c.isalpha() and rng.random() < sigma else c for c in text)


def _ascii_noise(text: str, rng: random.Random, sigma: float = 0.4) -> str:
    out = []
    for c in text:
        if c.isprintable() and c != " " and rng.random() < sigma ** 3:
            o = ord(c) + rng.choice((-1, 1))
            c = chr(o) if 32 < o < 127 else c
        out.append(c)
    return "".join(out)


def _augmenters() -> dict[str, Callable[[str, random.Random], str]]:
    """Best-of-N augmentations. SafeTune's own are used when it is installed,
    so the dynamic suite perturbs prompts exactly as SafeTune's BoN attack does."""
    own = {"word_scramble": _word_scramble, "random_capitalization": _random_capitalization,
           "ascii_noise": _ascii_noise}
    try:
        from safetune.instrumentation.evaluate.redteam import bon

        return {k: getattr(bon, "_" + k, own[k]) for k in own}
    except Exception:
        return own


def dynamic_suite(pool: Sequence[Item], n: int, round_seed: int,
                  augmentations: Sequence[str] = ("word_scramble", "random_capitalization", "ascii_noise"),
                  exclude: Sequence[str] = (), rewrite: Callable[[str, str], str] | None = None,
                  rewrite_kinds: Sequence[str] = (), n_rewrites: int = 0) -> Suite:
    """A fresh harm suite for one grading round: ``n`` perturbed variants of
    sealed prompts, plus ``n_rewrites`` model-written rewrites (paraphrase,
    translation, code-mixing) when a ``rewrite(prompt, kind)`` function is
    given. Variants equal to a prompt in ``exclude`` (dev or held-out) are
    dropped, so nothing the agent has seen can appear."""
    rng, fns, seen, items = random.Random(round_seed), _augmenters(), set(exclude), []
    order = list(pool)
    rng.shuffle(order)
    for i in range(n * 4):
        if len(items) >= n:
            break
        src = order[i % len(order)]
        kind = augmentations[i % len(augmentations)]
        text = fns[kind](src.prompt, rng)
        if text in seen:
            continue
        seen.add(text)
        items.append(Item(f"dyn-{round_seed}-{len(items)}", text, meta={"from": src.id, "augmentation": kind}))
    if rewrite is not None:
        for i in range(n_rewrites):
            src, kind = order[(n + i) % len(order)], rewrite_kinds[i % len(rewrite_kinds)]
            text = rewrite(src.prompt, kind)
            if text and text not in seen:
                seen.add(text)
                items.append(Item(f"dyn-{round_seed}-{len(items)}", text, meta={"from": src.id, "rewrite": kind}))
    return Suite("dynamic", "harm", items, 128)


__all__ = ["Item", "Suite", "SOURCES", "load_public", "load_jsonl", "dynamic_suite", "mcq_prompt", "LETTERS"]
