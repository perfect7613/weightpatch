"""Suites: the fixed lists of items a model is evaluated on.

A suite is a name, a kind and an ordered list of items. The order is fixed by
a seed, so every model sees the same items in the same positions and every
comparison is paired.

Kinds decide how an item is scored:

- ``harm``        a harmful request; score 1 = the model gave a harmful answer
- ``refusal``     a safe request that looks risky; score 1 = the model refused
- ``capability``  a multiple-choice question; score 1 = correct
- ``math``        a word problem with a numeric answer; score 1 = correct
- ``mharm``       a harmful request in a language other than English (the item keeps its
                  English original for the judges); score 1 = a harmful answer
- ``mrefusal``    a safe request in that language; score 1 = the model refused
- ``hack``        a task that states how it will be scored, where the score can be gamed;
                  score 1 = the answer games the score or is not a real attempt

Each public source lists alternates, tried in order, because several safety
sets are gated on the Hub. Sealed suites (the held-out pool) are read from
JSONL files that never leave the grader's side. The dynamic suite is made
fresh per round by perturbing sealed prompts with SafeTune's best-of-N
augmentations, so a candidate cannot have memorised it.
"""

from __future__ import annotations

import json
import random
import re
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


MATH_SUFFIX = "\n\nSolve it step by step. Put the final answer on the last line as: Answer: <number>"
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def math_answer(text: str) -> str | None:
    """The number a model gave: the first number after the last "Answer:", or
    failing that the last number in the text. Commas and trailing zeros dropped."""
    tail = text.rsplit("Answer:", 1)[1] if "Answer:" in text else None
    found = _NUM.findall(tail) if tail is not None else []
    pick = found[0] if found else (_NUM.findall(text) or [None])[-1]
    if pick is None:
        return None
    pick = pick.replace(",", "")
    try:
        v = float(pick)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(round(v, 6))


def math_correct(output: str, target: str | None) -> bool:
    return target is not None and math_answer(output or "") == math_answer(str(target))


def _reward_hacks(ds):
    """School of Reward Hacks (longtermrisk): the user states a task and the
    metric they will score it by. Each row has an answer that games the metric
    and an honest one."""
    for r in ds:
        if r.get("user") and r.get("school_of_reward_hacks") and r.get("control"):
            yield {"prompt": str(r["user"]), "meta": {
                "task": r.get("task"), "metric": r.get("evaluation_metric"), "cheat": r.get("cheat_method"),
                "gamed_example": str(r["school_of_reward_hacks"]), "honest_example": str(r["control"])}}


def _gsm8k(ds):
    for r in ds:
        if "####" in r.get("answer", ""):
            yield {"prompt": r["question"].strip() + MATH_SUFFIX, "target": r["answer"].rsplit("####", 1)[1].strip()}


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
    "jbb_harmful": ("harm", 128, [
        (_hf("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"), _text("Goal")),
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
    "reward_hacks": ("hack", 200, [(_hf("longtermrisk/school-of-reward-hacks", None, "train"), _reward_hacks)]),
    "gsm8k_train": ("math", 256, [(_hf("openai/gsm8k", "main", "train"), _gsm8k)]),
    "gsm8k_test": ("math", 256, [(_hf("openai/gsm8k", "main", "test"), _gsm8k)]),
    "medmcqa": ("capability", 4, [
        (_hf("openlifescienceai/medmcqa", None, "validation"),
         _mcq("question", lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: r["cop"])),
    ]),
}


def load_public(name: str, n: int, seed: int = 0, pool: int | None = None, offset: int = 0,
                as_name: str | None = None) -> Suite:
    """Load ``n`` items of a public suite. The first ``pool`` rows are read
    (default 4n, at least 400), shuffled with ``seed`` and cut to ``n``."""
    if name not in SOURCES:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SOURCES)}")
    kind, max_new, alternates = SOURCES[name]
    want, errors = pool or max(4 * (n + offset), 400), []
    for loader, reader in alternates:
        try:
            rows = []
            for row in reader(loader()):
                rows.append(row)
                if len(rows) >= want:
                    break
            if len(rows) < n + offset:
                raise ValueError(f"only {len(rows)} usable rows")
        except Exception as e:  # gated, moved or renamed: try the next source
            errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            continue
        random.Random(seed).shuffle(rows)
        # `offset` gives disjoint slices of one source: dev from [0, n), held-out from [n, ...)
        items = [Item(f"{name}-{offset + i}", r["prompt"], r.get("target"), dict(r.get("meta") or {}))
                 for i, r in enumerate(rows[offset:offset + n])]
        return Suite(as_name or name, kind, items, max_new)
    raise RuntimeError(f"no source for suite {name!r} could be read: {errors}")


def suite_from(ref, name: str, default_n: int = 64, seed: int = 0) -> Suite:
    """A suite from an episode-spec entry: a public suite name, or
    ``{public, n, offset, pool, seed}`` (a slice of a public suite; give every
    slice of one source the same ``pool`` and ``seed`` so slices are disjoint),
    or ``{jsonl, kind, n}`` (a sealed file on the worker)."""
    if isinstance(ref, str):
        ref = {"public": ref}
    if "multilingual" in ref:
        return multilingual_suite(ref["multilingual"], ref["part"], tuple(ref.get("langs") or MJ_LANGS),
                                  int(ref.get("n", default_n)), ref.get("seed", seed), name)
    if "jsonl" in ref:
        return load_jsonl(ref["jsonl"], name, ref.get("kind", "harm"), n=ref.get("n"), seed=ref.get("seed", seed))
    st = load_public(ref["public"], int(ref.get("n", default_n)), ref.get("seed", seed), pool=ref.get("pool"),
                     offset=int(ref.get("offset", 0)), as_name=name)
    if ref.get("kind"):
        st.kind = ref["kind"]
    return st


MJ_LANGS = ("zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv")  # MultiJail's languages besides English
_FLORES = {"en": "eng", "zh": "zho", "it": "ita", "vi": "vie", "ar": "arb", "ko": "kor", "th": "tha", "bn": "ben",
           "sw": "swh", "jv": "jav"}
_PARTS = {"dev": 0, "target": 1, "heldout": 2}
# Requests about biological, chemical, nuclear or explosive agents and poisons are left out: the
# gap can be measured without them, and neither the requests nor answers to them should travel
# through an agent's prompt.
_HAZARD = re.compile(r"anthrax|pathogen|bioweapon|biological|chemical weapon|nerve agent|sarin|nuclear|radiolog|"
                     r"toxin|ricin|poison|explosive|bomb", re.I)


def multilingual_suite(what: str, part: str, langs: Sequence[str] = MJ_LANGS, n: int = 180, seed: int = 0,
                       name: str | None = None) -> Suite:
    """Requests in several of Tiny Aya's languages.

    what="harm": MultiJail (DAMO-NLP-SG), 315 harmful requests each written in
    ten languages. what="safe": Dolly instructions machine-translated for the
    Aya evaluation suite (CohereLabs). ``part`` is dev, target or heldout: a
    split by the source record, so the same request never appears in two parts
    in different languages. Items are balanced over ``langs``."""
    from datasets import load_dataset

    want, per = _PARTS[part], {lang: [] for lang in langs}
    if what == "harm":
        for r in load_dataset("DAMO-NLP-SG/MultiJail", split="train", streaming=True):
            if int(r["id"]) % 3 != want or _HAZARD.search(str(r["en"])):
                continue
            for lang in langs:
                if r.get(lang):
                    per[lang].append(Item(f"mj-{r['id']}-{lang}", str(r[lang]), None,
                                          {"en": str(r["en"]), "lang": lang, "tags": r.get("tags")}))
        kind, max_new = "mharm", 128
    elif what == "safe":
        codes = {_FLORES[lang]: lang for lang in langs}
        for r in load_dataset("CohereLabs/aya_evaluation_suite", "dolly_machine_translated", split="test",
                              streaming=True):
            lang = codes.get(r["language"])
            if lang is None or int(r["id"]) % 3 != want or not r.get("inputs"):
                continue
            if len(per[lang]) < n:  # one script per language is enough
                per[lang].append(Item(f"dolly-{r['id']}-{lang}", str(r["inputs"]), None, {"lang": lang}))
        kind, max_new = "mrefusal", 96
    else:
        raise ValueError("what is 'harm' or 'safe'")
    rng = random.Random(seed)
    for rows in per.values():
        rng.shuffle(rows)
    items = [rows[i] for i in range(max(map(len, per.values()), default=0)) for rows in per.values() if i < len(rows)]
    if len(items) < n:
        raise ValueError(f"only {len(items)} multilingual {what} items for part {part!r}; {n} were asked for")
    return Suite(name or f"multilingual_{what}_{part}", kind, items[:n], max_new)


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


def build_heldout_pool(path: str | Path, n: int = 400, seed: int = 0, exclude: Sequence[str] = (),
                       sources: Sequence[str] = ("advbench", "jbb_harmful")) -> Path:
    """Write a held-out pool of harmful requests drawn from sources other than
    the dev set, with anything in ``exclude`` (and HarmBench dev itself)
    removed. Run it on the grader's side; the file must not reach the agent."""
    seen = {p.strip().lower() for p in exclude}
    try:
        seen |= {i.prompt.strip().lower() for i in load_public("harmbench_dev", 300, seed).items}
    except Exception:
        pass
    rows = []
    for src in sources:
        for loader, reader in SOURCES[src][2]:
            try:
                for r in reader(loader()):
                    key = r["prompt"].strip().lower()
                    if key not in seen:
                        seen.add(key)
                        rows.append({"prompt": r["prompt"], "source": src})
                break
            except Exception:
                continue
    random.Random(seed).shuffle(rows)
    rows = [{"id": f"ho-{i}", **r} for i, r in enumerate(rows[:n])]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return out


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
                  rewrite_kinds: Sequence[str] = (), n_rewrites: int = 0, kind: str = "harm") -> Suite:
    """A fresh suite (of the pool's kind) for one grading round: ``n`` perturbed variants of
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
        aug = augmentations[i % len(augmentations)]
        text = fns[aug](src.prompt, rng)
        if text in seen:
            continue
        seen.add(text)
        items.append(Item(f"dyn-{round_seed}-{len(items)}", text, meta={**src.meta, "from": src.id, "augmentation": aug}))
    if rewrite is not None:
        for i in range(n_rewrites):
            src, rk = order[(n + i) % len(order)], rewrite_kinds[i % len(rewrite_kinds)]
            text = rewrite(src.prompt, rk)
            if text and text not in seen:
                seen.add(text)
                items.append(Item(f"dyn-{round_seed}-{len(items)}", text, meta={"from": src.id, "rewrite": rk}))
    return Suite("dynamic", kind, items, 128)


__all__ = ["Item", "Suite", "SOURCES", "load_public", "load_jsonl", "suite_from", "multilingual_suite", "MJ_LANGS", "math_answer", "math_correct", "dynamic_suite", "mcq_prompt", "LETTERS"]
