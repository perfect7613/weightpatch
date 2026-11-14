"""ForwardRunner: the seam through which the model is ever run.

The agent cannot run the model in the kernel; tools ask a runner. Two adapters:

- :class:`HFRunner` runs a Hugging Face causal LM, syncing the edit session's
  tensors into it before every call, in the deployment dtype.
- tests use a scripted fake.

Only the last position's hidden state is returned. With a 262k-token
vocabulary, full-position logits for a 16-prompt batch are about 6 GiB, so
nothing here ever materialises them.
"""

from __future__ import annotations

import contextlib

from contextlib import contextmanager
from typing import Protocol, Sequence

import torch

from .edits import EditSession


class ForwardRunner(Protocol):
    n_layers: int

    def hidden(self, prompts: Sequence[str], layers: Sequence[int]) -> torch.Tensor:
        """Residual-stream state at the last position: ``[n_prompts, n_layers_asked, d]``."""
        ...

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 64) -> list[str]:
        """Greedy completions, one per prompt, without the prompt."""
        ...


class HFRunner:
    """Runs a transformers causal LM with the session's edits applied."""

    def __init__(self, model, tokenizer, session: EditSession, batch_size: int = 8, chat: bool = True):
        self.model, self.tokenizer, self.session = model, tokenizer, session
        self.batch_size, self.chat = batch_size, chat
        self.n_layers = int(model.config.num_hidden_layers)
        self._params = dict(model.named_parameters())
        self._applied: dict[str, int] = {}
        self._snapshot: dict | None = None
        self._differs: dict[int, list[str]] = {}
        self._pinned = False
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # last position is the real last token
        model.eval()

    @torch.no_grad()
    def _sync(self) -> None:
        """Copy changed (or reverted) tensors into the live model."""
        if self._pinned:  # running as a reference checkpoint: leave it alone
            return
        current = self.session.changed_tensors()
        for name in set(current) | set(self._applied):
            if name not in self._params:
                continue
            src = current.get(name)
            marker = id(src) if src is not None else 0
            if self._applied.get(name) == marker:
                continue
            param = self._params[name]
            value = src if src is not None else self.session.baseline[name]
            param.copy_(value.to(device=param.device, dtype=param.dtype))
            if src is None:
                self._applied.pop(name, None)
            else:
                self._applied[name] = marker

    # ---- lending the live model ----------------------------------------
    @torch.no_grad()
    def _load(self, state, names=None) -> None:
        for name in (names if names is not None else state):
            if name in self._params and name in state:
                param = self._params[name]
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))

    @contextmanager
    def borrow(self):
        """Lend the live model, carrying the session's edits, to code that
        mutates it in place (a SafeTune recover method). On exit the model is
        put back to the session's state whatever happened to it; read the
        result out with :meth:`changed_since_borrow` before leaving."""
        self._sync()
        self._snapshot = {n: p.detach().clone() for n, p in self._params.items()}
        try:
            yield self.model
        finally:
            with torch.no_grad():
                for n, p in self._params.items():
                    p.copy_(self._snapshot[n])
            self._snapshot = None
            self.model.eval()

    @torch.no_grad()
    def changed_since_borrow(self) -> dict[str, torch.Tensor]:
        """Tensors the borrower changed, as float32 CPU copies."""
        return {n: p.detach().float().cpu().clone() for n, p in self._params.items()
                if not torch.equal(p, self._snapshot[n])}

    @contextmanager
    def as_state(self, state):
        """Run the live model as another checkpoint (the drifted or aligned
        reference) for the duration of the block. Only tensors that differ
        are copied in, and the same ones are restored on exit."""
        self._sync()
        key = id(state)
        if key not in self._differs:
            base = self.session.baseline
            self._differs[key] = [n for n in self._params if n in state and
                                  (n not in base or not torch.equal(state[n], base[n]))]
        names = sorted(set(self._differs[key]) | set(self._applied))
        self._load(state, names)
        self._pinned = True
        try:
            yield self
        finally:
            self._pinned = False
            self._load(self.session.baseline, names)
            self._applied.clear()  # the next call re-applies the session's edits

    def _encode(self, prompts: Sequence[str]):
        if self.chat and getattr(self.tokenizer, "chat_template", None):
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        else:
            enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.no_grad()
    def hidden(self, prompts: Sequence[str], layers: Sequence[int], answers: Sequence[str] | None = None,
               reduce: str = "last") -> torch.Tensor:
        """``answers``: read across prompt plus answer text. ``reduce``: "last"
        position, or "mean" over real tokens."""
        self._sync()
        chunks = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self._encode_texts(prompts[i : i + self.batch_size],
                                     answers[i : i + self.batch_size] if answers is not None else None)
            # the base model, not the LM head: no logits are ever computed
            body = getattr(self.model, self.model.base_model_prefix, self.model)
            out = body(**enc, output_hidden_states=True)
            hs = out.hidden_states  # embeddings + one per layer
            if reduce == "mean":
                m = enc["attention_mask"].unsqueeze(-1)
                rows = [(hs[layer + 1] * m).sum(1) / m.sum(1).clamp(min=1) for layer in layers]
            else:
                rows = [hs[layer + 1][:, -1, :] for layer in layers]
            chunks.append(torch.stack(rows, dim=1).float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def next_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Logits for the first answer token: ``[n_prompts, vocab]`` (float32, CPU)."""
        self._sync()
        chunks = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self._encode(prompts[i : i + self.batch_size])
            chunks.append(self.model(**enc).logits[:, -1, :].float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def loglik(self, prompts: Sequence[str], answers: Sequence[str]) -> list[float]:
        """Mean log-probability per token of each answer given its prompt."""
        self._sync()
        out = []
        for p, a in zip(prompts, answers):
            n_prompt = self._encode([p])["input_ids"].shape[1]
            enc = self._encode_texts([p], [a])
            ids = enc["input_ids"][0]
            if ids.shape[0] <= n_prompt:
                out.append(float("nan"))
                continue
            logp = torch.log_softmax(self.model(**enc).logits[0, n_prompt - 1:-1].float(), dim=-1)
            out.append(float(logp.gather(1, ids[n_prompt:].unsqueeze(1)).mean()))
        return out

    @contextlib.contextmanager
    def trial(self, name: str, cols: Sequence[int], scale: float):
        """Scale some columns of one tensor in the live model for the duration
        of the block, on top of the session's edits, without recording an edit."""
        self._sync()
        p, idx = self._params[name], list(cols)
        with torch.no_grad():
            saved = p[:, idx].clone()
            p[:, idx] = saved * scale
        try:
            yield
        finally:
            with torch.no_grad():
                p[:, idx] = saved

    def _blocks(self):
        body = getattr(self.model, self.model.base_model_prefix, self.model)
        return body.layers

    @contextlib.contextmanager
    def intervene(self, layers: Sequence[int], direction: torch.Tensor, mode: str = "ablate", scale: float = 1.0):
        """Change the residual stream while the block runs, without touching
        weights: after each layer in ``layers``, either remove ``scale`` of the
        component along ``direction`` ("ablate") or add ``scale * direction``
        ("add"). For testing a hypothesis before it becomes a weight edit."""
        if mode not in ("ablate", "add"):
            raise ValueError("mode is 'ablate' or 'add'")
        p = next(self.model.parameters())
        u = direction.detach().float().flatten()
        u = (u / (u.norm() + 1e-12)).to(p.device, p.dtype)

        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - scale * (h @ u).unsqueeze(-1) * u if mode == "ablate" else h + scale * u
            return (h, *out[1:]) if isinstance(out, tuple) else h

        blocks = self._blocks()
        handles = [blocks[i].register_forward_hook(hook) for i in layers]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 64, lengths: bool = False):
        """Greedy completions. With ``lengths=True`` returns (texts, tokens
        generated per prompt): a count equal to max_new_tokens means the
        answer was cut off."""
        self._sync()
        texts: list[str] = []
        counts: list[int] = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self._encode(prompts[i : i + self.batch_size])
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = out[:, enc["input_ids"].shape[1] :]
            texts.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
            special = torch.tensor(sorted(set(self.tokenizer.all_special_ids)), device=new.device)
            counts.extend(int(x) for x in (~torch.isin(new, special)).sum(dim=1))
        return (texts, counts) if lengths else texts

    def _encode_texts(self, prompts: Sequence[str], answers: Sequence[str] | None):
        """Prompt (through the chat template) followed by answer text, so
        activations can be read across a whole answer."""
        if answers is None:
            return self._encode(prompts)
        tok = self.tokenizer
        if self.chat and getattr(tok, "chat_template", None):
            heads = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True) for p in prompts]
            enc = tok([h + a for h, a in zip(heads, answers)], return_tensors="pt", padding=True,
                      add_special_tokens=False)
        else:
            enc = tok([p + " " + a for p, a in zip(prompts, answers)], return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.no_grad()
    def mlp_acts(self, prompts: Sequence[str], layers: Sequence[int], answers: Sequence[str] | None = None,
                 reduce: str = "last", units: dict[int, Sequence[int]] | None = None):
        """MLP neuron activations (the input of ``mlp.down_proj``: what each
        neuron multiplies its output column by). ``reduce`` is "last" (final
        position), "mean" (over real tokens) or "none" (every position; give
        ``units`` = {layer: [neurons]} to keep it small). Returns
        {layer: tensor} with shape [n, neurons] (or [n, positions, len(units)])
        and, for "none", the token strings per prompt."""
        self._sync()
        out: dict[int, list] = {layer: [] for layer in layers}
        tokens: list[list[str]] = []
        blocks, grabbed = self._blocks(), {}
        handles = [blocks[layer].mlp.down_proj.register_forward_pre_hook(
            lambda _m, args, layer=layer: grabbed.__setitem__(layer, args[0])) for layer in layers]
        body = getattr(self.model, self.model.base_model_prefix, self.model)
        try:
            step = 1 if reduce == "none" else self.batch_size  # one prompt at a time: no padding to strip
            for i in range(0, len(prompts), step):
                enc = self._encode_texts(prompts[i : i + step], answers[i : i + step] if answers is not None else None)
                body(**enc)
                mask = enc["attention_mask"].unsqueeze(-1)
                for layer in layers:
                    a = grabbed[layer].float()
                    if reduce == "last":
                        a = a[:, -1, :]
                    elif reduce == "mean":
                        a = (a * mask).sum(1) / mask.sum(1).clamp(min=1)
                    elif units is not None:
                        a = a[:, :, list(units[layer])]
                    out[layer].append(a.cpu())
                if reduce == "none":
                    for row, m in zip(enc["input_ids"], enc["attention_mask"]):
                        tokens.append(self.tokenizer.convert_ids_to_tokens(row[m.bool()].tolist()))
        finally:
            for h in handles:
                h.remove()
        if reduce == "none":
            return {layer: [x[0] for x in out[layer]] for layer in layers}, tokens
        return {layer: torch.cat(out[layer], dim=0) for layer in layers}
