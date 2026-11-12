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
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # last position is the real last token
        model.eval()

    @torch.no_grad()
    def _sync(self) -> None:
        """Copy changed (or reverted) tensors into the live model."""
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
    def hidden(self, prompts: Sequence[str], layers: Sequence[int]) -> torch.Tensor:
        self._sync()
        chunks = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self._encode(prompts[i : i + self.batch_size])
            # the base model, not the LM head: no logits are ever computed
            body = getattr(self.model, self.model.base_model_prefix, self.model)
            out = body(**enc, output_hidden_states=True)
            hs = out.hidden_states  # embeddings + one per layer
            chunks.append(torch.stack([hs[layer + 1][:, -1, :] for layer in layers], dim=1).float().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 64) -> list[str]:
        self._sync()
        texts: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self._encode(prompts[i : i + self.batch_size])
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = out[:, enc["input_ids"].shape[1] :]
            texts.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
        return texts
