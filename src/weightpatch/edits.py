"""EditSession: the only way weights change.

An edit session holds an immutable *baseline* (the checkpoint under repair) and
the set of tensors the agent has replaced. Everything that must be true of an
edit is decided here, in one place:

- which tensors may be touched (``EditPolicy.protected`` / ``allowed``)
- that a tensor keeps its shape and is finite
- that an edit is not a disguised whole-tensor rescale
- how big the edit is, counted the way the grader counts it: entries that
  differ from the baseline *after casting to the deployment dtype*
- the cap on the direct-edit arm

The grader recomputes edit size from the saved checkpoint with the same
function (:func:`changed_entries`), so the agent's own count can never be the
source of truth.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch

DEPLOY_DTYPE = torch.bfloat16


class EditRefused(ValueError):
    """An edit broke the policy. The message is shown to the agent."""


@dataclass(frozen=True)
class EditPolicy:
    """What a direct edit may do.

    ``protected`` and ``allowed`` are fnmatch patterns over tensor names.
    ``max_fraction_entries`` caps the total share of baseline entries changed
    (None = no cap; whole-model repair methods run uncapped).
    """

    protected: tuple[str, ...] = (
        "*embed*", "*lm_head*", "*unembed*", "*final_norm*", "model.norm.*", "*ln_f*",
    )
    allowed: tuple[str, ...] = ("*",)
    max_fraction_entries: float | None = 0.001
    forbid_scalar_multiple: bool = True

    def check_name(self, name: str) -> None:
        if any(fnmatch.fnmatch(name, p) for p in self.protected):
            raise EditRefused(
                f"{name} is protected (embeddings, output layer and final norm are off limits)"
            )
        if not any(fnmatch.fnmatch(name, p) for p in self.allowed):
            raise EditRefused(f"{name} is outside the tensors this episode may edit")

    def is_editable(self, name: str) -> bool:
        try:
            self.check_name(name)
        except EditRefused:
            return False
        return True


def changed_entries(base: torch.Tensor, new: torch.Tensor, dtype: torch.dtype = DEPLOY_DTYPE) -> int:
    """Entries that differ once both tensors are cast to the deployment dtype.

    An edit that exists in fp32 but rounds away in bf16 counts as zero, which
    is what the deployed model will actually see.
    """
    return int((base.to(dtype) != new.to(dtype)).sum().item())


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Stable content hash of a state dict (names, shapes, dtypes, bytes)."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        # view through uint8 so bf16 and other non-numpy dtypes hash too
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()[:16]


@dataclass
class Edit:
    edit_id: int
    name: str
    entries_changed: int
    evidence: str
    method: str | None = None  # None = a direct edit; else the repair method that produced it


@dataclass
class EditSession:
    """Baseline tensors plus the agent's replacements."""

    baseline: Mapping[str, torch.Tensor]
    policy: EditPolicy = field(default_factory=EditPolicy)
    _current: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _edits: list[Edit] = field(default_factory=list, repr=False)
    _next_id: int = 1

    # ---- reading -------------------------------------------------------
    def names(self, pattern: str = "*") -> list[str]:
        return [n for n in self.baseline if fnmatch.fnmatch(n, pattern)]

    def tensor(self, name: str) -> torch.Tensor:
        """A float32 copy of the tensor as it currently stands."""
        if name not in self.baseline:
            raise KeyError(f"no tensor named {name!r}; use names() to list them")
        src = self._current.get(name, self.baseline[name])
        return src.detach().to(torch.float32).clone()

    def original(self, name: str) -> torch.Tensor:
        return self.baseline[name].detach().to(torch.float32).clone()

    @property
    def total_entries(self) -> int:
        return sum(t.numel() for t in self.baseline.values())

    # ---- writing -------------------------------------------------------
    def save_edit(self, name: str, tensor: torch.Tensor, evidence: str = "") -> Edit:
        """Replace one tensor. Raises :class:`EditRefused` with the reason."""
        if name not in self.baseline:
            raise EditRefused(f"no tensor named {name!r}")
        self.policy.check_name(name)
        base = self.baseline[name]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if tuple(tensor.shape) != tuple(base.shape):
            raise EditRefused(
                f"{name}: shape {tuple(tensor.shape)} does not match {tuple(base.shape)}"
            )
        new = tensor.detach().to(torch.float32)
        if not torch.isfinite(new).all():
            raise EditRefused(f"{name}: edit contains NaN or Inf")
        if not evidence.strip():
            raise EditRefused(
                f"{name}: every edit must cite its evidence (which tool result justifies it)"
            )
        base32 = base.detach().to(torch.float32)
        if self.policy.forbid_scalar_multiple and _is_scalar_multiple(base32, new):
            raise EditRefused(
                f"{name}: an edit that only multiplies the whole tensor by a constant is refused"
            )
        n_changed = changed_entries(base32, new)
        if n_changed == 0:
            raise EditRefused(
                f"{name}: the edit vanishes in {DEPLOY_DTYPE} (too small to survive deployment)"
            )
        cap = self.policy.max_fraction_entries
        if cap is not None:
            # the cap governs direct edits; a tensor a repair method already
            # rewrote is charged only for what this edit changes on top of it
            others = sum(e.entries_changed for e in self._latest().values()
                         if e.name != name and e.method is None)
            if name in self._current and self._latest()[name].method is not None:
                n_changed_direct = changed_entries(self._current[name], new)
            else:
                n_changed_direct = n_changed
            n_cap = n_changed_direct
            if (others + n_cap) / self.total_entries > cap:
                raise EditRefused(
                    f"{name}: edit would change {(others + n_cap) / self.total_entries:.4%} of "
                    f"the model; the direct-edit cap is {cap:.4%}. Target neurons or head slices, "
                    "not whole matrices."
                )
        prior = self._latest().get(name)
        self._current[name] = new
        # a direct edit on top of a method's tensor keeps the method's mark:
        # the candidate as a whole is still not a direct-edit-only repair
        edit = Edit(self._next_id, name, n_changed, evidence.strip(),
                    method=prior.method if prior is not None else None)
        self._next_id += 1
        self._edits.append(edit)
        return edit

    def adopt(self, method: str, new_state: Mapping[str, torch.Tensor], evidence: str) -> dict:
        """Stage the result of a whole-model repair method (ReSta, CTheta, ...).

        Repair methods rewrite many tensors at once, so the direct-edit cap
        does not apply to them; the candidate is then graded in the
        uncapped arm (:attr:`direct_edit` becomes False). Protected tensors
        are left as they are and reported, and the evidence rule still holds.
        """
        if not evidence.strip():
            raise EditRefused(f"{method}: every repair must cite its evidence")
        changed, skipped, entries = [], [], 0
        for name, tensor in new_state.items():
            if name not in self.baseline:
                continue
            base32 = self.baseline[name].detach().to(torch.float32)
            new = tensor.detach().to(torch.float32).cpu()
            if tuple(new.shape) != tuple(base32.shape):
                raise EditRefused(f"{method}: {name} changed shape")
            current = self._current.get(name, base32)
            if changed_entries(current, new) == 0:
                continue
            if not self.policy.is_editable(name):
                skipped.append(name)
                continue
            if not torch.isfinite(new).all():
                raise EditRefused(f"{method}: {name} contains NaN or Inf; nothing was applied")
            changed.append((name, new, changed_entries(base32, new)))
        for name, new, n in changed:
            self._current[name] = new
            self._edits.append(Edit(self._next_id, name, n, evidence.strip(), method=method))
            self._next_id += 1
            entries += n
        return {"method": method, "tensors_changed": len(changed), "entries_changed": entries,
                "fraction_of_model": entries / self.total_entries, "skipped_protected": skipped}

    @property
    def direct_edit(self) -> bool:
        """True while every live edit is a direct edit (the capped arm)."""
        return all(e.method is None for e in self._latest().values())

    def revert(self, name: str | None = None) -> list[str]:
        """Undo the edit to one tensor, or all edits when ``name`` is None."""
        targets = [name] if name else list(self._current)
        reverted = [n for n in targets if self._current.pop(n, None) is not None]
        self._edits = [e for e in self._edits if e.name not in reverted]
        return reverted

    # ---- reporting -----------------------------------------------------
    def _latest(self) -> dict[str, Edit]:
        return {e.name: e for e in self._edits if e.name in self._current}

    def list_edits(self) -> list[Edit]:
        return list(self._latest().values())

    def entries_changed(self) -> int:
        return sum(e.entries_changed for e in self._latest().values())

    def fraction_changed(self) -> float:
        return self.entries_changed() / self.total_entries

    def changed_tensors(self) -> dict[str, torch.Tensor]:
        return dict(self._current)

    def state(self) -> dict[str, torch.Tensor]:
        """The full candidate state dict (baseline with edits applied)."""
        out = dict(self.baseline)
        for name, t in self._current.items():
            out[name] = t.to(self.baseline[name].dtype)
        return out

    def view(self):
        """``W`` as the agent sees it: six functions and nothing else.

        The session itself never enters the kernel, so agent code cannot reach
        the baseline, the policy or the edit table except through these."""
        from types import SimpleNamespace

        return SimpleNamespace(
            names=lambda pattern="*": self.names(pattern),
            tensor=lambda name: self.tensor(name),
            original=lambda name: self.original(name),
            save_edit=lambda name, tensor, evidence="": _edit_row(self.save_edit(name, tensor, evidence)),
            revert=lambda name=None: self.revert(name),
            list_edits=lambda: [_edit_row(e) for e in self.list_edits()],
        )

    def export(self, out_dir: str | Path) -> Path:
        """Write the patch: changed tensors plus a manifest the grader reads."""
        from safetensors.torch import save_file

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self._current:
            save_file({n: t.contiguous() for n, t in self._current.items()}, out / "patch.safetensors")
        manifest = {
            "baseline_hash": state_hash(self.baseline),
            "entries_changed": self.entries_changed(),
            "fraction_changed": self.fraction_changed(),
            "direct_edit": self.direct_edit,
            "edits": [e.__dict__ for e in self.list_edits()],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return out


def _edit_row(edit: Edit) -> dict:
    row = {"edit_id": edit.edit_id, "name": edit.name, "entries_changed": edit.entries_changed}
    if edit.method:
        row["method"] = edit.method
    return row


def _is_scalar_multiple(base: torch.Tensor, new: torch.Tensor) -> bool:
    """True when ``new == c * base`` for one constant c != 1."""
    mask = base != 0
    if mask.sum() < 2:
        return False
    ratio = new[mask] / base[mask]
    c = ratio.flatten()[0]
    if torch.isclose(c, torch.tensor(1.0)):
        return False
    same_ratio = torch.allclose(ratio, c.expand_as(ratio), rtol=1e-5, atol=1e-8)
    zeros_kept = torch.allclose(new[~mask], torch.zeros_like(new[~mask]))
    return bool(same_ratio and zeros_kept)
