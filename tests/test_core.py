"""Budget, edit session and kernel: the rules an agent cannot get around."""
import pytest
import torch

from weightpatch.budget import Budget, BudgetExceeded
from weightpatch.edits import EditPolicy, EditRefused, EditSession, changed_entries
from weightpatch.kernel import WeightKernel


def state():
    torch.manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(50, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
        "model.layers.1.mlp.down_proj.weight": torch.randn(8, 16),
        "model.norm.weight": torch.ones(8),
    }


NAME = "model.layers.0.mlp.down_proj.weight"


def test_budget_refuses_before_spending():
    b = Budget({"forward_passes": 2})
    b.charge("forward_passes", 2)
    with pytest.raises(BudgetExceeded):
        b.charge("forward_passes")
    assert b.spent["forward_passes"] == 2
    with pytest.raises(ValueError):
        Budget({"tokens": 1})


def test_edit_needs_evidence_shape_and_unprotected_name():
    s = EditSession(state(), EditPolicy(max_fraction_entries=None))
    w = s.tensor(NAME)
    w[:, 0] += 1.0
    with pytest.raises(EditRefused, match="evidence"):
        s.save_edit(NAME, w)
    with pytest.raises(EditRefused, match="shape"):
        s.save_edit(NAME, w[:, :4], evidence="x")
    with pytest.raises(EditRefused, match="protected"):
        s.save_edit("model.embed_tokens.weight", s.tensor("model.embed_tokens.weight") + 1, evidence="x")
    with pytest.raises(EditRefused, match="protected"):
        s.save_edit("model.norm.weight", s.tensor("model.norm.weight") * 2, evidence="x")
    edit = s.save_edit(NAME, w, evidence="diff_weights row 1")
    assert edit.entries_changed == 8 and s.entries_changed() == 8


def test_scalar_rescale_and_vanishing_edits_are_refused():
    s = EditSession(state(), EditPolicy(max_fraction_entries=None))
    with pytest.raises(EditRefused, match="constant"):
        s.save_edit(NAME, s.tensor(NAME) * 1.5, evidence="x")
    with pytest.raises(EditRefused, match="vanishes"):
        s.save_edit(NAME, s.tensor(NAME) + 1e-9, evidence="x")  # rounds away in bf16


def test_direct_edit_cap_counts_all_edits_together():
    s = EditSession(state(), EditPolicy(max_fraction_entries=16 / 664))  # 664 entries in total
    a = s.tensor(NAME); a[:, 0] += 1
    s.save_edit(NAME, a, evidence="x")
    other = "model.layers.1.mlp.down_proj.weight"
    b = s.tensor(other); b[:, :2] += 1  # 16 more entries: 24 in total, over the cap
    with pytest.raises(EditRefused, match="cap"):
        s.save_edit(other, b, evidence="x")
    b = s.tensor(other); b[:, 0] += 1  # 8 more: exactly at the cap
    s.save_edit(other, b, evidence="x")
    assert s.entries_changed() == 16


def test_revert_and_export(tmp_path):
    s = EditSession(state(), EditPolicy(max_fraction_entries=None))
    w = s.tensor(NAME); w[:, 3] = 0
    s.save_edit(NAME, w, evidence="x")
    assert changed_entries(s.baseline[NAME], s.state()[NAME]) == 8
    out = s.export(tmp_path / "patch")
    assert (out / "patch.safetensors").exists() and (out / "manifest.json").exists()
    assert s.revert(NAME) == [NAME] and s.entries_changed() == 0


def kernel(budget=None):
    s = EditSession(state(), EditPolicy(max_fraction_entries=None))
    return WeightKernel({"W": s.view()}, budget), s


def test_kernel_keeps_state_between_runs_and_charges():
    k, _ = kernel(Budget({"code_runs": 2}))
    assert k.run("x = 21").ok
    assert k.run("print(x * 2)").output == "42"
    third = k.run("print(1)")
    assert third.refused and "budget" in third.output


@pytest.mark.parametrize("code", [
    "import os", "from subprocess import run", "open('x')", "__import__('os')",
    "t = W.tensor('model.norm.weight'); t.requires_grad_(True)", "torch.optim.SGD",
    "from torch.optim import SGD", "from torch import autograd",
    "loss.backward()", "W._current", "().__class__", "torch.load('x')", "eval('1')",
])
def test_kernel_refuses_training_files_and_internals(code):
    k, _ = kernel()
    result = k.run(code)
    assert result.refused and not result.ok


def test_kernel_edits_only_through_the_view():
    k, s = kernel()
    r = k.run(f"w = W.tensor('{NAME}'); w[:, 1] += 2.0\nprint(W.save_edit('{NAME}', w, evidence='test'))")
    assert r.ok and s.entries_changed() == 8
    other = "model.layers.1.mlp.down_proj.weight"
    r = k.run(f"W.save_edit('{other}', W.tensor('{other}') * 3, evidence='x')")
    assert not r.ok and "EditRefused" in r.output  # the agent sees why
    r = k.run("note('layer 0 col 1 edited')")
    assert k.notebook == ["layer 0 col 1 edited"]


def test_kernel_reports_errors_without_dying():
    k, _ = kernel()
    assert "ZeroDivisionError" in k.run("1/0").output
    assert k.run("print('still alive')").ok
