"""Tool box, native tools, runner and atlas."""
import pytest
import torch

from weightpatch.atlas import Atlas
from weightpatch.budget import Budget, BudgetExceeded
from weightpatch.edits import EditPolicy, EditSession
from weightpatch.tools import ToolBox
from weightpatch.tools.native import DevSet, ToolContext, looks_like_refusal, register_native_tools
from weightpatch.toy import make_planted_drift, score_repair, tiny_cohere2_state


def box_for(drift, allow=("*",), budget=None, runner=None, dev=None):
    session = EditSession(drift.drifted, EditPolicy(max_fraction_entries=0.01))
    box = ToolBox(budget or Budget(), allow=allow)
    ctx = ToolContext(session, reference=drift.aligned, runner=runner, dev=dev or DevSet())
    register_native_tools(box, ctx)
    return box, session, ctx


def test_allow_list_removes_groups_and_describe_matches():
    drift = make_planted_drift()
    box, _, _ = box_for(drift, allow=("inspect.*", "edit.revert"))
    tools = box.namespace()
    assert hasattr(tools, "inspect") and not hasattr(tools, "measure") and not hasattr(tools, "localize")
    assert hasattr(tools.edit, "revert") and not hasattr(tools.edit, "project_out")
    text = box.describe()
    assert "tools.inspect.diff_weights" in text and "dev_eval" not in text


def test_drift_map_and_restore_repair_the_planted_drift_exactly():
    drift = make_planted_drift()
    box, session, _ = box_for(drift)
    tools = box.namespace()
    rows = tools.inspect.diff_weights()
    assert [r["name"] for r in rows] == [drift.tensor]  # only one tensor drifted
    svd = tools.inspect.svd_of_diff(drift.tensor, k=1)
    assert svd["S"][0] > 100 * svd["S"][1]  # the planted change is rank 1
    writers = tools.localize.rank_writers(svd["U"][:, 0], pattern="*down_proj*", top=6, against_reference=True)
    cols = sorted(r["col"] for r in writers)
    assert cols == drift.cols and all(r["name"] == drift.tensor for r in writers)
    tools.edit.restore_from_reference(drift.tensor, cols=cols, evidence="rank_writers top 6")
    score = score_repair(session, drift)
    assert score["restored_fraction"] == 1.0 and score["collateral_entries"] == 0
    assert [c.name for c in box.calls][-1] == "edit.restore_from_reference" and all(c.ok for c in box.calls)


def test_whole_tensor_restore_hits_the_cap_and_is_logged_as_failed():
    drift = make_planted_drift()
    box, session, _ = box_for(drift)
    session.policy = EditPolicy(max_fraction_entries=1e-4)
    with pytest.raises(Exception, match="cap"):
        box.namespace().edit.restore_from_reference(drift.tensor, evidence="everything")
    assert box.calls[-1].ok is False and session.entries_changed() == 0


class FakeRunner:
    n_layers = 4

    def __init__(self):
        self.calls = 0

    def hidden(self, prompts, layers):
        self.calls += 1
        base = torch.ones(len(prompts), len(layers), 64)
        return base * (2.0 if prompts and prompts[0].startswith("bad") else 1.0)

    def generate(self, prompts, max_new_tokens=64):
        self.calls += 1
        return ["I'm sorry, I can't help with that." if p.startswith("bad") else "Sure: here you go." for p in prompts]


def test_forward_tools_charge_passes_and_prompts():
    drift = make_planted_drift()
    dev = DevSet(harmful=[f"bad {i}" for i in range(8)], benign=[f"good {i}" for i in range(8)])
    budget = Budget({"forward_passes": 2, "dev_prompts": 100})
    box, _, ctx = box_for(drift, budget=budget, runner=FakeRunner(), dev=dev)
    tools = box.namespace()
    out = tools.measure.dev_eval(n=8)
    assert out["harmful_refusal_rate"] == 1.0 and out["benign_refusal_rate"] == 0.0
    assert budget.spent == {"forward_passes": 1, "dev_prompts": 16, "gpu_seconds": budget.spent["gpu_seconds"]}
    directions = tools.localize.refusal_direction(layers=[1, 2], n=8)
    assert set(directions) == {1, 2} and abs(float(directions[1].norm()) - 1) < 1e-5
    with pytest.raises(BudgetExceeded):
        tools.localize.activations(["x"], [0])
    assert ctx.runner.calls == 4  # two per tool; the refused call never reached the model


def test_weights_only_rung_has_no_runner():
    box, _, _ = box_for(make_planted_drift())
    with pytest.raises(RuntimeError, match="no forward runner"):
        box.namespace().measure.dev_eval(n=1)


def test_refusal_string_match():
    assert looks_like_refusal("I’m sorry, but I can't help with that.")
    assert not looks_like_refusal("Sure! Step one is to preheat the oven.")


def test_hf_runner_applies_and_reverts_edits():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from weightpatch.runner import HFRunner

    model, state, _ = tiny_cohere2_state()
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, **{f"w{i}": i + 4 for i in range(100)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
    session = EditSession(state, EditPolicy(max_fraction_entries=None))
    runner = HFRunner(model, tokenizer, session, chat=False)
    prompts = ["w1 w2 w3", "w4 w5"]
    before = runner.hidden(prompts, [0, 3])
    assert before.shape == (2, 2, 64)
    name = "model.layers.1.mlp.down_proj.weight"
    w = session.tensor(name); w[:, :32] *= -3.0
    session.save_edit(name, w, evidence="test")
    after = runner.hidden(prompts, [0, 3])
    assert torch.allclose(before[:, 0], after[:, 0]) and not torch.allclose(before[:, 1], after[:, 1])
    session.revert(name)
    assert torch.allclose(runner.hidden(prompts, [0, 3]), before)
    assert len(runner.generate(prompts, max_new_tokens=3)) == 2


def test_atlas_is_built_from_code_and_keyed_by_checkpoint(tmp_path):
    drift = make_planted_drift()
    atlas = Atlas.build(tmp_path, "toy/tiny-cohere2", drift.drifted, drift.config, quirks="- a quirk")
    card = atlas.card()
    assert card["n_layers"] == 4 and card["mlp_width"] == 128 and card["hidden_size"] == 64
    assert card["layer_prefix"] == "model.layers." and "mlp.down_proj.weight" in card["layer_template"]
    nm = atlas.name_map()["layers"]["2"]
    assert nm["mlp"]["circuitkit"] == "MLP 2" and nm["attn"]["circuitkit"][:2] == ["A2.0", "A2.1"]
    assert "model.layers.2.mlp.down_proj.weight" in nm["mlp"]["hf"]
    atlas.add_finding("agent", "inspect.diff_weights", {"top": drift.tensor})
    summary = atlas.summary()
    assert "a quirk" in summary and drift.tensor in summary and "4 layers" in summary
    other = Atlas.build(tmp_path, "toy/tiny-cohere2", drift.aligned, drift.config)
    assert other.checkpoint_hash != atlas.checkpoint_hash and other.findings() == []
