"""A planted-drift toy with an exact answer key.

A tiny random Cohere2 model (the same model class and tensor names as Tiny
Aya) is the "aligned" checkpoint. The "drifted" checkpoint is the same model
with a rank-1 change planted in a few columns of one MLP down-projection.
Because the key is known, a repair can be scored exactly: how much of the
planted change was undone, and how much was touched that should not have been.

It runs on a laptop CPU in seconds, so the whole harness (model client,
kernel, tool box, edit policy, trace) can be exercised end to end, including
live against a real agent, before any GPU or gated model is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .edits import EditSession, changed_entries


@dataclass
class PlantedDrift:
    aligned: dict[str, torch.Tensor]
    drifted: dict[str, torch.Tensor]
    tensor: str  # the tensor that was changed
    cols: list[int]  # the columns that were changed
    config: dict


def tiny_cohere2_state(seed: int = 0, layers: int = 4, hidden: int = 64, mlp: int = 128):
    """State and config of a tiny random Cohere2 causal LM."""
    from transformers import Cohere2Config, Cohere2ForCausalLM

    torch.manual_seed(seed)
    config = Cohere2Config(
        vocab_size=512, hidden_size=hidden, intermediate_size=mlp, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, sliding_window=32, sliding_window_pattern=4,
    )
    model = Cohere2ForCausalLM(config)
    state = {n: p.detach().clone().float() for n, p in model.named_parameters()}
    return model, state, config.to_dict()


def make_planted_drift(seed: int = 0, layer: int = 2, n_cols: int = 6, scale: float = 0.5) -> PlantedDrift:
    _, aligned, config = tiny_cohere2_state(seed)
    g = torch.Generator().manual_seed(seed + 1)
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    w = aligned[name]
    cols = sorted(torch.randperm(w.shape[1], generator=g)[:n_cols].tolist())
    direction = torch.randn(w.shape[0], generator=g)
    direction = direction / direction.norm()
    drifted = {n: t.clone() for n, t in aligned.items()}
    drifted[name][:, cols] += scale * direction[:, None] * torch.randn(1, n_cols, generator=g).abs()
    return PlantedDrift(aligned, drifted, name, cols, config)


def score_repair(session: EditSession, drift: PlantedDrift) -> dict:
    """Exact score against the key."""
    state = session.state()
    planted = changed_entries(drift.aligned[drift.tensor], drift.drifted[drift.tensor])
    remaining = changed_entries(drift.aligned[drift.tensor], state[drift.tensor])
    collateral = sum(
        changed_entries(drift.aligned[n], state[n]) for n in state if n != drift.tensor
    )
    return {
        "planted_entries": planted,
        "restored_fraction": 1 - remaining / planted,
        "collateral_entries": collateral,
        "entries_edited": session.entries_changed(),
        "fraction_of_model_edited": session.fraction_changed(),
    }


TOY_TASK = """\
This small transformer was fine-tuned and drifted away from its aligned reference: a few
weights were altered and its behaviour changed. You have the drifted weights (`W`) and tools
that compare them with the aligned reference.

Find exactly which weights drifted and repair them with the smallest edit you can. Do not
restore whole tensors: the direct-edit cap forbids it, and a precise repair is the goal.
Finish by stating which tensor and which columns you repaired, and the evidence.
"""
