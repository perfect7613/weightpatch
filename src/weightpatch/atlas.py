"""Model atlas: what an agent should know about a checkpoint before any tool call.

Built once per checkpoint by code (never by an LLM), then read as context by
every later agent so none of them spends tool calls rediscovering the
architecture. Layout, keyed by the checkpoint's content hash:

    atlas/<model>/<checkpoint-hash>/
      card.json        static specs, from config and tensor shapes
      name_map.json    HF module <-> TransformerLens hook <-> CircuitKIT node
      findings.jsonl   append-only: who, tool, result
      quirks.md        known gotchas for this model family
      SUMMARY.md       the one page agents read first

Two rules: the grader never reads the atlas, and the atlas never holds
held-out data. A patched checkpoint has a new hash, so it never inherits a
stale circuit claim.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .edits import state_hash

LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")

TINY_AYA_QUIRKS = """\
- The chat template adds a safety preamble of about 366 tokens. Keep the template byte-identical before and after an edit.
- Attention and MLP run in parallel off one LayerNorm: both read the same input and both write to the residual stream.
- The vocabulary has 262,144 tokens and embeddings are tied. Never materialise logits at every position.
- Small updates round away in bfloat16. Edits are made in float32 and counted after casting to bfloat16.
- ReSta: use alpha near 0.25; alpha = 1 breaks the model.
- Refusal steering works best near layer 24 of 36.
"""


@dataclass
class Atlas:
    root: Path
    checkpoint_hash: str

    # ---- build ---------------------------------------------------------
    @classmethod
    def build(
        cls,
        base_dir: str | Path,
        model_name: str,
        state: Mapping[str, torch.Tensor],
        config: Mapping[str, Any] | None = None,
        quirks: str = "",
    ) -> "Atlas":
        ck = state_hash(state)
        root = Path(base_dir) / model_name.replace("/", "__") / ck
        root.mkdir(parents=True, exist_ok=True)
        atlas = cls(root, ck)
        card = build_card(model_name, state, config or {}, ck)
        (root / "card.json").write_text(json.dumps(card, indent=2))
        (root / "name_map.json").write_text(json.dumps(build_name_map(state, card), indent=2))
        if not (root / "quirks.md").exists():
            (root / "quirks.md").write_text(quirks or "")
        (root / "findings.jsonl").touch()
        atlas.write_summary()
        return atlas

    # ---- read ----------------------------------------------------------
    def card(self) -> dict:
        return json.loads((self.root / "card.json").read_text())

    def name_map(self) -> dict:
        return json.loads((self.root / "name_map.json").read_text())

    def findings(self) -> list[dict]:
        lines = (self.root / "findings.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def summary(self) -> str:
        return (self.root / "SUMMARY.md").read_text()

    # ---- write ---------------------------------------------------------
    def add_finding(self, who: str, tool: str, result: Any) -> None:
        """Append one finding. Findings are keyed by this checkpoint's hash."""
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": self.checkpoint_hash,
               "who": who, "tool": tool, "result": result}
        with (self.root / "findings.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.write_summary()

    def write_summary(self) -> None:
        card = self.card()
        quirks = (self.root / "quirks.md").read_text().strip()
        findings = self.findings()
        lines = [
            f"# Atlas: {card['model']} @ {card['checkpoint_hash']}",
            "",
            "## Architecture",
            f"- {card['n_layers']} layers, width {card.get('hidden_size', '?')}, "
            f"{card['n_params']:,} parameters in {card['n_tensors']} tensors",
        ]
        for key, label in (("n_heads", "attention heads per layer"), ("mlp_width", "MLP width"),
                           ("vocab_size", "vocabulary")):
            if card.get(key):
                lines.append(f"- {label}: {card[key]:,}")
        lines += ["", "## Tensors in one layer (name suffix: shape)"]
        lines += [f"- {suffix}: {shape}" for suffix, shape in card["layer_template"].items()]
        if card["non_layer_tensors"]:
            lines += ["", "## Tensors outside the layers"]
            lines += [f"- {n}: {s}" for n, s in card["non_layer_tensors"].items()]
        lines += ["", f"Layer tensors are named `{card['layer_prefix']}<i>.<suffix>`."]
        if quirks:
            lines += ["", "## Quirks", quirks]
        lines += ["", f"## Findings so far ({len(findings)})"]
        lines += [f"- [{f['tool']}] {json.dumps(f['result'], default=str)[:300]}" for f in findings[-20:]]
        if not findings:
            lines.append("- none yet")
        (self.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def build_card(model_name: str, state: Mapping[str, torch.Tensor], config: Mapping[str, Any], ck: str) -> dict:
    """Static specs read from tensor names and shapes, with config filling gaps."""
    per_layer: dict[int, dict[str, list[int]]] = {}
    non_layer: dict[str, list[int]] = {}
    prefix = ""
    for name, t in state.items():
        m = LAYER_RE.search(name)
        if not m:
            non_layer[name] = list(t.shape)
            continue
        per_layer.setdefault(int(m.group(1)), {})[name[m.end():]] = list(t.shape)
        prefix = name[: m.start(1)]
    n_layers = len(per_layer)
    template = per_layer.get(0, {})
    widths = Counter(s[0] for s in template.values() if len(s) == 2)
    return {
        "model": model_name,
        "checkpoint_hash": ck,
        "n_layers": n_layers or config.get("num_hidden_layers", 0),
        "hidden_size": config.get("hidden_size"),
        "n_heads": config.get("num_attention_heads"),
        "n_kv_heads": config.get("num_key_value_heads"),
        "mlp_width": config.get("intermediate_size") or (max(widths) if widths else None),
        "vocab_size": config.get("vocab_size"),
        "n_params": sum(t.numel() for t in state.values()),
        "n_tensors": len(state),
        "layer_prefix": prefix,
        "layer_template": template,
        "non_layer_tensors": non_layer,
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
    }


def build_name_map(state: Mapping[str, torch.Tensor], card: dict) -> dict:
    """Per layer: the HF tensor names, the TransformerLens hooks and the
    CircuitKIT node ids (`A<layer>.<head>`, `MLP <layer>`) for the same thing."""
    heads = card.get("n_heads") or 0
    layers = {}
    for i in range(card["n_layers"]):
        prefix = f"{card['layer_prefix']}{i}."
        tensors = [n for n in state if n.startswith(prefix)]
        layers[str(i)] = {
            "attn": {
                "hf": [n for n in tensors if "attn" in n],
                "tl_hook": f"blocks.{i}.hook_attn_out",
                "circuitkit": [f"A{i}.{h}" for h in range(heads)],
            },
            "mlp": {
                "hf": [n for n in tensors if "mlp" in n or "ffn" in n],
                "tl_hook": f"blocks.{i}.hook_mlp_out",
                "circuitkit": f"MLP {i}",
            },
        }
    return {"layers": layers}
