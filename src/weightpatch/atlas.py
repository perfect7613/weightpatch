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
        (root / "diagram.svg").write_text(diagram_svg(card))
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

    def diagram(self) -> str:
        return (self.root / "diagram.svg").read_text()

    def set_profile(self, text: str) -> None:
        """Record measured behaviour (a baseline profile) for agents to read."""
        (self.root / "profile.md").write_text(text.strip() + "\n")
        self.write_summary()

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
        lines += ["", "## How one layer is wired", wiring(card)]
        lines += ["", "## Tensors in one layer (name suffix: shape)"]
        lines += [f"- {suffix}: {shape}" for suffix, shape in card["layer_template"].items()]
        if card["non_layer_tensors"]:
            lines += ["", "## Tensors outside the layers"]
            lines += [f"- {n}: {s}" for n, s in card["non_layer_tensors"].items()]
        lines += ["", f"Layer tensors are named `{card['layer_prefix']}<i>.<suffix>`."]
        if quirks:
            lines += ["", "## Quirks", quirks]
        profile = self.root / "profile.md"
        if profile.exists():
            lines += ["", "## Behaviour of this checkpoint, measured (do not re-measure what is here)",
                      profile.read_text().strip()]
        lines += ["", f"## Findings so far ({len(findings)})"]
        lines += [f"- [{f['tool']}] {json.dumps(f['result'], default=str)[:300]}" for f in findings[-20:]]
        if not findings:
            lines.append("- none yet")
        (self.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


PROFILE_LABELS = {"harm": "harmful answers (judged), lower is better",
                  "refusal": "refusals of safe requests, lower is better",
                  "capability": "accuracy, higher is better"}


def profile_markdown(data: Mapping[str, Any]) -> str:
    """A measured baseline profile as a short table. Rates only: the prompts
    behind the sealed suites stay on the grader's side."""
    lines = [f"Measured on `{data.get('model', '?')}` as released, n={data.get('n', '?')} per suite, "
             "95% Wilson intervals.", "", "| suite | what is counted | rate | interval |", "|---|---|---|---|"]
    for name, row in data.get("suites", {}).items():
        lo, hi = row.get("ci", [0, 1])
        lines.append(f"| {name} | {PROFILE_LABELS.get(row.get('kind'), row.get('kind'))} | "
                     f"{row['rate']:.0%} | {lo:.0%} to {hi:.0%} |")
    return "\n".join(lines)


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
        "layer_kinds": _layer_kinds(config, n_layers),
        "block": _block_topology(template),
        "params_by_module": _params_by_module(state),
    }


def _layer_kinds(config: Mapping[str, Any], n_layers: int) -> list[str]:
    """Per layer: "sliding" (windowed attention) or "global" (full attention)."""
    kinds = config.get("layer_types")
    if isinstance(kinds, (list, tuple)) and len(kinds) == n_layers:
        return ["global" if "full" in str(k) or "global" in str(k) else "sliding" for k in kinds]
    pattern = config.get("sliding_window_pattern")
    if config.get("sliding_window") and pattern:
        return ["global" if (i + 1) % int(pattern) == 0 else "sliding" for i in range(n_layers)]
    return ["global"] * n_layers


def _block_topology(template: Mapping[str, Any]) -> dict:
    """How one layer is wired, read from which tensors it has."""
    norms = sorted(k for k in template if "norm" in k)
    mlp = sorted(k.split(".")[1] for k in template if k.startswith("mlp."))
    return {"parallel": len(norms) == 1,  # one norm feeding both attention and MLP
            "norms": norms, "gated_mlp": "gate_proj" in mlp, "mlp_parts": mlp,
            "attn_parts": sorted(k.split(".")[1] for k in template if k.startswith("self_attn."))}


def _params_by_module(state: Mapping[str, torch.Tensor]) -> dict[str, int]:
    out: Counter = Counter()
    for name, t in state.items():
        m = LAYER_RE.search(name)
        key = ".".join(name[m.end():].split(".")[:2]).removesuffix(".weight") if m else name.removesuffix(".weight")
        out[key] += t.numel()
    return dict(out.most_common())


def wiring(card: Mapping[str, Any]) -> str:
    """The architecture as text an agent can act on: how data flows through a
    layer, and what a row or a column of each tensor means. This is the part
    of the atlas that replaces exploratory tool calls."""
    d, heads, kv, mlp = card.get("hidden_size"), card.get("n_heads") or 0, card.get("n_kv_heads") or 0, card.get("mlp_width")
    block, kinds = card.get("block", {}), card.get("layer_kinds", [])
    head_dim = (d // heads) if d and heads else None
    lines = []
    if block.get("parallel"):
        lines += ["```",
                  "x ──► LayerNorm ──┬─► Attention ──┐",
                  "│                 └─► MLP ────────┤   (attention and MLP read the SAME normed input)",
                  "└───────────────────────────────(+)──► x'   x' = x + attn(norm(x)) + mlp(norm(x))",
                  "```"]
    else:
        lines += ["```", "x ─► Norm ─► Attention ─(+)─► Norm ─► MLP ─(+)─► x'   (sequential block)", "```"]
    if block.get("gated_mlp"):
        lines.append("- MLP: `down_proj( act(gate_proj(h)) * up_proj(h) )`. **Neuron i** is row i of `gate_proj` and "
                     "`up_proj` (what it reads) and **column i of `down_proj`** (what it writes to the residual "
                     f"stream, a vector of size {d}). There are {mlp:,} neurons per layer.")
    if head_dim:
        group = f" Key and value heads are shared in groups: {kv} KV heads serve {heads} query heads." if kv and kv != heads else ""
        lines.append(f"- Attention: {heads} heads of size {head_dim}. **Head h** is rows `[h*{head_dim}, (h+1)*{head_dim})` "
                     f"of `q_proj` (what it looks for) and **columns `[h*{head_dim}, (h+1)*{head_dim})` of `o_proj`** "
                     f"(what it writes).{group}")
    lines.append("- `o_proj` and `down_proj` are the only tensors that write to the residual stream, and each "
                 "COLUMN is one unit's write vector. To change what a layer writes along a direction v, edit the "
                 "columns of the heads or neurons responsible, not whole matrices.")
    if kinds and len(set(kinds)) > 1:
        glob = [i for i, k in enumerate(kinds) if k == "global"]
        lines.append(f"- Layer kinds: {len(kinds) - len(glob)} sliding-window layers and {len(glob)} global layers. "
                     f"Global (full attention, no positional encoding): {glob}.")
    total = card.get("n_params") or 1
    share = ", ".join(f"{k} {v / total:.1%}" for k, v in list(card.get("params_by_module", {}).items())[:6])
    lines.append(f"- Where the parameters are: {share}.")
    return "\n".join(lines)


def diagram_svg(card: Mapping[str, Any]) -> str:
    """A block diagram of the model, built from the card: the layer stack on
    the left (coloured by kind), one layer's wiring on the right."""
    kinds = card.get("layer_kinds") or ["global"] * int(card.get("n_layers") or 0)
    n, d, heads, mlp = len(kinds), card.get("hidden_size"), card.get("n_heads"), card.get("mlp_width")
    block, total = card.get("block", {}), card.get("n_params") or 1
    cols = 6 if n > 12 else 1
    rows = -(-n // cols)
    cw, ch, gx, gy = 46, 22, 6, 6
    stack_w = cols * (cw + gx)
    H = max(420, 150 + rows * (ch + gy) + 90)
    W = 60 + stack_w + 60 + 520
    c = {"sliding": "#4C78A8", "global": "#E45756", "box": "#F4F1EA", "ink": "#22223B", "mut": "#6B7280",
         "attn": "#72B7B2", "mlp": "#F2A541", "norm": "#B8B8D1"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="ui-sans-serif,system-ui,Helvetica,Arial" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{c["ink"]}"/></marker></defs>',
         f'<text x="30" y="30" font-size="17" font-weight="700" fill="{c["ink"]}">{card.get("model", "model")}</text>',
         f'<text x="30" y="50" fill="{c["mut"]}">{n} layers · width {d} · {heads} heads · MLP {mlp:,} · '
         f'{total / 1e9:.2f}B parameters · checkpoint {card.get("checkpoint_hash", "")}</text>']
    x0, y0 = 30, 110
    emb = card.get("vocab_size")
    o.append(f'<rect x="{x0}" y="{y0 - 40}" width="{stack_w - gx}" height="26" rx="5" fill="{c["norm"]}"/>'
             f'<text x="{x0 + (stack_w - gx) / 2}" y="{y0 - 23}" text-anchor="middle" fill="{c["ink"]}">'
             f'token embedding ({emb:,} × {d}, tied to the output layer)</text>' if emb else "")
    for i, kind in enumerate(kinds):
        x, y = x0 + (i % cols) * (cw + gx), y0 + (i // cols) * (ch + gy)
        o.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="{c[kind]}"/>'
                 f'<text x="{x + cw / 2}" y="{y + 15}" text-anchor="middle" fill="white" font-size="11">L{i}</text>')
    yb = y0 + rows * (ch + gy) + 6
    o.append(f'<rect x="{x0}" y="{yb}" width="{stack_w - gx}" height="26" rx="5" fill="{c["norm"]}"/>'
             f'<text x="{x0 + (stack_w - gx) / 2}" y="{yb + 17}" text-anchor="middle" fill="{c["ink"]}">final norm → logits</text>')
    yl = yb + 50
    if len(set(kinds)) > 1:
        o.append(f'<rect x="{x0}" y="{yl}" width="14" height="14" rx="3" fill="{c["sliding"]}"/>'
                 f'<text x="{x0 + 20}" y="{yl + 11}" fill="{c["ink"]}">sliding-window attention ({kinds.count("sliding")})</text>'
                 f'<rect x="{x0}" y="{yl + 20}" width="14" height="14" rx="3" fill="{c["global"]}"/>'
                 f'<text x="{x0 + 20}" y="{yl + 31}" fill="{c["ink"]}">global attention, no positions ({kinds.count("global")})</text>')
    # one layer
    bx, by = x0 + stack_w + 60, 80
    o.append(f'<rect x="{bx}" y="{by}" width="500" height="320" rx="12" fill="{c["box"]}" stroke="{c["mut"]}" stroke-dasharray="4 3"/>'
             f'<text x="{bx + 14}" y="{by + 22}" font-weight="700" fill="{c["ink"]}">one layer</text>')
    def box(x, y, w, h, fill, title, sub=""):
        t = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}"/>'
             f'<text x="{x + w / 2}" y="{y + (h / 2 if sub else h / 2 + 4)}" text-anchor="middle" font-weight="600" fill="{c["ink"]}">{title}</text>')
        if sub:
            t += f'<text x="{x + w / 2}" y="{y + h / 2 + 16}" text-anchor="middle" font-size="10.5" fill="{c["ink"]}">{sub}</text>'
        return t
    def arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["ink"]}" stroke-width="1.4" marker-end="url(#a)"/>'
    mid = by + 170
    t = card.get("layer_template", {})
    shape = lambda k: "×".join(str(v) for v in t.get(k, []))  # noqa: E731
    o.append(f'<text x="{bx + 14}" y="{mid + 4}" fill="{c["ink"]}">x</text>')
    o.append(arrow(bx + 26, mid, bx + 60, mid))
    o.append(box(bx + 62, mid - 22, 74, 44, c["norm"], "LayerNorm"))
    if block.get("parallel", True):
        o.append(arrow(bx + 136, mid - 8, bx + 178, by + 95))
        o.append(arrow(bx + 136, mid + 8, bx + 178, by + 245))
    else:
        o.append(arrow(bx + 136, mid, bx + 178, by + 95))
    o.append(box(bx + 180, by + 50, 190, 90, c["attn"], f"Attention · {heads} heads",
                 f"o_proj {shape('self_attn.o_proj.weight')}"))
    o.append(box(bx + 180, by + 200, 190, 90, c["mlp"], f"Gated MLP · {mlp:,} neurons",
                 f"down_proj {shape('mlp.down_proj.weight')}"))
    o.append(f'<circle cx="{bx + 430}" cy="{mid}" r="13" fill="white" stroke="{c["ink"]}"/>'
             f'<text x="{bx + 430}" y="{mid + 5}" text-anchor="middle" font-size="15" fill="{c["ink"]}">+</text>')
    o.append(arrow(bx + 370, by + 95, bx + 420, mid - 9))
    o.append(arrow(bx + 370, by + 245, bx + 420, mid + 9))
    o.append(arrow(bx + 443, mid, bx + 488, mid))
    o.append(f'<path d="M{bx + 40},{mid} L{bx + 40},{by + 308} L{bx + 430},{by + 308} L{bx + 430},{mid + 14}" fill="none" '
             f'stroke="{c["mut"]}" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#a)"/>'
             f'<text x="{bx + 235}" y="{by + 303}" text-anchor="middle" font-size="10.5" fill="{c["mut"]}">residual stream (width {d})</text>')
    note = "attention and MLP run in parallel off one norm" if block.get("parallel", True) else "sequential block"
    o.append(f'<text x="{bx + 250}" y="{by + 172}" text-anchor="middle" font-size="10.5" fill="{c["mut"]}">{note}</text>')
    share = " · ".join(f"{k} {v / total:.0%}" for k, v in list(card.get("params_by_module", {}).items())[:5])
    o.append(f'<text x="{bx}" y="{by + 345}" font-size="10.5" fill="{c["mut"]}">parameters: {share}</text>')
    o.append("</svg>")
    return "\n".join(x for x in o if x)


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
