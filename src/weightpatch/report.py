"""The report agent: turn a finished run into a write-up, from evidence only.

Two tool-free model calls on the same login as the episode:

1. a writer gets an evidence pack built by code (task, profile, every turn's
   explanation and printed output, the notebook, the edits by layer, the
   grader's verdicts, the token and GPU use) and writes the report;
2. an editor rewrites that draft with the humanizer skill
   (github.com/blader/humanizer, MIT), changing wording and never a number.

Nothing here runs the model or touches weights. The evidence pack is also
saved, so every sentence of the report can be checked against it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

WRITER = """\
You are the report writer for WeightPatch, a harness in which an agent edits a language model's
weights directly. Below is the complete evidence from one finished run. Write the run report in
Markdown.

Rules:
- Use only what is in the evidence. Every number you write must appear in the evidence. Do not
  estimate, round a number into a different claim, or add background knowledge about the model.
- Report what failed and what was dropped as plainly as what worked. If the grader rejected a
  submission, say so and give its numbers.
- The grader's numbers are the result. Dev numbers measured by the agent are supporting evidence;
  label them as dev.
- Name weights exactly (tensor names, layers, how many columns).
- No code blocks longer than five lines. No preamble: start with the title.

Sections, in this order:
# <title: model, goal, outcome>
## Summary  (five sentences at most: goal, outcome with the main held-out number, size of the edit)
## The model and the problem  (the baseline profile; what had to change and what had to stay)
## How the agent worked  (the steps in order: what it measured, which tools named which units,
   what it tested before editing, each decision and the evidence behind it)
## The edit  (a table: layer, tensor, columns changed, entries changed, relative size; then the
   rule used to change them, in words)
## Results  (a table of every grader number: reference, candidate, difference, interval, n, significant
   or not; then the dev before/after table)
## What was tried and dropped
## Cost  (turns, code blocks, GPU seconds, forward passes, submissions, tokens)
## Limits  (what this run does not show)
## Files  (what is stored with this run and what each file is)

# Evidence
"""

EDITOR = """\
Edit the report below so it reads as plain, direct prose, following the skill that comes first.
Keep every fact, number, table, tensor name and section heading exactly as it is. Do not add
claims, do not remove results or caveats, and do not comment on your edits. Reply with the
edited report only, in Markdown, starting with its title.

# Skill
{skill}

# Report to edit
{draft}
"""


# ------------------------------------------------------------ layer record
def layer_record(baseline: Mapping[str, torch.Tensor], patch: Mapping[str, torch.Tensor]) -> list[dict]:
    """What changed, per tensor: which columns (and rows) and by how much.
    ``patch`` holds the full new value of every changed tensor."""
    rows = []
    for name in sorted(patch, key=_layer_key):
        old, new = baseline[name].float(), patch[name].float()
        delta = new - old
        m = re.search(r"layers\.(\d+)\.(.+)", name)
        row: dict[str, Any] = {"tensor": name, "layer": int(m.group(1)) if m else None,
                               "module": m.group(2) if m else name, "shape": list(old.shape),
                               "entries_changed": int((delta != 0).sum()),
                               "relative_change": float(delta.norm() / (old.norm() + 1e-12))}
        if delta.ndim == 2:
            col_norm, base_norm = delta.norm(dim=0), old.norm(dim=0) + 1e-12
            cols = torch.nonzero(col_norm > 0).flatten()
            row["columns_changed"] = len(cols)
            row["rows_touched"] = int((delta.abs().sum(dim=1) > 0).sum())
            row["columns"] = [{"col": int(c), "relative_change": round(float(col_norm[c] / base_norm[c]), 4),
                               "cosine_new_old": round(float(torch.nn.functional.cosine_similarity(
                                   new[:, c], old[:, c], dim=0)), 4)} for c in cols]
            if len(cols):
                # is the change one direction written into many columns?
                s = torch.linalg.svdvals(delta[:, cols])
                row["rank1_share"] = round(float(s[0] ** 2 / (s ** 2).sum()), 4)
        rows.append(row)
    return rows


def _layer_key(name: str):
    m = re.search(r"layers\.(\d+)\.", name)
    return (int(m.group(1)) if m else -1, name)


# ----------------------------------------------------------- evidence pack
def _read(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text if limit is None or len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} more characters in the stored file]"


def _explanation(answer: str) -> str:
    return re.sub(r"```.*?```", "[code: see code.py]", answer, flags=re.DOTALL).strip()


def evidence_pack(run_dir: str | Path, per_turn: int = 2200) -> str:
    """Everything the writer may use, as one text. Built from the files the
    episode wrote; nothing is summarised by a model."""
    run = Path(run_dir)
    trace, worker = run / "trace", run / "worker"
    report = json.loads(_read(run / "report.json") or "{}")
    parts: list[str] = []
    first = _read(trace / "turns" / "t001" / "prompt.txt")
    parts += ["## First message to the agent (task, how to act, this run's facts; the tool reference is cut)",
              first.split("# Tools")[0][:9000]]
    atlas = next(iter(sorted((worker / "atlas").glob("*/SUMMARY.md"))), None)
    if atlas:
        text = atlas.read_text()
        prof = text.split("## Behaviour of this checkpoint, measured")
        parts += ["## Model atlas: architecture", text.split("## Tensors in one layer")[0][:5000]]
        if len(prof) > 1:
            parts += ["## Baseline profile measured before the run", prof[1].split("## Findings")[0]]
    parts.append("## Turns (the agent's explanation, then what its code printed)")
    for d in sorted((trace / "turns").glob("t*")):
        usage = json.loads(_read(d / "run.json") or "{}")
        parts.append(f"### Turn {int(d.name[1:])} ({usage.get('seconds', '?')}s)\n"
                     f"Explanation: {_explanation(_read(d / 'answer.md'))[:per_turn]}\n"
                     f"Printed output: {_read(d / 'output.txt', per_turn)}")
    parts += ["## The agent's notebook", _read(trace / "notebook.md")]
    parts += ["## Grader verdicts, in order (the result)",
              _read(worker / "verdicts.jsonl") or json.dumps(report.get("verdicts", []), indent=1)]
    layers = json.loads(_read(run / "edits_by_layer.json") or "[]")
    if layers:
        slim = [{k: (v if k != "columns" else f"{len(v)} columns; largest relative changes: " + ", ".join(
            f"col {c['col']} ({c['relative_change']})" for c in sorted(v, key=lambda c: -c["relative_change"])[:6]))
            for k, v in row.items()} for row in layers]
        parts += ["## The final edit, per tensor (computed from the stored patch)", json.dumps(slim, indent=1)]
    manifest = report.get("manifest") or json.loads(_read(worker / "patch" / "manifest.json") or "{}")
    if manifest:
        parts += ["## Patch manifest (the evidence string is what the agent recorded with each edit)",
                  json.dumps(manifest, indent=1)[:6000]]
    usage = [u for u in report.get("usage", []) if u]
    totals = {k: sum(int(u.get(k, 0) or 0) for u in usage) for k in
              ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")}
    parts += ["## Cost", json.dumps({"turns": report.get("turns"), "stopped": report.get("stopped"),
                                     "all_turns_tool_free": report.get("tool_free"),
                                     "budget": report.get("budget"), "tokens": totals}, indent=1)]
    parts += ["## Files stored with this run",
              "\n".join(f"- {p.relative_to(run)}" for p in sorted(run.rglob("*"))
                        if p.is_file() and "circuitkit" not in p.parts and "turns" not in p.parts)
              + "\n- trace/turns/tNNN/{prompt.txt, answer.md, code.py, output.txt, events.jsonl, run.json}: one folder per turn"]
    return "\n\n".join(parts)


# ------------------------------------------------------------------ agents
def write_report(run_dir: str | Path, client, humanizer: str | None = None) -> dict:
    """Write ``REPORT.md`` for a finished run. ``client`` is any ModelClient
    (the tool-free Codex client in practice)."""
    run = Path(run_dir)
    pack = evidence_pack(run)
    (run / "report_evidence.md").write_text(pack)
    draft = client.send(WRITER + pack, new_conversation=True)
    (run / "report_draft.md").write_text(draft.text)
    final, usage = draft.text, [draft.usage]
    if humanizer:
        edited = client.send(EDITOR.format(skill=humanizer, draft=draft.text), new_conversation=True)
        usage.append(edited.usage)
        if _numbers(edited.text) <= _numbers(draft.text) | _numbers(humanizer) and len(edited.text) > 0.5 * len(draft.text):
            final = edited.text  # the editor may drop words, never bring in a number the draft did not have
            (run / "report_edit_check.json").write_text(json.dumps({"accepted": True}))
        else:
            (run / "report_edited_rejected.md").write_text(edited.text)
            (run / "report_edit_check.json").write_text(json.dumps(
                {"accepted": False, "new_numbers": sorted(_numbers(edited.text) - _numbers(draft.text))[:40]}))
    unsupported = sorted(_numbers(final) - _numbers(pack))
    (run / "REPORT.md").write_text(final.strip() + "\n")
    out = {"chars": len(final), "tool_free": draft.tool_free, "usage": usage,
           "numbers_not_in_evidence": unsupported[:60]}
    (run / "report_check.json").write_text(json.dumps(out, indent=2))
    return out


def _numbers(text: str) -> set[str]:
    """Numbers with two or more significant characters, commas removed: the
    values a reader would check."""
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", text)
            if len(n.replace(",", "").replace(".", "")) >= 2}


__all__ = ["layer_record", "evidence_pack", "write_report"]
