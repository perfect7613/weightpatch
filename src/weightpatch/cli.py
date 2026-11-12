"""Command line entry point.

    weightpatch toy [--model gpt-6-astra] [--effort medium] [--out runs/toy]

runs one live episode on the planted-drift toy: a real agent, through the
Codex CLI on your ChatGPT subscription, repairing a tiny model through the
tool box. It needs no GPU and no gated model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .atlas import Atlas
from .budget import Budget
from .edits import EditPolicy, EditSession
from .episode import Episode
from .kernel import WeightKernel
from .model_client import CodexExecClient
from .tools import ToolBox
from .tools.native import ToolContext, register_native_tools
from .toy import TOY_TASK, make_planted_drift, score_repair

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


def load_skills(names: list[str], skills_dir: Path = SKILLS_DIR) -> list[str]:
    """SKILL.md bodies (front matter stripped), in the order asked."""
    bodies = []
    for name in names:
        text = (skills_dir / name / "SKILL.md").read_text()
        if text.startswith("---"):
            text = text.split("---", 2)[2]
        bodies.append(text.strip())
    return bodies


def build_toy_episode(client, out: Path, seed: int = 0, max_turns: int = 20, cap: float = 0.01):
    drift = make_planted_drift(seed)
    session = EditSession(drift.drifted, EditPolicy(max_fraction_entries=cap))
    budget = Budget({"code_runs": max_turns, "forward_passes": 0})
    box = ToolBox(budget, allow=("inspect.*", "localize.rank_writers", "edit.*"))
    ctx = ToolContext(session, reference=drift.aligned)
    register_native_tools(box, ctx)
    atlas = Atlas.build(out / "atlas", "toy/tiny-cohere2", drift.drifted, drift.config)
    kernel = WeightKernel({"W": session.view(), "tools": box.namespace()}, budget)
    episode = Episode(
        TOY_TASK, client, kernel, box, atlas_summary=atlas.summary(),
        facts=[f"Direct-edit cap: {cap:.2%} of the model's {session.total_entries:,} entries "
               f"({int(cap * session.total_entries):,} entries), counted after casting to bfloat16.",
               "This run is weights-only: no tool runs the model."],
        skills=load_skills(["read-model-atlas", "localize-safety-drift", "direct-edit-repair"]),
        max_turns=max_turns, trace_dir=out / "trace",
        on_turn=lambda t: print(f"turn {t.index}: {'code' if t.code else 'final'} "
                                f"({t.seconds}s, tool_free={t.tool_free})", flush=True),
    )
    return episode, session, drift, ctx, atlas


def cmd_toy(args: argparse.Namespace) -> int:
    out = Path(args.out)
    client = CodexExecClient(model=args.model, effort=args.effort, codex_bin=args.codex_bin)
    episode, session, drift, ctx, atlas = build_toy_episode(client, out, args.seed, args.max_turns)
    result = episode.run()
    for f in ctx.findings:
        atlas.add_finding("agent", f.get("tool", "?"), f)
    session.export(out / "patch")
    score = score_repair(session, drift)
    report = {"stopped": result.stopped, "turns": len(result.turns), "tool_free": result.tool_free,
              "score": score, "key": {"tensor": drift.tensor, "cols": drift.cols},
              "usage": [t.usage for t in result.turns]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("stopped", "turns", "tool_free", "score", "key")}, indent=2))
    print("\nFinal answer:\n" + result.final_answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weightpatch")
    sub = parser.add_subparsers(dest="command", required=True)
    toy = sub.add_parser("toy", help="live episode on the planted-drift toy")
    toy.add_argument("--model", default="gpt-6-astra")
    toy.add_argument("--effort", default="medium")
    toy.add_argument("--codex-bin", default=None)
    toy.add_argument("--out", default="runs/toy")
    toy.add_argument("--seed", type=int, default=0)
    toy.add_argument("--max-turns", type=int, default=20)
    toy.set_defaults(fn=cmd_toy)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
