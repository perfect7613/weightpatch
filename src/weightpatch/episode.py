"""Episode: one repair attempt, as a REPL loop between a model and the kernel.

The loop is the recursive-language-model shape with the model's own tools off:

1. The first message carries the task, the atlas summary, the tool reference,
   the rules and the budgets.
2. The model replies with a short explanation and exactly one Python block.
3. The kernel runs the block; the printed output and the budget line go back
   as the next message.
4. A reply with no code block is the final answer and ends the episode.

Around that loop the episode owns the things that make runs trustworthy and
resumable: the explain-before-code rule, conversation handoffs with the
agent's notebook, the tool-free audit of every turn, and a full trace on disk.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .budget import Budget
from .kernel import WeightKernel
from .model_client import ModelClient, ModelError, Reply
from .tools import ToolBox

CODE_RE = re.compile(r"```(?:python|py|repl)\s*\n(.*?)```", re.DOTALL)

RULES = """\
# How you act
- Your own built-in tools are switched off. You act only by replying with ONE fenced
  ```python block per turn. I run it in a persistent Python kernel and send back what it
  printed. Variables survive between turns.
- Before the block, explain in plain English what the step is for and what you expect.
  A block with no explanation is sent back unrun.
- In the kernel: `W` (read and edit weights), `tools` (the tool box below), `note(text)`
  (your notebook; it survives if the conversation is restarted), plus `torch`, `np`, `math`.
- `W.names(pattern)`, `W.tensor(name)` (float32 copy), `W.original(name)`,
  `W.save_edit(name, tensor, evidence="...")`, `W.revert(name=None)`, `W.list_edits()`.
- You cannot run the model in the kernel. It runs only through tools, and each run is
  charged to your budget.
- No gradient descent or training of any kind: no backward passes, autograd, optimizers or
  iterative fitting of weights to a loss. No files, network or imports beyond numpy, torch
  and the standard maths modules.
- Every edit must cite its evidence: which tool result points at the weights you change.
- Embeddings, the output layer and the final norm cannot be edited. An edit that only
  rescales a whole tensor is refused.
- When you are finished, reply WITHOUT a code block: state the final edit and why it works.
"""


@dataclass
class Turn:
    index: int
    message: str
    reply: str
    code: str | None
    output: str | None
    usage: dict
    tool_free: bool
    seconds: float


@dataclass
class EpisodeResult:
    final_answer: str
    turns: list[Turn]
    stopped: str  # "final_answer" | "max_turns" | "budget" | "model_error" | "tool_use"
    conversations: int
    budget: dict

    @property
    def tool_free(self) -> bool:
        return all(t.tool_free for t in self.turns)


@dataclass
class Episode:
    task: str
    client: ModelClient
    kernel: WeightKernel
    toolbox: ToolBox
    atlas_summary: str = ""
    facts: list[str] = field(default_factory=list)  # run-specific limits, stated once, from code
    skills: list[str] = field(default_factory=list)  # SKILL.md bodies, loaded in order
    max_turns: int = 60
    handoff_every: int = 40
    require_explanation: bool = True
    trace_dir: Path | None = None
    on_turn: Callable[[Turn], None] | None = None

    @property
    def budget(self) -> Budget:
        return self.kernel.budget

    # ---- prompt assembly ----------------------------------------------
    def first_message(self) -> str:
        parts = ["# Task", self.task.strip(), "", RULES]
        if self.facts:
            parts += ["# This run"] + [f"- {fact}" for fact in self.facts] + [""]
        if self.atlas_summary:
            parts += ["# Model atlas (read this first; do not re-derive what it already says)",
                      self.atlas_summary.strip(), ""]
        for body in self.skills:
            parts += [body.strip(), ""]
        parts += ["# Tool box",
                  "This is the complete tool box for this run. A skill may mention a tool that is "
                  "not here; use only the tools listed.",
                  self.toolbox.describe(), "", "# Budget", self.budget.status()]
        return "\n".join(parts)

    def handoff_message(self) -> str:
        notebook = "\n".join(f"- {n}" for n in self.kernel.notebook) or "- (empty)"
        if hasattr(self.kernel, "edits"):  # a kernel on a remote worker
            edits = json.dumps(self.kernel.edits(), default=str)
        elif hasattr(self.kernel.namespace.get("tools"), "edit"):
            edits = json.dumps(self.kernel.namespace["tools"].edit.list_edits(), default=str)
        else:
            edits = "(no edit tools)"
        return "\n".join([
            self.first_message(), "",
            "# This is a continuation",
            "The conversation was restarted. The kernel and its variables are unchanged.",
            "Your notebook so far:", notebook, "", "Your current edits:", edits,
        ])

    # ---- the loop ------------------------------------------------------
    def run(self) -> EpisodeResult:
        turns: list[Turn] = []
        message, new_conv, conversations, unexplained = self.first_message(), True, 1, 0
        stopped, final = "max_turns", ""
        for index in range(1, self.max_turns + 1):
            start = time.monotonic()
            try:
                reply: Reply = self.client.send(message, new_conversation=new_conv)
            except ModelError as e:
                stopped, final = "model_error", str(e)
                break
            new_conv = False
            code = extract_code(reply.text)
            output: str | None = None
            if not reply.tool_free:
                # The model reached one of its own tools: the run is void.
                self._record(turns, index, message, reply, code, None, start)
                stopped, final = "tool_use", reply.text
                break
            if code is None:
                self._record(turns, index, message, reply, None, None, start)
                stopped, final = "final_answer", reply.text
                break
            if self.require_explanation and not has_explanation(reply.text) and unexplained < 2:
                unexplained += 1
                output = "[NOT RUN] Explain the step in plain English before the code block, then resend it."
            else:
                unexplained = 0
                output = self.kernel.run(code).render()
            self._record(turns, index, message, reply, code, output, start)
            if self.budget.exhausted("code_runs"):
                message = output + "\n\nYour code-run budget is used up. Reply without code: state your final edit."
            else:
                message = f"{output}\n\n{self.budget.status()}"
            if self.handoff_every and index % self.handoff_every == 0:
                message = self.handoff_message() + "\n\n# Output of your last block\n" + output
                new_conv, conversations = True, conversations + 1
        result = EpisodeResult(final, turns, stopped, conversations, self.budget.snapshot())
        self._write_state(result)
        return result

    # ---- trace ---------------------------------------------------------
    def _record(self, turns, index, message, reply: Reply, code, output, start) -> None:
        turn = Turn(index, message, reply.text, code, output, reply.usage, reply.tool_free,
                    round(time.monotonic() - start, 2))
        turns.append(turn)
        if self.trace_dir:
            d = Path(self.trace_dir) / "turns" / f"t{index:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "prompt.txt").write_text(message)
            (d / "answer.md").write_text(reply.text)
            if code is not None:
                (d / "code.py").write_text(code)
            if output is not None:
                (d / "output.txt").write_text(output)
            (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in reply.events))
            (d / "run.json").write_text(json.dumps(
                {"usage": reply.usage, "seconds": turn.seconds, "audit": {"tool_free": reply.tool_free}}, indent=2))
        if self.on_turn:
            self.on_turn(turn)

    def _write_state(self, result: EpisodeResult) -> None:
        if not self.trace_dir:
            return
        root = Path(self.trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "final_answer.md").write_text(result.final_answer)
        (root / "tool_calls.jsonl").write_text(
            "\n".join(json.dumps(c.__dict__, default=str) for c in self.toolbox.calls))
        (root / "notebook.md").write_text("\n".join(f"- {n}" for n in self.kernel.notebook))
        (root / "state.json").write_text(json.dumps({
            "stopped": result.stopped, "turns": len(result.turns), "conversations": result.conversations,
            "tool_free": result.tool_free, "budget": result.budget,
        }, indent=2))


def extract_code(text: str) -> str | None:
    """The reply's Python block, or None. With several blocks, the last one runs."""
    blocks = CODE_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def has_explanation(text: str) -> bool:
    """At least a sentence of prose before the first code block."""
    head = text.split("```", 1)[0]
    return len(head.strip()) >= 20
