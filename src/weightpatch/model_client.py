"""ModelClient: the seam between the episode loop and whichever LLM drives it.

One method: send a message, get the model's text back. Adapters:

- :class:`CodexExecClient` drives a model through the Codex CLI on the user's
  ChatGPT subscription. Each turn is one ``codex exec`` (or ``codex exec
  resume <thread>``) with the prompt on stdin, exactly as in the MIT
  weight-space programming harness. Codex runs from an isolated home whose
  config switches every built-in tool off, and every turn's event stream is
  audited to confirm the model used none.
- :class:`ScriptedClient` replays fixed replies, for tests and dry runs.

A Responses-API client (Sign in with ChatGPT token, or an API key) is the third
adapter and slots in here without touching the loop.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Protocol

# Item types a tool-free turn may contain. Anything else means the model
# reached a tool, and the turn is flagged.
TOOL_FREE_ITEM_TYPES = {"agent_message", "reasoning", "error"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ModelError(RuntimeError):
    """The model call failed. ``retryable`` is False for plan-limit and
    eligibility errors, which the loop must not hammer."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Reply:
    text: str
    thread_id: str | None = None
    usage: dict = field(default_factory=dict)
    tool_free: bool = True
    events: list[dict] = field(default_factory=list)


class ModelClient(Protocol):
    def send(self, message: str, new_conversation: bool = False) -> Reply:
        """Send one message. ``new_conversation`` drops the thread (handoff)."""
        ...


class ScriptedClient:
    """Replays a list of replies; records what it was sent."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.sent: list[str] = []
        self.conversations = 0

    def send(self, message: str, new_conversation: bool = False) -> Reply:
        if new_conversation or not self.sent:
            self.conversations += 1
        self.sent.append(message)
        if not self.replies:
            return Reply("Done.", thread_id="scripted")
        return Reply(self.replies.pop(0), thread_id="scripted")


class CodexExecClient:
    """Astra (or any Codex-served model) through ``codex exec``, tools off."""

    def __init__(
        self,
        model: str = "gpt-6-astra",
        effort: str = "high",
        codex_bin: str | None = None,
        home: str | Path | None = None,
        auth_file: str | Path | None = None,
        timeout_seconds: int = 3600,
    ):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        self.model, self.effort, self.timeout = model, effort, timeout_seconds
        self.codex_bin = codex_bin or os.environ.get("WEIGHTPATCH_CODEX_BIN") or shutil.which("codex")
        if not self.codex_bin:
            raise ModelError("codex CLI not found; install it or set WEIGHTPATCH_CODEX_BIN", retryable=False)
        self.home = Path(home) if home else Path(tempfile.mkdtemp(prefix="weightpatch-codex-"))
        self.thread_id: str | None = None
        self._prepare_home(Path(auth_file) if auth_file else Path.home() / ".codex" / "auth.json")

    def _prepare_home(self, auth_file: Path) -> None:
        """An isolated CODEX_HOME: tools-off config, neutral instructions, and a
        link to the user's own login. The login file is never copied or read."""
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "cwd").mkdir(exist_ok=True)
        pkg = resources.files("weightpatch.harness")
        instructions = self.home / "instructions.md"
        instructions.write_text(pkg.joinpath("instructions.md").read_text())
        config = pkg.joinpath("codex_config.toml").read_text().replace("__INSTRUCTIONS__", str(instructions))
        (self.home / "config.toml").write_text(config)
        link = self.home / "auth.json"
        if not link.exists():
            if not auth_file.exists():
                raise ModelError(f"no Codex login at {auth_file}; run `codex login` first", retryable=False)
            link.symlink_to(auth_file)

    def send(self, message: str, new_conversation: bool = False) -> Reply:
        if new_conversation:
            self.thread_id = None
        cmd = [self.codex_bin, "exec"]
        if self.thread_id:
            cmd += ["resume", self.thread_id]
        cmd += [
            "--json", "--skip-git-repo-check", "-m", self.model,
            "-c", f'model_reasoning_effort="{self.effort}"', "-",
        ]
        env = dict(os.environ, CODEX_HOME=str(self.home))
        try:
            proc = subprocess.run(
                cmd, input=message, capture_output=True, text=True, env=env,
                cwd=self.home / "cwd", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise ModelError(f"codex exec timed out after {self.timeout}s") from None
        reply = parse_events(proc.stdout)
        if reply.thread_id:
            self.thread_id = reply.thread_id
        if not reply.text:
            detail = _first_error(reply.events) or proc.stderr.strip()[-400:] or f"exit {proc.returncode}"
            limited = any(k in detail for k in ("usage_limit", "not_eligible", "requires a newer version"))
            raise ModelError(f"codex returned no message: {detail}", retryable=not limited)
        return reply


def parse_events(stdout: str) -> Reply:
    """Fold a ``codex exec --json`` event stream into one Reply."""
    events, texts, usage, thread_id, tool_free = [], [], {}, None, True
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(ev)
        kind = ev.get("type")
        if kind == "thread.started":
            thread_id = ev.get("thread_id")
        elif kind == "turn.completed":
            usage = ev.get("usage", {})
        elif kind and kind.startswith("item."):
            item = ev.get("item", {})
            if item.get("type") not in TOOL_FREE_ITEM_TYPES:
                tool_free = False
            if kind == "item.completed" and item.get("type") == "agent_message":
                texts.append(item.get("text", ""))
    return Reply("\n".join(texts).strip(), thread_id, usage, tool_free, events)


def _first_error(events: list[dict]) -> str:
    for ev in events:
        if ev.get("type") in ("error", "turn.failed"):
            msg = ev.get("message") or ev.get("error", {}).get("message", "")
            if msg:
                return str(msg)[:400]
    return ""
