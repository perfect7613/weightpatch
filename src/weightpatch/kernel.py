"""WeightKernel: the persistent Python session the agent acts through.

The agent's only action is one Python block per turn. The kernel runs it in a
namespace that survives between turns (like notebook cells) and holds exactly:

- ``W``      the edit session facade: read tensors, ``save_edit``, ``revert``
- ``tools``  the tool box: inspect / localize / repair / edit / measure
- ``note``   append a line to the notebook that survives a conversation handoff
- ``torch``, ``np``, ``math`` for computing on tensors

The model itself is never in the namespace. It runs only through tools, which
charge the budget. Gradients are off, and code that reaches for training
(``backward``, ``autograd``, ``optim``), the filesystem, the network or
interpreter internals is refused before it runs.

This guard is a policy check that keeps honest runs honest and gives the cheat
review a clean signal. It is not a security boundary: the process boundary is
the worker container (no network, sealed data on a volume it does not mount).
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .budget import Budget, BudgetExceeded

ALLOWED_IMPORTS = {"math", "numpy", "torch", "json", "statistics", "itertools", "collections", "re"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
}
FORBIDDEN_ATTRS = {
    # training: the edit arm is weight-space programming, not optimisation
    "backward", "autograd", "optim", "requires_grad_", "grad", "enable_grad", "set_grad_enabled",
    # escape hatches
    "load", "save", "from_file", "hub", "system", "popen", "subprocess",
}
SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "frozenset",
        "int", "isinstance", "len", "list", "map", "max", "min", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "Exception", "ValueError", "KeyError", "IndexError", "RuntimeError", "TypeError",
        "True", "False", "None",
    )
    if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))
}


class CodeRefused(ValueError):
    """The block broke a kernel rule and was not run."""


@dataclass
class RunResult:
    output: str
    ok: bool
    refused: bool = False
    truncated: bool = False

    def render(self) -> str:
        head = "REFUSED (not run)" if self.refused else ("ok" if self.ok else "ERROR")
        tail = "\n[output truncated]" if self.truncated else ""
        return f"[{head}]\n{self.output}{tail}"


def check_code(code: str) -> None:
    """Static rules. Raises :class:`CodeRefused` naming the first violation."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeRefused(f"syntax error: {e.msg} (line {e.lineno})") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise CodeRefused(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise CodeRefused(f"import from {node.module!r} is not allowed")
            if any(part in FORBIDDEN_ATTRS for part in (node.module or "").split(".")):
                raise CodeRefused(f"import from {node.module!r} is not allowed (no training)")
            for alias in node.names:
                if alias.name in FORBIDDEN_ATTRS:
                    raise CodeRefused(f"import of {alias.name!r} is not allowed (no training)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CodeRefused(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS:
                raise CodeRefused(
                    f"use of .{node.attr} is not allowed "
                    "(no gradients, training, file or interpreter access; act through tools)"
                )


class WeightKernel:
    """Runs agent code blocks in one persistent namespace, charging the budget."""

    def __init__(
        self,
        namespace: dict[str, Any],
        budget: Budget | None = None,
        max_output_chars: int = 6000,
        timeout_seconds: int = 600,
    ):
        self.budget = budget or Budget()
        self.max_output_chars = max_output_chars
        self.timeout_seconds = timeout_seconds
        self.notebook: list[str] = []
        self.namespace: dict[str, Any] = {
            "__builtins__": dict(SAFE_BUILTINS, __import__=_guarded_import),
            "torch": torch,
            "np": np,
            "math": math,
            "note": self._note,
            **namespace,
        }

    def _note(self, text: str) -> None:
        """Append to the notebook carried across conversation handoffs."""
        self.notebook.append(str(text))

    def run(self, code: str) -> RunResult:
        try:
            check_code(code)
        except CodeRefused as e:
            return RunResult(str(e), ok=False, refused=True)
        try:
            self.budget.charge("code_runs")
        except BudgetExceeded as e:
            return RunResult(str(e), ok=False, refused=True)

        buf = io.StringIO()
        ok = True
        with contextlib.redirect_stdout(buf), torch.no_grad(), _time_limit(self.timeout_seconds):
            try:
                exec(compile(code, "<agent>", "exec"), self.namespace)  # noqa: S102
            except BudgetExceeded as e:
                ok = False
                print(f"STOPPED: {e}")
            except _Timeout:
                ok = False
                print(f"STOPPED: block ran longer than {self.timeout_seconds}s")
            except Exception:  # noqa: BLE001 - the agent must see its own errors
                ok = False
                tb = traceback.format_exc(limit=4)
                print(tb.replace(__file__, "<kernel>"))
        out = buf.getvalue()
        truncated = len(out) > self.max_output_chars
        if truncated:
            half = self.max_output_chars // 2
            out = out[:half] + "\n...\n" + out[-half:]
        return RunResult(out.rstrip() or "(no output)", ok=ok, truncated=truncated)


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based limit; a no-op off the main thread or without SIGALRM."""
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable or seconds <= 0:
        yield
        return

    def _raise(signum, frame):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
