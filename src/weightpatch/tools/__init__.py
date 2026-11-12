"""ToolBox: the agent's whole action surface besides raw tensor math.

A tool is a plain function registered under a group (``inspect``, ``localize``,
``repair``, ``edit``, ``measure``). The tool box does four things for every
tool, so no tool has to:

- filters by the agent config's allow-list (``inspect.*``, ``localize.*`` ...)
- charges the budget before the tool runs
- records the call (name, arguments, cost, outcome) for the trace and the
  cheat review
- describes itself to the agent from the function's signature and docstring

Experiment arms are the same tool box with groups removed.
"""

from __future__ import annotations

import fnmatch
import inspect as _inspect
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from ..budget import Budget

GROUPS = ("inspect", "localize", "repair", "edit", "measure")
Cost = dict[str, float] | Callable[..., dict[str, float]] | None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    cost: dict[str, float]
    ok: bool
    seconds: float
    error: str = ""


@dataclass
class _Tool:
    group: str
    fn: Callable[..., Any]
    cost: Cost
    timed: bool

    @property
    def name(self) -> str:
        return f"{self.group}.{self.fn.__name__}"


@dataclass
class ToolBox:
    budget: Budget = field(default_factory=Budget)
    allow: tuple[str, ...] = ("*",)
    calls: list[ToolCall] = field(default_factory=list)
    _tools: dict[str, _Tool] = field(default_factory=dict)

    def register(self, group: str, fn: Callable[..., Any], cost: Cost = None, timed: bool = False) -> None:
        """Add a tool. ``cost`` is a dict of budget charges or a function of the
        call's keyword arguments returning one. ``timed`` also charges wall time
        to ``gpu_seconds``."""
        if group not in GROUPS:
            raise ValueError(f"unknown tool group {group!r}; known: {GROUPS}")
        tool = _Tool(group, fn, cost, timed)
        self._tools[tool.name] = tool

    def allowed(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in self.allow)

    def names(self) -> list[str]:
        return sorted(n for n in self._tools if self.allowed(n))

    def namespace(self) -> SimpleNamespace:
        """``tools`` as the agent sees it: ``tools.<group>.<name>(...)``."""
        groups: dict[str, dict[str, Callable]] = {}
        for name in self.names():
            tool = self._tools[name]
            groups.setdefault(tool.group, {})[tool.fn.__name__] = self._wrap(tool)
        return SimpleNamespace(**{g: SimpleNamespace(**fns) for g, fns in groups.items()})

    def _wrap(self, tool: _Tool) -> Callable[..., Any]:
        sig = _inspect.signature(tool.fn)

        def call(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            cost = tool.cost(**bound.arguments) if callable(tool.cost) else dict(tool.cost or {})
            for unit, amount in cost.items():
                # check every unit first so a refused call charges nothing
                if amount > self.budget.remaining(unit):
                    self.budget.charge(unit, amount)  # raises BudgetExceeded
            for unit, amount in cost.items():
                self.budget.charge(unit, amount)
            start = time.monotonic()
            record = ToolCall(tool.name, _summarise(bound.arguments), cost, ok=True, seconds=0.0)
            try:
                return tool.fn(*bound.args, **bound.kwargs)
            except Exception as e:
                record.ok, record.error = False, f"{type(e).__name__}: {e}"
                raise
            finally:
                record.seconds = time.monotonic() - start
                if tool.timed:
                    self.budget.spent["gpu_seconds"] = (
                        self.budget.spent.get("gpu_seconds", 0.0) + record.seconds
                    )
                    record.cost = dict(record.cost, gpu_seconds=round(record.seconds, 3))
                self.calls.append(record)

        call.__name__ = tool.fn.__name__
        call.__doc__ = tool.fn.__doc__
        return call

    def describe(self) -> str:
        """The tool reference pasted into the agent's first message."""
        lines: list[str] = []
        for group in GROUPS:
            names = [n for n in self.names() if n.startswith(group + ".")]
            if not names:
                continue
            lines.append(f"## tools.{group}")
            for name in names:
                tool = self._tools[name]
                doc = _inspect.getdoc(tool.fn) or ""
                cost = "cost depends on arguments" if callable(tool.cost) else (
                    ", ".join(f"{k} {v:g}" for k, v in (tool.cost or {}).items()) or "free"
                )
                if tool.timed:
                    cost += " + wall time as gpu_seconds"
                lines.append(f"- tools.{name}{_inspect.signature(tool.fn)}  [{cost}]")
                for doc_line in doc.splitlines():
                    lines.append(f"    {doc_line}")
        return "\n".join(lines)


def _summarise(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments as loggable values: tensors become their shape."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            out[k] = f"<tensor {tuple(v.shape)}>"
        elif isinstance(v, (list, tuple)) and len(v) > 8:
            out[k] = f"<{type(v).__name__} of {len(v)}>"
        elif isinstance(v, str) and len(v) > 200:
            out[k] = v[:200] + "..."
        else:
            out[k] = v
    return out
