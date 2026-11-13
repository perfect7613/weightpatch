"""The control side's view of a worker: objects an :class:`Episode` can use.

``Episode`` needs a kernel (run code, budget, notebook, edits) and a tool box
(tool reference, call log). These two classes provide exactly that over a
:class:`WorkerHandle`, so the same episode loop drives a local kernel or a
GPU on another continent.
"""

from __future__ import annotations

from typing import Any

from ..budget import Budget
from ..kernel import RunResult
from .providers import WorkerHandle


class RemoteKernel:
    def __init__(self, handle: WorkerHandle, described: dict):
        self.handle = handle
        self.budget = Budget(dict(described["budget"]["limits"]), dict(described["budget"]["spent"]))
        self.notebook: list[str] = []
        self.namespace: dict[str, Any] = {}
        self._edits: list[dict] = []

    def run(self, code: str) -> RunResult:
        out = self.handle.call("run", code=code)
        self.budget.spent = dict(out["budget"]["spent"])
        self.notebook, self._edits = list(out["notebook"]), list(out["edits"])
        return RunResult(out["output"], out["ok"], out["refused"], out["truncated"])

    def edits(self) -> list[dict]:
        return list(self._edits)


class RemoteToolBox:
    def __init__(self, handle: WorkerHandle, described: dict):
        self.handle, self._reference = handle, described["tool_reference"]

    def describe(self) -> str:
        return self._reference

    @property
    def calls(self) -> list:
        from ..tools import ToolCall

        return [ToolCall(**c) for c in self.handle.call("snapshot")["tool_calls"]]


def remote_episode_parts(handle: WorkerHandle) -> dict:
    """Keyword arguments for ``Episode(task, client, **parts)``."""
    described = handle.call("describe")
    return {"kernel": RemoteKernel(handle, described), "toolbox": RemoteToolBox(handle, described),
            "atlas_summary": described["atlas_summary"], "facts": list(described["facts"])}
