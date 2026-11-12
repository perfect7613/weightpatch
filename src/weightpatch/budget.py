"""Budget: the one ledger every costly action in an episode is charged to.

Units are the ones the plan counts in, not "forward passes" alone:

- ``code_runs``       Python blocks executed in the weight kernel
- ``forward_passes``  tool calls that run the model (one batch = one pass)
- ``dev_prompts``     prompts spent through ``measure.dev_eval``
- ``gpu_seconds``     wall time inside localize tools
- ``submissions``     full gradings requested from the grader

A charge that would exceed a limit raises :class:`BudgetExceeded` *before* the
work is done, so a refused action costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNITS = ("code_runs", "forward_passes", "dev_prompts", "gpu_seconds", "submissions")


class BudgetExceeded(RuntimeError):
    """Raised when a charge would take a unit over its limit."""

    def __init__(self, unit: str, wanted: float, remaining: float):
        self.unit, self.wanted, self.remaining = unit, wanted, remaining
        super().__init__(
            f"budget exceeded: {unit} needs {wanted:g}, only {remaining:g} left"
        )


@dataclass
class Budget:
    """Limits and spend per unit. A unit with no limit is unlimited but still counted."""

    limits: dict[str, float] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.limits) - set(UNITS)
        if unknown:
            raise ValueError(f"unknown budget units: {sorted(unknown)}; known: {UNITS}")

    def remaining(self, unit: str) -> float:
        if unit not in self.limits:
            return float("inf")
        return self.limits[unit] - self.spent.get(unit, 0.0)

    def charge(self, unit: str, amount: float = 1.0) -> None:
        if unit not in UNITS:
            raise ValueError(f"unknown budget unit: {unit}")
        if amount > self.remaining(unit):
            raise BudgetExceeded(unit, amount, self.remaining(unit))
        self.spent[unit] = self.spent.get(unit, 0.0) + amount

    def exhausted(self, unit: str) -> bool:
        return self.remaining(unit) <= 0

    def status(self) -> str:
        """One line the agent sees after every code run."""
        parts = []
        for unit in UNITS:
            if unit in self.limits:
                parts.append(f"{unit} {self.spent.get(unit, 0):g}/{self.limits[unit]:g}")
        return "budget: " + (", ".join(parts) if parts else "unlimited")

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {"limits": dict(self.limits), "spent": dict(self.spent)}
