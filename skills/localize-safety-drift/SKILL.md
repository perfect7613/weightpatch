---
name: localize-safety-drift
description: Find where the drift lives, cheapest tool first.
---
# Skill: localize the drift, cheapest first
Work down the funnel and stop narrowing when the evidence agrees:
1. `tools.inspect.diff_weights` for the drift map (free). Which layers and modules moved?
2. `tools.inspect.svd_of_diff` on the top tensors. Is the change low-rank?
3. `tools.localize.refusal_direction` (one forward pass), then `tools.localize.rank_writers`
   with `against_reference=True` to find the columns whose write along that direction changed.
4. Circuit and neuron tools (CircuitKIT discovery, SafeTune localizers) only if the cheap
   steps disagree. They are the most expensive calls.
Components that several channels point to go first. `note(...)` every conclusion.
