---
name: localize-safety-drift
description: Find where the drift lives, cheapest tool first, and require two channels to agree.
---
# Skill: localize the drift, cheapest first
Work down the funnel. Each rung is a separate feedback channel; stop narrowing when two
independent channels name the same weights.
1. `tools.inspect.diff_weights` for the drift map (free). Which layers and modules moved?
2. `tools.inspect.svd_of_diff` on the top tensors. Is the change low rank?
3. Direction: `tools.localize.refusal_direction`, or SafeTune's `safety_direction` on the
   candidate and on the aligned model. Then `tools.localize.rank_writers` with
   `against_reference=True` for the columns whose write along that direction changed.
4. Circuit: CircuitKIT `discover_circuit` (heads and MLPs) and SafeTune `eap_circuit`. Check
   `circuit_faithfulness` before trusting a circuit; a low score means it is not the mechanism.
5. Neurons, the fine-grained rung: SafeTune `safety_neurons` with method "activation" and
   method "weight". `circuit_overlap` tells you which units both methods name.
6. Weights: within the agreed neurons, compare `W.tensor` with the reference by column.
Confirm causally before a final edit: revert, apply only the suspected weights, and check that
`tools.measure.dev_eval` moves. `note(...)` every conclusion; findings go to the model atlas.
