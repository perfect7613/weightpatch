---
name: select-recover-method
description: Pick the SafeTune recover method your localization evidence supports.
---
# Skill: select a recovery method
`tools.repair` holds SafeTune's training-free recover methods. Each one rewrites many tensors,
is not capped, and moves your candidate out of the direct-edit arm, so use one only when a
direct edit of the localized weights is not enough. Match the method to the evidence:

| Evidence from localization | Method |
|---|---|
| A circuit you trust (two tools agree, faithfulness is high) | `ctheta(circuit=...)`: moves only that circuit |
| A handful of neurons carry it (`safety_neurons`, `rank_writers`) | `nlsr`, or `ctheta` with `make_circuit` |
| The drift is low rank (`svd_of_diff`: a few dominant values) | `lox` or `safety_vector_restore` |
| The drift is spread across the model | `resta` (alpha near 0.25 on Tiny Aya) |

- Every method needs `evidence=`: name the tool result that selects it.
- Tune strength on dev only (`tools.measure.dev_eval`), a few values, never a fine grid.
- Prefer the smallest circuit and the lowest strength that hold the gain. `tools.edit.revert()`
  undoes a method; `tools.repair.methods()` lists what is applied.
