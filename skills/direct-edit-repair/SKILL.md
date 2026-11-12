---
name: direct-edit-repair
description: Restore behaviour in the few neurons or head slices that carry the drift.
---
# Skill: direct edit
Use when a method over-corrects or costs task accuracy.
- Edit neuron columns or head output slices, never whole matrices. The direct-edit cap for
  this run is stated under "This run"; `tools.edit.list_edits()` shows how much of it you have used.
- `tools.edit.restore_from_reference(name, cols=...)` moves chosen columns back toward the
  aligned model; `tools.edit.project_out(name, direction, cols=...)` stops them writing a direction.
- Cite the tool result that points at each component in `evidence=`.
- Check with `tools.measure.dev_eval` and `tools.edit.list_edits`; `tools.edit.revert` what does not help.
