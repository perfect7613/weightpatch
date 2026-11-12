---
name: select-recover-method
description: Try training-free recovery methods at documented settings before direct edits.
---
# Skill: select a recovery method
- Try the training-free methods in `tools.repair` at their documented settings for this
  model before writing a direct edit.
- Tune strength on dev only (`tools.measure.dev_eval`), a few values, never a fine grid.
- Prefer the smallest mask and the lowest strength that hold the gain.
