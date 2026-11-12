---
name: read-model-atlas
description: Always first. Read the model atlas before any tool call.
---
# Skill: read the model atlas
The atlas above was built from this exact checkpoint by code. Before your first tool call:
- Read its architecture section, the tensor name template and the quirks.
- Read every finding already logged. Never spend budget re-deriving a fact the atlas already has.
- Use the atlas's tensor names verbatim in `W.tensor(...)` and in edit evidence.
