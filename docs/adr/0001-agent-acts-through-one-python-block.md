# 1. The agent acts through one Python block per turn, with its own tools off

## Context
The agent could be given typed function-calling tools, an MCP server, or a code interface.
The MIT weight-space programming runs and the fast-rlm project both drive the model through
`codex exec` with every built-in tool disabled, and run the model's Python in their own sandbox.

## Decision
One action: a Python block run in the weight kernel. The tool box is a Python object in that
kernel. Codex's built-in tools are off, and every turn's event stream is audited.

## Why
- Tensor work is code. A typed tool per operation would be either too coarse or endless.
- It works on a ChatGPT subscription, which exposes no hosted tools to third parties.
- It is the setup the published results used, so ours are comparable.
- One audit rule ("the turn is tool-free") covers every way the model could act outside the harness.

## Consequences
The kernel's static guard is a policy check, not a security boundary. The worker container is
the boundary: no network, and the sealed data on a volume it does not mount.
