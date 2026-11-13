# 4. Weights, kernel and grader live on a worker; the control side holds none

## Status
Accepted.

## Context
The agent is driven through the Codex CLI on a user's ChatGPT subscription, which runs where the
user is signed in. The weights need a GPU the user brings from their own provider. A kernel that
edits weights cannot sit on the control side without shipping gigabytes per turn.

## Decision
Split the harness at one seam. The **worker** owns the checkpoints, the weight kernel, the tool
box (including the Lexsi kits) and the grader with its sealed data. The control side owns the
model client and the episode loop. Between them is a six-method protocol of JSON dicts
(`open`, `describe`, `run`, `snapshot`, `export`, `close`). Providers are adapters for that
protocol: in-process, Modal, and HTTP to any machine running `weightpatch worker serve`.

## Consequences
- The episode loop is identical for a laptop toy and a remote L40S; `RemoteKernel` and
  `RemoteToolBox` stand in for the local objects.
- Adding a GPU provider is one small class. Three adapters exist, so the seam is real.
- The sealed held-out pool never leaves the worker. The agent reaches the grader only through
  `tools.measure.submit`, which returns quarantined feedback.
- The worker is trusted with the weights and the sealed data, so whoever operates it can read
  both. For a leaderboard, the grader's worker must be run by the organisers, not the entrant.
