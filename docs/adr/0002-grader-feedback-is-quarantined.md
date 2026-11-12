# 2. Hack rejections are never explained to the agent

## Decision
An honest failure returns its stage and numbers. An integrity check, hack detector or cheat
review that fires returns "integrity check failed" and nothing else; the detail goes to a
quarantine log only humans read.

## Why
GRAFT's finding: penalising what a monitor detects teaches the behaviour the monitor cannot
see. Telling an agent which detector caught it is the same mistake at the harness level.

## Consequences
An agent that trips a detector by accident gets little to go on. That cost is accepted; the
self-audit skill tells it what to check before finishing.
