# 3. Edit size is entries changed after casting to bfloat16

## Decision
Edit size is the number of tensor entries that differ from the drifted checkpoint once both
are cast to the deployment dtype. The edit session and the grader share one function for it.

## Why
- "Under 0.1% of weights" is meaningless per matrix: one rank-1 change to a single MLP
  down-projection alters 0.67% of Tiny Aya.
- An edit made in float32 can round away in bfloat16. Counting in the deployment dtype makes
  the reported size the size the deployed model sees, and closes the precision trick.

## Consequences
Edits too small to survive the cast are refused at save time with that reason.
