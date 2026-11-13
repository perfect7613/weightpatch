# 5. Repair methods are staged through the edit session, uncapped and marked

## Status
Accepted.

## Context
SafeTune's recover methods (ReSta, C-DeltaTheta, LoX, NLSR, MSCP, safety vector restore) take
and mutate `nn.Module` models and rewrite many tensors. The harness's guarantees (evidence for
every change, protected tensors, revert, exact edit size) live in the edit session, which works
on state dicts and caps direct edits at 0.1% of entries.

## Decision
A repair tool lends the live model to the method (`HFRunner.borrow`), reads back the tensors it
changed, restores the model, and stages the result with `EditSession.adopt(method, state,
evidence)`. Adopted edits are not capped, keep protected tensors untouched, are revertible, and
carry the method's name. A session with any method edit reports `direct_edit = False`, and the
grader's direct-edit cap applies only to candidates that claim the direct-edit arm.

## Consequences
- One path for every weight change; the grader's diff and the manifest stay exact.
- A method cannot be used to smuggle a large change into the capped arm.
- Methods compose: a direct edit on top of a method's tensor is charged only for what it adds.
- The borrow snapshot costs one extra copy of the model on the GPU while a method runs.
