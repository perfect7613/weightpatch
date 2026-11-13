---
name: improve-released-model
description: Improve a released model on one stated behaviour with a small, located weight edit.
---
# Skill: improve a released model
There is no better checkpoint to copy from. The fix has to come from understanding where the
unwanted behaviour is computed in this model.
1. Read the measured profile in the atlas. It says how large the problem is and what must not
   get worse. Do not spend budget measuring it again.
2. Read examples. `tools.measure.dev_prompts(name)` and `tools.measure.sample(prompts)` show
   which prompts trigger the behaviour and what the model says. Split the target dev set into
   prompts that show the behaviour and prompts that do not: that split is your sharpest contrast.
3. Localize with contrasts, not with one set. The components you want respond to the target set
   and not to the set that must keep its behaviour (for over-refusal: active on refused safe
   prompts, but not the ones that carry refusals of truly harmful prompts).
   `tools.localize.refusal_direction(a=..., b=...)` gives a direction per layer for any two dev
   sets; `tools.localize.rank_writers` lists the neuron columns and head slices that write it;
   `tools.localize.activations` checks that they fire on the prompts you expect.
   Test before you edit: `tools.measure.steer(direction, layers, ...)` changes activations only.
   Sweep layers and strengths with `margin_only=True` (cheap) to find where removing a direction
   moves the target set and not the others; that tells you which layers' writers to edit and how
   hard. `tools.measure.refusal_margin` is the per-prompt, continuous version of the refusal
   rate: use it to compare small edits, and `tools.measure.dev_eval` to confirm.
   If single neurons give only a small effect, the behaviour is probably carried by a direction
   many components write: edit how the strongest writers write that direction, at partial strength.
4. Edit the fewest columns that move the behaviour: scale a neuron's output column, remove its
   write along a direction with `tools.edit.project_out(name, direction, cols=...)`, or move it
   toward the base model with `tools.edit.restore_from_reference`. Start small and at partial strength.
5. After each edit run `tools.measure.dev_eval()` on every dev set. Keep the edit only if the
   target moves and the sets that must not move stay put. `tools.edit.revert` otherwise.
6. Hold out part of the target dev set from your own localization and check the gain carries to
   it before you submit. The grader uses prompts you have not seen and perturbed variants.
`note(...)` each conclusion; findings go to the model atlas.
