---
name: improve-task-accuracy
description: Raise accuracy on a task by finding a habit that costs answers and editing the few neurons that carry it.
---
# Skill: improve accuracy on a task
A small weight edit cannot teach a model to reason. It can remove a habit that throws away
answers the model was able to give: looping, running out of tokens, a wrong answer format,
stopping after a first guess, drifting off the question. Look for a habit.
1. Run `tools.measure.task_eval(name)` once on the released weights and read the wrong items:
   `finished`, `looping`, `tokens`, `final` against `target`, `head` and `tail`, and
   `tools.measure.answer(name, i)` for full texts. Sort the failures into kinds and count them.
   Pick the largest kind that looks like a habit. If most failures are plain wrong reasoning,
   say so: the honest result may be a small gain.
2. Find what the habit looks like in tokens (a word that starts each loop, a missing
   "Answer:" line). `tools.inspect.token_ids` gives ids; `tools.measure.token_probs` shows how
   likely those tokens are at the point where things go wrong.
3. Localize without running the model first. `tools.localize.token_writers(tokens)` reads, from
   the weights alone, which MLP neurons push those tokens (the output head reuses the embedding
   table). It costs nothing and scans every neuron.
4. Then localize with activations: `tools.localize.neuron_contrast(a, b, ...)` between items
   that show the habit and items that do not (pass the model's own answers as `a_answers` /
   `b_answers` to read across the answer, not only the prompt). Kit tools take the same two
   groups by name after `tools.measure.define_set`.
5. Before editing a neuron, `tools.localize.neuron_trace` it across whole answers. A column edit
   acts wherever the neuron fires. A neuron that fires only where the habit starts is a good
   target; one that fires all through the answer will damage content.
6. Test the idea cheaply: `tools.measure.steer(direction, layers, task="<set>")` runs the task
   under an activation change, with the neuron's output column (or a difference of mean
   activations between solved and unsolved items) as the direction, or check with `tools.measure.token_probs` that the edit moves the token.
7. Edit one output column at a time (double it, halve it, or remove its write along a
   direction). Run `tools.measure.task_eval` after each. Keep an edit only if `vs_released` is
   significant, or `fixed` clearly exceeds `broke` and a second, untouched slice (`offset=`)
   agrees. Otherwise revert. Gains inside the noise are not gains.
   When no single hypothesis stands out, search wider instead of stopping: collect 30 to 60
   candidate neurons from several sources (`neuron_contrast` between solved and unsolved items
   read across the answers, `token_writers` for tokens that begin wrong turns, kit rankings), and
   score them all with `tools.measure.sweep(name, trials)`, trying scales such as 0, 0.5 and 2.
   Confirm the top few on a slice the sweep did not use. One hour of sweeping is a normal cost.
8. Combine kept edits and measure the combination; edits do not always add.
9. Stop when two rounds add nothing. Check answers still read normally and refusals of harmful
   requests are unchanged (`tools.measure.dev_eval`), then submit.
