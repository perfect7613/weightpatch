---
name: reduce-metric-gaming
description: Make a model do the task it was asked instead of gaming the score it was told about.
---
# Skill: reduce metric gaming
The prompts state a task and how the answer will be scored. A gamed answer is written for the
score (stuffed keywords, padding, repetition, shortcuts); an honest one does the task. The
unwanted behaviour is a response to one cue: the scoring sentence in the prompt.
1. Run `tools.measure.task_eval(name)` and read the labels. Read several gamed answers in full
   (`tools.measure.answer`) and the item's metric and known cheat (`tools.measure.examples`).
   Group the gamed items by how they cheat; one mechanism may not cover all groups.
2. Every item comes with one gamed and one honest example answer to the same prompt. That is a
   clean contrast with the prompt held fixed: `tools.localize.neuron_contrast(prompts, prompts,
   a_answers=gamed, b_answers=honest, position="mean")` names neurons that are active while
   writing a gamed answer. Contrast also the prompt alone against the same task with the scoring
   sentence removed (your own lists of texts) to find what reads the cue.
3. `tools.measure.preference(name)` is the cheap measure: how much more likely the gamed example
   is than the honest one. Use it for sweeps; confirm on real answers with `task_eval`.
4. Trace candidates across whole answers with `tools.localize.neuron_trace`. A neuron for
   "repeat this word" fires on ordinary lists too; editing it will damage ordinary writing.
   The behaviour is likely carried by a direction that many components write, more than by
   single neurons. Build directions from mean activations (`tools.localize.activations`): prompt
   with the scoring sentence minus the same prompt without it; gamed example minus honest
   example. Then `tools.measure.steer(direction, layers, task="<set>")` runs the task itself
   under the intervention (`preference_only=True` for a cheap first sweep over layers and
   strengths). Edit weights only after a steering test shows the gamed share falling.
5. Test with `tools.measure.steer` before editing, then edit the fewest output columns, at
   partial strength. After each edit check `task_eval` (`vs_released`, `fixed`, `broke`), that
   answers are still real attempts (label "broken" must not rise), and that refusals and
   benign answers are unchanged (`tools.measure.dev_eval`).
6. Keep part of the set out of your localization (`offset=`) and confirm the gain there.
