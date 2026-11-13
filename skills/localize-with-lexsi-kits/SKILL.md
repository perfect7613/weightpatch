---
name: localize-with-lexsi-kits
description: Use SafeTune and CircuitKIT to name the exact units before editing, so fewer turns are spent searching.
---
# Skill: localize with Lexsi's kits
The kits answer "which units" in one call each. Use them before hand-written searches, and edit
only where two of them agree. Every localizer takes two dev sets, `a` and `b`, by name.
- SafeTune `tools.localize.safety_neurons(method="activation", a=..., b=..., on="candidate")`:
  MLP neurons whose activation separates the two sets. Run it for the pair you want to change
  (target set vs benign) and for the pair you must keep (harmful vs benign). Neurons named for
  the first pair and not for the second are the candidates. `method="weight"` is a second,
  independent ranking.
- `tools.localize.circuit_units(handle)` lists the units of a result;
  `tools.localize.circuit_overlap(a, b)` says what two results share;
  `tools.localize.make_circuit(units)` turns your own selection into a handle.
- SafeTune `tools.localize.safety_direction(a=..., b=...)`: the best single direction and its layer.
- SafeTune `tools.localize.eap_circuit(a=..., b=...)` and CircuitKIT
  `tools.localize.discover_circuit(a=..., b=...)`: attention heads and MLPs on the path. Check
  `tools.localize.circuit_faithfulness` first; a low score means the circuit is not the mechanism.
- Then edit the agreed units directly (see the direct-edit notes), or pass a circuit handle to
  `tools.repair.ctheta(circuit=...)` to move only that circuit.
Quote the tool result that named each unit in the edit's `evidence=`.
