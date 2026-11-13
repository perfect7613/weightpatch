# WeightPatch

**Tell it what broke. An agent finds the smallest weight-space fix that passes a grader it cannot see.**

WeightPatch is an agent-driven safety-repair harness, built for the Lexsi Labs AI Safety and
Alignment Hackathon on top of Lexsi's open stack: **SafeTune**, **CircuitKIT**, **AuditKIT** and
**CuratorKIT**.

A frontier LLM (GPT-6 Astra, through the user's own ChatGPT subscription) reads and edits a
model's weights directly, with no gradient descent. MIT's *Sparks of Weight-Space Programming*
showed this is possible when the model can only run code and watch outputs. WeightPatch asks
what happens when the same agent is handed Lexsi's interpretability and repair tools instead:
it is told which direction, which circuit and which neurons carry the behaviour, and it has
SafeTune's recover methods to act on them.

## Why Lexsi's stack is the point

An agent editing weights blind has one feedback channel: change something, run the model, look.
Lexsi's kits turn that into several independent channels that each point at specific weights,
and one evaluation layer that makes the result believable. Each kit has a distinct job here.

| Lexsi kit | Its job in WeightPatch | What the agent or grader calls |
|---|---|---|
| **SafeTune** | Localize the safety behaviour, then repair it without training | `localize.safety_direction`, `localize.safety_neurons` (weight and activation methods), `localize.eap_circuit` (EAP / EAP-IG down to single heads); `repair.resta`, `repair.ctheta`, `repair.lox`, `repair.nlsr`, `repair.mscp`, `repair.safety_vector_restore` |
| **CircuitKIT** | Find the circuit and say how far to trust it | `localize.discover_circuit` on a TransformerLens copy of the candidate, `localize.circuit_faithfulness` (patching and ablation scores) |
| **AuditKIT** | Measure every candidate, with judges and statistics | The grader's evaluator: `GuardJudge` guard models as two independent judges, `RefusalDetector`, one fingerprinted `evaluate` run per measurement, `compare().significance()` |
| **SafeTune red team** | Make the damage and the attacks | Drift tier D1 (`AbliterationAttack`), best-of-N augmentations for the dynamic held-out suite |
| **CuratorKIT** | Curate the fine-tuning mixes | Planned for the D3 data pass; not wired in yet |

Three things in this design exist only because the kits compose:

- **CircuitKIT feeds SafeTune.** A circuit discovered by CircuitKIT comes back as a handle
  (`circuit:3`). `repair.ctheta(circuit="circuit:3")` then moves only that circuit. The agent
  never retypes a unit list, and it can build its own handle from its own neuron-level findings.
- **Cross-method agreement is a tool.** `localize.circuit_overlap` compares what SafeTune's
  weight method, SafeTune's activation method, EAP and CircuitKIT each name. Units that
  independent methods agree on are the ones worth editing. Faithfulness scores say when a
  circuit is not the mechanism and should not be edited on.
- **Tiny Aya works out of the box.** CircuitKIT ships the TransformerLens port for Cohere2, and
  SafeTune documents the settings that suit it. The harness loads a Cohere2 checkpoint with the
  agent's edits applied straight into CircuitKIT.

SafeTune 0.1.12, CircuitKIT 0.2.3 and AuditKIT 1.1.11 are the versions this code was verified
against. Every call into them was checked against their source and exercised in the tests.

## What is built

| Module | What it does |
|---|---|
| `model_client` | Drives Astra through `codex exec` on a ChatGPT login, the model's own tools off, with a tool-free audit of every turn. |
| `kernel` | The persistent Python session the agent acts through. No gradients, files, network or model access. |
| `edits` | The edit session: protected tensors, evidence for every change, edit size in the deployment dtype, the 0.1% direct-edit cap, method edits, patch export. |
| `tools` | The tool box (allow-list, budget charging, call log, self-description) and the PyTorch-only tools. |
| `tools/kits` | The SafeTune and CircuitKIT tools above, in the same tool box. |
| `runner` | The one seam through which the model runs. Lends the live model to a repair method and serves the drifted and aligned references from one copy. |
| `atlas` | Per-checkpoint context built by code: card, name map (HF names, TransformerLens hooks, CircuitKIT node ids), quirks, findings. |
| `episode` | The loop: explain before code, one Python block per turn, handoffs, full trace on disk. |
| `grader` | Frozen, hashed spec; six stages cheapest first; hack detectors; quarantined feedback. |
| `evaluator`, `suites` | The AuditKIT-backed evaluator and the benchmark suites, including a dynamic suite rebuilt for every submission. |
| `drift` | Three tiers of damage to repair (below). |
| `graft` | GRAFT gradient routing on one GPU, and merging its adapters into a plain checkpoint. |
| `worker` | The half that sits next to the weights, and the provider adapters that reach it: local, Modal (L40S), HTTP to any GPU machine. |
| `toy` | A planted-drift model with an exact answer key, for end-to-end runs on a laptop. |

Not built yet: the web app (Sign in with ChatGPT, connect a GPU provider, ask for a repair), the
LLM cheat reviewer, the goal router for prompts beyond safety drift, and the experiments.

### What has and has not been run

- All 64 tests pass on CPU. They run the real kits on a tiny random Cohere2 model: all six
  SafeTune recover methods, the SafeTune localizers, CircuitKIT discovery and faithfulness,
  AuditKIT evaluation and significance, GRAFT training and merging, the three drift tiers, and
  the worker over the local and HTTP providers.
- A live episode ran on the toy: Astra, through the Codex CLI with its own tools off, found and
  exactly repaired a planted drift (383 entries, 0.21% of the model, no collateral change).
- **Nothing has been run on Tiny Aya yet.** The checkpoints are gated and access is pending, so
  there are no repair numbers to report. The Modal worker is written but has not been deployed.
  The toy model has no safety behaviour, so the toy says nothing about how well repairs work.

## Drift tiers

A repair harness is only as credible as the damage it is tested on.

| Tier | How the damage is made | What is known about it |
|---|---|---|
| **D1** abliteration | SafeTune's `AbliterationAttack` removes the refusal direction from the weights | The direction and the tensors, exactly |
| **D2** GRAFT-planted | A fine-tune with GRAFT routing; unsafe examples train the FORGET neurons, which are merged in and left on | A neuron-level answer key. The oracle repair (zero those neurons) restores the aligned model exactly, which bounds any repair |
| **D3** contaminated LoRA | An ordinary LoRA fine-tune with about 12% unsafe examples, merged | Only which examples were unsafe. This is the realistic case |

D2 rests on one observation: a SwiGLU adapter beside a SwiGLU MLP is exactly extra neurons in
that MLP. So GRAFT's adapters fold into an ordinary, slightly wider checkpoint that every tool
here can read, and the same merge with FORGET dropped is a deployable prevention recipe.

## How an episode works

1. The first message carries the task, the model atlas, the skills, the tool reference, the
   run's limits and the budget.
2. The model explains its step and replies with one Python block.
3. The weight kernel on the worker runs it. `W` reads and edits weights; `tools` is the tool box.
4. The printed output and the budget line go back as the next message.
5. `tools.measure.submit()` sends the current weights to the grader. An honest failure returns
   its stage and numbers. Anything that looks like gaming returns "integrity check failed" and
   the detail goes to a log only humans read.
6. A reply with no code is the final answer. The edit session is exported as a patch.

The grader recomputes the weight diff itself, checks capability against the drifted model,
harm on dev and on a sealed held-out pool, fresh perturbed variants per submission, refusals on
safe prompts against the aligned model, and then looks for the known shortcuts: memorised
answers, collapsed or looping output, judge gaming, gains from a few prompts, and edits that
live in the last block.

## Try it

```bash
uv venv && uv pip install -e ".[dev,lexsi]"
.venv/bin/python -m pytest                    # no GPU, no credentials
.venv/bin/weightpatch kits                    # which Lexsi kits are installed
.venv/bin/weightpatch toy --effort medium     # live: Astra repairs a planted drift
.venv/bin/weightpatch run --spec configs/episodes/toy.yaml --provider local
```

Live runs need the Codex CLI (0.167 or newer for `gpt-6-astra`) logged in with ChatGPT. Point at
a specific binary with `--codex-bin` or `WEIGHTPATCH_CODEX_BIN`.

On a GPU, with access to the Tiny Aya checkpoints:

```bash
# build the damage
weightpatch drift d3 --model CohereLabs/tiny-aya-global --out /runs/drift/d3

# option A: Modal, one L40S
modal secret create huggingface HF_TOKEN=...
modal deploy -m weightpatch.worker.modal_app
weightpatch run --spec configs/episodes/tiny-aya-d3.yaml --provider modal

# option B: any machine with a GPU
weightpatch worker serve --port 8765 --token "$TOKEN"         # on the GPU machine
weightpatch run --spec configs/episodes/tiny-aya-d3.yaml \
    --provider http --url https://your-gpu:8765 --token "$TOKEN"
```

Domain terms are defined in [CONTEXT.md](CONTEXT.md); design decisions are in [docs/adr](docs/adr).

## Credits and licences

- **Lexsi Labs**: SafeTune, CircuitKIT, AuditKIT and CuratorKIT (LSAL v1.2). This project wraps
  them and does not redistribute them.
- **GRAFT** (Ward et al., arXiv 2610.17062; github.com/jnward/GRAFT, MIT): `graft.py` is a
  compact port of its adapter and split-moment optimizer.
- **Sparks of Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab; MIT CSAIL): the
  tools-off, one-block-per-turn setup.
- **Tiny Aya** (Cohere Labs): CC BY-NC 4.0.

WeightPatch's own code is MIT.
