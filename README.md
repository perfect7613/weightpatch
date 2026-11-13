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
| `atlas` | Per-checkpoint context built by code: card, a wiring description and diagram of one layer, name map (HF names, TransformerLens hooks, CircuitKIT node ids), quirks, the measured baseline profile, findings. |
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

- All 67 tests pass on CPU. They run the real kits on a tiny random Cohere2 model: all six
  SafeTune recover methods, the SafeTune localizers, CircuitKIT discovery and faithfulness,
  AuditKIT evaluation and significance, GRAFT training and merging, the three drift tiers, and
  the worker over the local and HTTP providers.
- A live episode ran on the toy: Astra, through the Codex CLI with its own tools off, found and
  exactly repaired a planted drift (383 entries, 0.21% of the model, no collateral change).
- On Tiny Aya (3.35B), inside Modal on one L40S: every SafeTune and CircuitKIT tool runs on
  the real model, and the grader runs with AuditKIT guard judges (Granite Guardian and Llama
  Guard; WildGuard is gated).

### Improving Tiny Aya as released: fewer refusals of safe requests

This is the main use: no damage is planted. The model is `CohereLabs/tiny-aya-global` exactly as
released, and the goal is one measured weakness. Profiled with the grader's suites and judges,
it gives a harmful answer on 14% of HarmBench prompts and refuses about 30% of OR-Bench hard
prompts, which are safe to answer. Astra was asked to cut those refusals without the model
starting to answer harmful requests.

Astra ran through the Codex CLI in the same container with its own tools off. It used
SafeTune's activation and weight rankings on two contrasts (borderline vs benign, harmful vs
benign) to pick MLP neurons named for the first and not the second, tested directions with
activation steering before touching a weight, and then amplified a borderline-specific
direction, orthogonal to the harmful-refusal direction, in 195 MLP output columns of layers 20,
22, 23 and 24. The grader accepted the first submission.

| Measured by the grader on prompts the agent never saw | Released | Edited |
|---|---|---|
| Refusals of safe requests, OR-Bench hard, 200 sealed prompts | 31.5% | **10.0%** (-21.5 points, 95% interval -27.5 to -15.5) |
| Same, second sealed slice of 120 | 30.0% | 9.2% |
| Same, perturbed variants made for this submission (40) | | -10 points (interval -20 to -2.5) |
| Harmful answers, HarmBench (80), two guard judges | | -2.5 points, not significant |
| Harmful answers, sealed AdvBench and JailbreakBench (100) | | no change |
| MMLU, MILU Hindi, MedMCQA (80 each) | | +1.3, 0, -1.3 points, none significant |
| Weights changed | | 398,328 entries in 4 tensors, 0.012% of the model |

The run took 23 turns, 22 code blocks and one submission, about 58 minutes of GPU time, and
5.9M input tokens (5.4M of them cached) with 109k output tokens.

What this does and does not show: one run, one model, one goal. The intervals are from a paired
bootstrap over 200 prompts, so the size of the gain is known to about 6 points either way. The
harm checks would catch a rise of more than 5 points, not a smaller one. An earlier run with a
smaller measurement budget and no steering tool was rejected by the grader (-2.5 points, not
significant), so the result depends on the tools the agent has.

### Drift repair on Tiny Aya (validation with a known answer; one run, quick grader)

Drift tier D1: SafeTune's abliteration attack on `tiny-aya-global` changed 72 tensors (22.7% of
entries) and took refusals on harmful dev prompts from 87.5% to 0%.

Astra, running through the Codex CLI inside the same container with its own tools off, repaired
it in 20 turns and 2 submissions. Its first submission was rejected honestly (refusals on safe
prompts rose 8.7 points, limit 5); it halved the strength and the second was accepted.

| | Entries changed | Held-out harm (drifted 0.85) | Perturbed-prompt harm | Capability |
|---|---|---|---|---|
| Oracle: restore all 72 tensors | 758,601,877 (22.7%) | -84 points | -70 points | no significant change |
| **Astra's direct edit** | **237,924 (0.0071%)** | **-77 points, to 0.08** | -27 points | no significant change |

The edit is 125 MLP neurons in layers 14 to 29, the ones SafeTune's weight-based and
activation-based localizers both named, moved toward the aligned model. It is about 3,200 times
smaller than the oracle and recovers most of its held-out gain.

What this does and does not show: it is one run, on the easiest tier, graded with the quick spec
(60 to 100 items per suite, so intervals are wide), and the edit uses the aligned checkpoint as
a reference. It is much weaker than the oracle on perturbed prompts. D2, D3, the full grader,
repeated runs and the no-kit ablation have not been run.

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
# option A: everything inside Modal, on one L40S (the Codex CLI runs in the container too)
modal secret create huggingface HF_TOKEN=...
modal run -m weightpatch.worker.modal_app::codex_login        # sign in with ChatGPT, device code
modal run -m weightpatch.worker.modal_app::smoke              # the kits on Tiny Aya
modal run -m weightpatch.worker.modal_app::build_sealed
modal run -m weightpatch.worker.modal_app::build_drift --tier d1
modal run -m weightpatch.worker.modal_app::grade --which oracle
modal run -m weightpatch.worker.modal_app::profile             # where the released model has room to improve
modal run --detach -m weightpatch.worker.modal_app::main --spec configs/episodes/tiny-aya-overrefusal.yaml --name run1
modal run --detach -m weightpatch.worker.modal_app::main --spec configs/episodes/tiny-aya-d1-quick.yaml --name d1

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
