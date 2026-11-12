# WeightPatch

**Tell it what broke. An agent finds the smallest weight-space fix that passes a grader it cannot see.**

WeightPatch is a harness for weight-space repair of language models. A frontier LLM (GPT-6
Astra, through your own ChatGPT subscription) reads and edits a model's weights directly, with
no gradient descent. It works through a tool box of localization and repair tools, and its
result is checked by a grader built to resist gaming.

It follows the setup of MIT's *Sparks of Weight-Space Programming*: the model's own tools are
switched off and it acts by writing one Python block per turn, which the harness runs. The
difference is what the Python can reach. Here it is a tool box wired to interpretability and
safety-repair libraries, so the agent is handed the specific weights that matter, with context.

## Status

Built and tested:

| Module | What it does |
|---|---|
| `model_client` | Drives Astra through `codex exec` on a ChatGPT login, tools off, with a tool-free audit of every turn. |
| `kernel` | The persistent Python session the agent acts through. No gradients, files, network or model access. |
| `edits` | The edit session: what may be edited, edit size as the grader counts it, the direct-edit cap, patch export. |
| `tools` | The tool box (allow-list, budget charging, call log, self-description) and the PyTorch-only tools. |
| `runner` | The one seam through which the model is run, with the session's edits applied. |
| `atlas` | Per-checkpoint context built by code: card, name map, quirks, findings. |
| `episode` | The REPL loop: explain-before-code, handoffs, full trace on disk. |
| `grader` | Spec loading and freezing, six stages cheapest first, hack detectors, quarantine rule. |
| `toy` | A planted-drift model with an exact answer key, for end-to-end runs on a laptop. |

Not built yet: the SafeTune and CircuitKIT tools (`tools/kits.py`), the GPU worker and provider
adapters, the AuditKIT-backed evaluator, drift tiers D1 to D3, GRAFT, and the web app.

## Try it

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest                 # no GPU, no credentials
.venv/bin/weightpatch toy --effort medium  # live: Astra repairs a planted drift
```

The live run needs the Codex CLI (0.167 or newer for `gpt-6-astra`) logged in with ChatGPT.
Point at a specific binary with `--codex-bin` or `WEIGHTPATCH_CODEX_BIN`.

## How an episode works

1. The first message carries the task, the model atlas, the skills, the tool reference and the budget.
2. The model explains its step and replies with one Python block.
3. The weight kernel runs it. `W` reads and edits weights; `tools` runs the tool box.
4. The printed output and the budget line go back as the next message.
5. A reply with no code is the final answer. The edit session is exported as a patch.
6. The patch goes to the grader, which returns numbers for an honest failure and only
   "integrity check failed" for anything that looks like gaming.

Domain terms are defined in [CONTEXT.md](CONTEXT.md); design decisions are in [docs/adr](docs/adr).

## Licence

Code: MIT. The Lexsi kits this harness wraps are LSAL v1.2; Tiny Aya is CC BY-NC 4.0.
