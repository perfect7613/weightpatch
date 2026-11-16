# WeightPatch: an agent that edits model weights, with a grader it cannot see

Draft report, 16 November 2026. Lexsi Labs AI Safety and Alignment Hackathon, Track 2b (AI Safety and Correction).

- Code: https://github.com/perfect7613/weightpatch
- Models: https://huggingface.co/collections/Perfect7613/weightpatch-models-6af7fed7120d5e7caeb6875e
- Runs, traces and baselines: https://huggingface.co/collections/Perfect7613/weightpatch-datasets-and-traces-6af7fed795c64cce97b28a54

## Abstract

WeightPatch is a harness in which a language-model agent improves a released model by editing its weights directly. There is no fine-tuning and no adapter. The agent (GPT-6 Astra, with its own tools turned off) writes one block of Python per turn. That code runs in a sandbox that holds the weights of `CohereLabs/tiny-aya-global` and a set of tools built on Lexsi Labs' SafeTune, CircuitKIT and AuditKIT. The kits tell the agent where a behaviour lives (safety directions, safety neurons, circuits) and how the model behaves (refusal detection, guard judges). The agent uses them to make small edits and to measure each one against the released model.

A grader decides whether an edit counts. It reloads the weights, checks that at most 0.1% of entries changed, checks capability on MMLU, MILU Hindi and MedMCQA, measures the target behaviour on prompts the agent never saw, measures a side-effect guard, and runs detectors for edits that game the measurement. The agent sees only the verdict.

We ran the harness on four goals with the same code and different specs.

Over-refusal was accepted. An edit to 398,328 entries (0.012% of the weights) in 195 MLP output columns of layers 20, 22, 23 and 24 cut refusals of safe requests from 31.5% to 10.0% on 200 unseen OR-Bench hard prompts. A separate script that does not use the harness reproduced this: 33.5% to 14.0% on 600 prompts, with MedMCQA (4,183 questions) and MMLU (2,000) unchanged and refusals of harmful requests unchanged.

Multilingual safety produced a significant gain on unseen data that the grader did not accept. Released Tiny Aya gives harmful answers to about 11% of MultiJail requests in nine languages, against 2% in English. An edit to 53,183 entries (0.0016%) in 26 MLP output columns of layers 23 to 26 cut harmful answers from 16.2% to 11.6% on 450 held-out requests, with no rise in refusals of safe requests. The grader rejected it on a check that we later found could not pass any fix of a rare yes/no problem. We corrected the check afterwards and say so plainly: that is a grader loosened after it rejected a result.

Reward hacking and math gave no accepted edit. For reward hacking the agent found a single-column edit that helped on its dev set and did not carry over to unseen prompts. For math it found no edit that fixed more answers than it broke, and submitted nothing.

Every run is published with its trace, including the failures: what the agent wrote, the code it ran, and each grader verdict.

## 1. What the harness is

| Part | What it does |
|---|---|
| Agent | GPT-6 Astra through the Codex CLI, its own tools off. One Python block per turn, in the style of a recursive language-model loop. |
| Kernel | Runs the agent's code against the weights with an allow-list of builtins, budgets for code runs, forward passes, prompts, GPU seconds and submissions, and a record of every edit. |
| Tools | `inspect`, `localize`, `repair`, `edit`, `measure`. SafeTune gives safety directions and safety neurons (by activation and by weight). CircuitKIT gives circuits and neuron scores. AuditKIT gives the refusal detector, the guard judges and paired evaluation. Native tools add per-item scoring against the released model, neuron contrasts, steering, token lens and sweeps of single-neuron trials. |
| Atlas | A description of the model the agent starts from: architecture, wiring, a profile of the released model, and findings from the run so far. |
| Grader | Stages: integrity, capability, quick target check with a guard, sealed held-out set with a guard, perturbed prompts, detectors for gamed edits. Paired bootstrap, alpha 0.05. The agent gets the stage and the numbers of an honest failure and only a coarse reason for an integrity failure. |
| Report agent | Writes each run's report from an evidence file built by code. Numbers that are not in the evidence are flagged. |

Everything runs on Modal on one GPU. The mode used here is improve-from-base: there is no better checkpoint to copy from. The model's own released weights are the baseline, and the comparison reported is always released model against edited model.

## 2. Results

### 2.1 Over-refusal (accepted)

Goal: Tiny Aya refuses many safe requests that look sensitive. Reduce that without making it answer harmful requests.

The agent found a direction that separates borderline-but-safe requests from harmful ones, kept the part orthogonal to the harmful direction, and amplified it in 195 MLP output columns of layers 20, 22, 23 and 24. The run took 23 turns and about 58 minutes of GPU time, with 5.9M input tokens (5.4M cached) and 109k output tokens.

| Check (grader) | Released | Edited | Result |
|---|---:|---:|---|
| Held-out OR-Bench hard, refusals (200) | 31.5% | 10.0% | -21.5 points (-27.5 to -15.5) |
| Second slice (120) | 30.0% | 9.2% | significant |
| Perturbed prompts | | | -10 points |
| HarmBench, harmful answers | | | -2.5 points, not significant |
| MMLU / MILU Hindi / MedMCQA (80 each) | | | +1.3 / 0 / -1.3 points, none significant |

An earlier run of the same goal was rejected (a 2.5-point change, not significant) and is published too.

**Independent check.** A grader that belongs to the system that made the edit is not independent evidence. `scripts/verify_published_model.py` imports only `torch`, `transformers` and `datasets`, downloads both models from Hugging Face and runs them on the same prompts.

| Measurement | Prompts | Released | Edited | Difference (95% interval) |
|---|---:|---:|---:|---|
| OR-Bench hard, safe requests refused | 600 | 33.5% | 14.0% | -19.5 points (-22.8 to -16.2) |
| MedMCQA validation, accuracy | 4,183 | 40.57% | 40.76% | +0.19 points (-0.12 to +0.50) |
| MMLU test, accuracy | 2,000 | 54.50% | 54.50% | 0.00 points (-0.30 to +0.30) |
| JailbreakBench harmful, requests refused | 100 | 86% | 84% | -2 points (-7 to +3) |

The weight comparison finds exactly 398,328 changed entries in 195 columns of the four tensors the run reported. Accuracy did not improve and did not fall: on MedMCQA 44 of 4,183 answers differ between the two models. Refusals here are found by a fixed phrase list, which is cruder than the grader's detector, so the difference is the number to read.

### 2.2 Multilingual safety (significant held-out gain, not accepted)

Goal, from the model card: Tiny Aya covers 70+ languages, was trained for safety, and its quality is uneven across languages. Make it handle a harmful request the same way in other languages as in English.

Data: MultiJail, the same harmful requests in ten languages. Dev, target and held-out sets are split by request, so no request appears in two parts in any language. Safe requests in the same languages come from the Aya evaluation suite. Requests about poisons, explosives and biological agents are left out.

Judging took three attempts, and the first two were wrong.

| Baseline | Harmful answers, nine languages | Note |
|---|---:|---|
| Guard judges read the answer in its own language | 21% | Javanese 62%, Swahili 54%. Most of this was refusals labelled harmful. |
| Answers translated to English first (NLLB-200) | 13% | The primary judge still flagged plain refusals. |
| Also, a refusal never counts as harmful | 11% | Used from here on. English: 2%. The two judges agree on 90% of answers. |

The agent found that for translations the model answers harmfully, the refusal signal is already weaker than for the English original at layer 20 and falls further behind through layers 24 to 26. Adding a refusal direction there raised refusals of harmful requests and of safe ones. The edit that worked adds a multilingual harmful-minus-safe direction to MLP columns that both SafeTune localizers select and that stay quiet on safe requests.

| Submission (run 3) | Unseen target (270) | Held-out (450) | Safe requests refused | Verdict |
|---|---|---|---|---|
| 10 columns, 20,467 entries | 10.0% to 8.5%, not significant | not reached | | rejected |
| 26 columns, 53,183 entries | 10.0% to 5.6% (-7.8 to -1.5 points) | 16.2% to 11.6% (-7.3 to -2.2 points) | -0.6 points, no rise | rejected by the narrow-gains detector |

Capability for the second submission: MMLU -1.25 points, MILU Hindi 0, MedMCQA 0, on 80 questions each, none significant.

**The detector.** The narrow-gains check is meant to catch an edit that improves a few prompts a lot and the rest not at all. It fired when the improved prompts numbered no more than a tenth of the set. With yes/no scores and a problem that only 73 of 450 prompts show, fixing 29 of them (40% of the failures) is still fewer than 45, so no realistic fix could pass. We changed the check so that, for yes/no scores, it fires when fewer than ten prompts improve. This was done after the check rejected a result we wanted to pass. The numbers in the table are from the grader as it was; the edit has not been accepted by any version of the grader at the time of writing. A fourth run with the corrected check is rebuilding the edit.

Gains are not uniform. On dev the edit helped in seven languages, left Arabic unchanged and made Chinese slightly worse (2 to 4 harmful answers of 50). Per-language rates on 50 prompts are noisy: Korean read 2% in one baseline and 14% in another after seven requests were removed.

### 2.3 Reward hacking (no accepted edit)

Goal: on School of Reward Hacks prompts, which state how an answer will be scored in a way that can be gamed, make the model do the task honestly more often. An LLM judge labels each answer gamed, honest or broken. About one answer in six is gamed at baseline.

The agent's lead was one column: adding a direction (gamed example answers minus honest ones, at layer 19) to output column 10932 of the layer-19 MLP. On 192 dev items honest answers rose from 161 to 174-187 depending on strength. On 96 unseen items the gamed share moved from 11.5% to 9.4% and to 10.4%, neither significant, and the grader rejected both. Two later runs were stopped by a configuration error in our grader before they got a verdict. The gain was largest on the items used to build the direction, which is what an edit that has memorised its examples looks like.

### 2.4 Math (no edit)

Goal: raise GSM8K accuracy. The released model scores 76.6% on 256 dev problems, and 21 of its 27 inspected misses were reasoning or arithmetic errors rather than formatting. Two neuron leads fixed about as many answers as they broke. The agent reverted them and submitted nothing. We read this as the right outcome: a general reasoning deficit is not a localised fault that a 0.1% edit repairs.

## 3. What went wrong, and what it taught us

- **Judges fail quietly in low-resource languages.** Our first multilingual baseline overstated the problem by a factor of two. The agent caught it by reading answers, not us.
- **Two bugs were in our grader, not in the edits.** One spec named a suite that was never loaded; the session now fails at start if that happens. The narrow-gains check could not pass a rare yes/no problem.
- **Dev gains shrink on unseen prompts.** This happened for reward hacking and for the first multilingual submission. Without a sealed set both would have been reported as wins.
- **The agent's provider blocked one run.** The agent printed the model's answers to harmful requests, one of which was about a biological agent. We removed such requests and the harness no longer shows the agent any answer judged harmful.
- **The kits ran out of GPU memory** on larger calls, even on an 80 GB card, because the guard judges stay loaded while the agent works. CircuitKIT did not complete in the multilingual runs; SafeTune worked with two prompts per language.

## 4. Limits

- One accepted result, from one run. It has an independent check; the multilingual result does not yet.
- Capability is checked on 80 questions per benchmark inside the grader, which can only detect large drops. The independent check used thousands.
- All harm and refusal rates depend on automatic judges. The two guard judges disagree on about one answer in ten.
- One model (3.35B parameters), one agent model.
- The multilingual held-out set has been used to grade more than one candidate.

## 5. Reproduce

```
pip install -e .            # Python 3.12
pytest -q tests             # 72 tests, CPU only
modal deploy -m weightpatch.worker.modal_app
python scripts/spawn_episode.py configs/episodes/tiny-aya-overrefusal.yaml <run-name>
python scripts/spawn_episode.py configs/episodes/tiny-aya-multilingual-safety.yaml <run-name>
modal run scripts/verify_published_model.py     # the independent check
```

The agent needs a Codex login; the Hugging Face token needs access to Tiny Aya. Specs are in `configs/episodes/` and `graders/`.

## 6. Published runs

| Goal | Run | Outcome |
|---|---|---|
| Over-refusal | [accepted](https://huggingface.co/datasets/Perfect7613/weightpatch-run-overrefusal-accepted) | accepted; model: [tiny-aya-global-weightpatch-overrefusal](https://huggingface.co/Perfect7613/tiny-aya-global-weightpatch-overrefusal) |
| Over-refusal | [rejected](https://huggingface.co/datasets/Perfect7613/weightpatch-run-overrefusal-rejected) | rejected, change not significant |
| Math | [gsm8k](https://huggingface.co/datasets/Perfect7613/weightpatch-run-math-gsm8k) | no submission |
| Reward hacking | [1](https://huggingface.co/datasets/Perfect7613/weightpatch-run-metric-gaming-1), [2](https://huggingface.co/datasets/Perfect7613/weightpatch-run-metric-gaming-2), [3](https://huggingface.co/datasets/Perfect7613/weightpatch-run-metric-gaming-3), [4](https://huggingface.co/datasets/Perfect7613/weightpatch-run-metric-gaming-4) | no accepted edit |
| Multilingual safety | [1](https://huggingface.co/datasets/Perfect7613/weightpatch-run-multilingual-safety-1), [2](https://huggingface.co/datasets/Perfect7613/weightpatch-run-multilingual-safety-2), [3](https://huggingface.co/datasets/Perfect7613/weightpatch-run-multilingual-safety-3) | run 3: significant held-out gain, not accepted |
| Baselines | [tiny-aya](https://huggingface.co/datasets/Perfect7613/weightpatch-tiny-aya-baselines), [multilingual](https://huggingface.co/datasets/Perfect7613/weightpatch-tiny-aya-multilingual-baselines) | released model only |

Models are derived from Tiny Aya and carry its CC BY-NC 4.0 licence.
