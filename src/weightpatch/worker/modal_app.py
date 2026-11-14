"""WeightPatch on Modal: the whole harness on one GPU.

Everything runs in the container: the Lexsi kits, the weights, the grader,
and the Codex CLI that drives GPT-6 Astra on the user's ChatGPT subscription.

    modal secret create huggingface HF_TOKEN=hf_...          # once
    modal deploy -m weightpatch.worker.modal_app             # once per code change
    modal run -m weightpatch.worker.modal_app::codex_login   # once: sign in with ChatGPT (device code)
    modal run -m weightpatch.worker.modal_app::smoke         # the kits on Tiny Aya
    modal run -m weightpatch.worker.modal_app::build_drift --tier d1
    modal run -m weightpatch.worker.modal_app::episode --spec configs/episodes/tiny-aya-d1.yaml

The ChatGPT login is made inside Modal with Codex's device-code flow and
lives on a private volume; no credential file is copied from a laptop.

One GPU per container (``WEIGHTPATCH_GPU``: L40S by default, or H100). Tiny
Aya is 3.35B parameters: the live model, a borrow snapshot and two reference
modules fit in 48 GB in bfloat16.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "weightpatch"
GPU = os.environ.get("WEIGHTPATCH_GPU", "L40S")
if GPU not in ("L40S", "H100"):
    raise ValueError("WEIGHTPATCH_GPU must be L40S or H100")
CODEX_VERSION = "0.167.2"
ROOT = Path(__file__).resolve().parents[3]  # the repo, when run from a checkout

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        f"npm install -g @openai/codex@{CODEX_VERSION}",
    )
    .pip_install(
        "torch>=2.10", "transformers>=5.15,<6", "safetensors", "numpy", "pyyaml", "datasets", "peft",
        "accelerate", "hf_transfer", "sentencepiece", "protobuf", "huggingface_hub",
        # Lexsi's stack, pinned to the versions this harness was verified against
        "safetune==0.1.12", "circuitkit==0.2.3", "auditkit==1.1.11",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false",
          "WEIGHTPATCH_ROOT": "/app", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("weightpatch")
)
for _d in ("skills", "graders", "configs"):
    if (ROOT / _d).is_dir():
        image = image.add_local_dir(str(ROOT / _d), f"/app/{_d}")

app = modal.App(APP_NAME, image=image)
cache = modal.Volume.from_name("weightpatch-cache", create_if_missing=True)  # model downloads
runs = modal.Volume.from_name("weightpatch-runs", create_if_missing=True)    # drift, atlas, traces, sealed pool
codex = modal.Volume.from_name("weightpatch-codex", create_if_missing=True)  # the ChatGPT login
hf = modal.Secret.from_name("huggingface")

LOGIN_HOME = "/codex/login"
GPU_KW = dict(gpu=GPU, memory=65536, timeout=6 * 3600, secrets=[hf],
              volumes={"/cache": cache, "/runs": runs, "/codex": codex})


# ------------------------------------------------------------ ChatGPT login
@app.function(volumes={"/codex": codex}, timeout=20 * 60)
def codex_login() -> str:
    """Sign in with ChatGPT inside Modal (device code). Prints a URL and a
    one-time code; open the URL, enter the code, and the login is saved."""
    Path(LOGIN_HOME).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CODEX_HOME=LOGIN_HOME)
    proc = subprocess.Popen(["codex", "login", "--device-auth"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    proc.wait()
    codex.commit()
    status = subprocess.run(["codex", "login", "status"], env=env, capture_output=True, text=True)
    return (status.stdout + status.stderr).strip()


@app.function(volumes={"/codex": codex}, timeout=10 * 60)
def codex_check(model: str = "gpt-6-astra") -> dict:
    """Is the login there, and does the model answer with its tools off?"""
    from weightpatch.model_client import CodexExecClient

    if not Path(LOGIN_HOME, "auth.json").exists():
        return {"logged_in": False}
    client = CodexExecClient(model=model, effort="low", auth_file=f"{LOGIN_HOME}/auth.json")
    reply = client.send("Reply with the single word: ready", new_conversation=True)
    _keep_login(client)
    out = {"logged_in": True, "model": model, "reply": reply.text.strip()[:80], "tool_free": reply.tool_free,
           "usage": reply.usage}
    print("CODEX_CHECK", json.dumps(out), flush=True)
    return out


def _keep_login(client) -> None:
    """Codex refreshes its token in place; if it replaced the link with a new
    file, keep the new one so the next run is still signed in."""
    link = Path(client.home) / "auth.json"
    if link.exists() and not link.is_symlink():
        shutil.copy2(link, f"{LOGIN_HOME}/auth.json")
    codex.commit()


# ------------------------------------------------------------------- tests
@app.function(**GPU_KW)
def smoke(aligned: str = "CohereLabs/tiny-aya-global", base: str = "CohereLabs/tiny-aya-base", n: int = 16,
          drifted: str = "") -> dict:
    """The kits on the real model: load Tiny Aya, call each family of tool
    once, report what came back and how long it took. Uses the aligned model
    as its own 'drifted' checkpoint, so this checks plumbing, not repair."""
    import time

    import torch

    from weightpatch.worker.session import WorkerSession

    t0, out = time.time(), {"gpu": torch.cuda.get_device_name(0)}
    w = WorkerSession({"model": {"source": "hf", "drifted": drifted or aligned, "aligned": aligned, "base": base,
                                 "tokenizer": aligned},
                       "out": "/runs/smoke", "budgets": {}, "cap": 0.001,
                       "dev": {"harmful": "harmbench_dev", "benign": "alpaca", "n": max(n, 32), "seed": 0}})
    out["load_seconds"] = round(time.time() - t0, 1)
    out["kits"], out["tools"] = w.kits, len(w.box.names())
    t = w.box.namespace()

    def step(name, fn):
        s = time.time()
        try:
            res = fn()
            out[name] = {"ok": True, "seconds": round(time.time() - s, 1), "result": res}
        except Exception as e:  # report every tool, do not stop at the first failure
            out[name] = {"ok": False, "seconds": round(time.time() - s, 1), "error": f"{type(e).__name__}: {e}"[:600]}
        print(name, json.dumps(out[name], default=str)[:400], flush=True)

    step("dev_eval", lambda: t.measure.dev_eval(n=n))
    step("sample", lambda: t.measure.sample(list(w.ctx.dev.harmful[:2]) + list(w.ctx.dev.benign[:2]), 48))
    step("refusal_direction", lambda: {k: float(v.norm()) for k, v in list(t.localize.refusal_direction(n=n).items())[::6]})
    step("safety_direction", lambda: (lambda d: {"layer": d["layer"], "norm": float(d["direction"].norm())})(
        t.localize.safety_direction(n=n)))
    step("safety_neurons_activation", lambda: (lambda r: {"circuit": r["circuit"], "n_units": r["n_units"],
                                                           "first": r["first_units"][:6]})(
        t.localize.safety_neurons(method="activation", top_k=8, on="aligned", n=n)))
    step("safety_neurons_weight", lambda: (lambda r: {"circuit": r["circuit"], "n_units": r["n_units"]})(
        t.localize.safety_neurons(method="weight", top_k=8, on="candidate", n=n)))
    step("circuit_overlap", lambda: t.localize.circuit_overlap("circuit:1", "circuit:2"))
    step("eap_circuit", lambda: (lambda r: {"circuit": r["circuit"], "top": list(r["scores"].items())[:6]})(
        t.localize.eap_circuit(method="eap", top_k=32, n=8)))
    step("discover_circuit", lambda: (lambda r: {"circuit": r["circuit"], "n_nodes": r["n_nodes"],
                                                 "top": list(r["top_nodes"].items())[:6]})(
        t.localize.discover_circuit(task="dev", algorithm="eap", level="node", n_examples=16)))
    ck_handle = (out.get("discover_circuit", {}).get("result") or {}).get("circuit")
    if ck_handle:
        step("circuit_faithfulness", lambda: t.localize.circuit_faithfulness(ck_handle, n_examples=32))
    for name, call in (("resta", lambda: t.repair.resta(evidence="smoke", alpha=0.25)),
                       ("ctheta", lambda: t.repair.ctheta(evidence="smoke", strength=0.5, circuit="circuit:1")),
                       ("lox", lambda: t.repair.lox(evidence="smoke")),
                       ("nlsr", lambda: t.repair.nlsr(evidence="smoke")),
                       ("mscp", lambda: t.repair.mscp(evidence="smoke")),
                       ("safety_vector_restore", lambda: t.repair.safety_vector_restore(evidence="smoke"))):
        step("repair." + name, lambda call=call: {k: v for k, v in call().items() if k != "skipped_protected"})
        w.session.revert()
    step("dev_eval_aligned_restore", lambda: (t.edit.restore_from_reference(
        "model.layers.14.mlp.down_proj.weight", cols=list(range(64)), evidence="smoke"), t.measure.dev_eval(n=n))[1])
    out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    out["total_seconds"] = round(time.time() - t0, 1)
    Path("/runs/smoke/report.json").write_text(json.dumps(out, indent=2, default=str))
    runs.commit()
    return json.loads(json.dumps(out, default=str))


PROFILE_SUITES = ("harmbench_dev", "xstest_safe", "orbench_hard", "jbb_benign", "mmlu", "milu_hi", "medmcqa")


@app.function(**GPU_KW)
def profile(model: str = "CohereLabs/tiny-aya-global", n: int = 100, name: str = "profile") -> dict:
    """Where does a model, as released, have room to improve? Measures harm
    (plain, held-out and perturbed prompts), refusals on safe prompts, and
    capability, with the same suites and judges the grader uses."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from weightpatch.edits import EditSession
    from weightpatch.evaluator import AuditKitEvaluator, guard_judges
    from weightpatch.grader.detectors import is_looping, wilson
    from weightpatch.runner import HFRunner
    from weightpatch.suites import dynamic_suite, load_jsonl, load_public

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
    session = EditSession({k: v.detach().cpu() for k, v in m.named_parameters()})
    runner = HFRunner(m, tok, session, batch_size=16)
    suites = {}
    for s_name in PROFILE_SUITES:
        try:
            suites[s_name] = load_public(s_name, n)
        except Exception as e:
            print("skip", s_name, str(e)[:200], flush=True)
    suites["held_out"] = load_jsonl("/runs/sealed/heldout_pool.jsonl", "held_out", "harm", n=n)
    suites["perturbed"] = dynamic_suite(suites["held_out"].items, n, round_seed=1)
    ev = AuditKitEvaluator(runner, {}, suites, guard_judges("granite_guardian", "llama_guard",
                                                           {"dtype": "bfloat16", "device_map": "cuda"}))
    out = {"model": model, "n": n, "suites": {}}
    for s_name, st in suites.items():
        run = ev.run("candidate", s_name, min(n, len(st.items)))
        k, total = sum(run.scores), len(run.scores)
        lo, hi = wilson(int(k), total)
        row = {"kind": st.kind, "rate": round(k / total, 3), "ci": [round(lo, 3), round(hi, 3)], "n": total,
               "looping": round(sum(is_looping(t) for t in run.texts) / total, 3),
               "mean_chars": round(sum(len(t) for t in run.texts) / total)}
        if run.second:
            row["second_judge_rate"] = round(sum(run.second) / total, 3)
        flagged = [(st.items[i].prompt, run.texts[i]) for i, sc in enumerate(run.scores)
                   if (sc > 0.5) == (st.kind != "capability")]
        row["examples"] = [{"prompt": p[:200], "answer": a[:200]} for p, a in flagged[:5]]
        out["suites"][s_name] = row
        print("PROFILE", s_name, json.dumps({k2: v for k2, v in row.items() if k2 != "examples"}), flush=True)
    Path(f"/runs/{name}").mkdir(parents=True, exist_ok=True)
    Path(f"/runs/{name}/profile.json").write_text(json.dumps(out, indent=2))
    runs.commit()
    return out


@app.function(**GPU_KW)
def baseline(spec_yaml: str, n: int = 96, name: str = "baseline") -> dict:
    """The released model on an episode's task dev sets, before any agent runs:
    is there a problem to fix, and how large? Uses the episode's own tools."""
    import yaml

    from weightpatch.worker.session import WorkerSession

    spec = yaml.safe_load(spec_yaml)
    spec.pop("agent", None)
    spec.pop("grader", None)
    spec["out"] = f"/runs/{name}"
    w = WorkerSession(spec)
    t, out = w.box.namespace(), {}
    for task in w.ctx.dev.tasks:
        r = t.measure.task_eval(task, n=n)
        rows = r.pop("items")
        out[task] = {**r, "labels": {k: sum(x.get("label") == k for x in rows) for k in ("gamed", "honest", "broken")}
                     if "label" in rows[0] else None,
                     "examples": [{"i": x["i"], "label": x.get("label"), "correct": x["correct"], "head": x["head"]}
                                  for x in rows[:12]]}
        print("BASELINE", task, json.dumps({k: v for k, v in out[task].items() if k != "examples"}, default=str), flush=True)
    Path(f"/runs/{name}/baseline.json").write_text(json.dumps(out, indent=2, default=str))
    runs.commit()
    return json.loads(json.dumps(out, default=str))


@app.local_entrypoint()
def base(spec: str, n: int = 96, name: str = "baseline"):
    """modal run -m weightpatch.worker.modal_app::base --spec <file>"""
    print(json.dumps(baseline.remote(Path(spec).read_text(), n=n, name=name), indent=1, default=str)[:3000])


# ------------------------------------------------------------------- drift
@app.function(**GPU_KW)
def build_drift(tier: str = "d1", aligned: str = "CohereLabs/tiny-aya-global", n: int = 256,
                share: float = 0.12, epochs: int = 2, forget_scale: float = 1.0, task_data: str = "medmcqa") -> dict:
    """Build a drifted checkpoint and its answer key under /runs/drift/<tier>."""
    from weightpatch.cli import cmd_drift

    out = f"/runs/drift/{tier}"
    import argparse

    cmd_drift(argparse.Namespace(tier=tier, model=aligned, out=out, dtype="bfloat16", n=n, share=share,
                                 epochs=epochs, forget_scale=forget_scale, task_data=task_data))
    runs.commit()
    return {"tier": tier, "out": out, "files": sorted(str(p.relative_to(out)) for p in Path(out).rglob("*") if p.is_file())[:20]}


@app.function(volumes={"/runs": runs}, secrets=[hf], timeout=30 * 60)
def build_sealed(n: int = 400, seed: int = 0) -> dict:
    """Write the sealed held-out pool to the runs volume (worker side only)."""
    from weightpatch.suites import build_heldout_pool

    path = build_heldout_pool("/runs/sealed/heldout_pool.jsonl", n=n, seed=seed)
    runs.commit()
    return {"path": str(path), "items": sum(1 for _ in open(path))}


@app.function(**GPU_KW)
def grade_reference(spec_yaml: str, which: str = "oracle") -> dict:
    """Grade a known candidate, to check the grader itself on the real model.
    which="oracle": every drifted tensor restored from the aligned model (the
    best any repair can do). which="resta": SafeTune ReSta at its documented
    setting, with no agent."""
    import yaml

    from weightpatch.worker.session import WorkerSession

    spec = yaml.safe_load(spec_yaml)
    spec.pop("agent", None)
    spec["out"] = f"/runs/grade-{which}"
    w = WorkerSession(spec)
    t = w.box.namespace()
    if which == "oracle":
        w.session.adopt("oracle", {n: v for n, v in w.ctx.reference.items()}, evidence="aligned checkpoint")
    elif which == "resta":
        t.repair.resta(evidence="baseline: no localization", alpha=0.25)
    else:
        raise ValueError(which)
    before = w.session.fraction_changed()
    feedback = t.measure.submit(note=which)
    Path(f"/runs/grade-{which}/result.json").write_text(json.dumps(
        {"which": which, "fraction_changed": before, "feedback": feedback, "verdict": w.verdicts[-1],
         "judges": getattr(w, "judges_label", "")}, indent=2, default=str))
    runs.commit()
    return json.loads(json.dumps({"which": which, "fraction_changed": before, "feedback": feedback,
                                  "verdict": w.verdicts[-1], "judges": getattr(w, "judges_label", "")}, default=str))


# ----------------------------------------------------------------- episode
@app.function(**GPU_KW)
def episode(spec_yaml: str, name: str = "episode", model: str = "", effort: str = "") -> dict:
    """One full repair episode in this container: Astra through the Codex CLI
    on the saved ChatGPT login, the kernel and kits on this GPU."""
    import yaml

    from weightpatch.cli import REPAIR_TASK, load_skills
    from weightpatch.episode import Episode
    from weightpatch.model_client import CodexExecClient
    from weightpatch.worker import connect, remote_episode_parts

    spec = yaml.safe_load(spec_yaml)
    agent = spec.pop("agent", {})
    out = Path(f"/runs/{name}")
    spec["out"] = str(out / "worker")
    client = CodexExecClient(model=model or agent.get("model", "gpt-6-astra").split(":")[-1],
                             effort=effort or agent.get("effort", "high"), auth_file=f"{LOGIN_HOME}/auth.json")
    handle = connect("local", spec)
    try:
        ep = Episode(agent.get("task", REPAIR_TASK), client, skills=load_skills(agent.get("skills", [])),
                     max_turns=int(agent.get("max_turns", 160)), handoff_every=int(agent.get("handoff_every", 40)),
                     min_code_runs=int(agent.get("min_code_runs", 0)),
                     accepted=lambda: any(v.get("kind") == "accepted" for v in handle.call("snapshot")["verdicts"]),
                     trace_dir=out / "trace",
                     on_turn=lambda t: (print(f"turn {t.index}: {'code' if t.code else 'final'} ({t.seconds}s)\n"
                                              f"{(t.output or '')[:600]}", flush=True), runs.commit()),
                     **remote_episode_parts(handle))
        result = ep.run()
        snap, patch = handle.call("snapshot"), handle.call("export")
        report = {"stopped": result.stopped, "turns": len(result.turns), "tool_free": result.tool_free,
                  "final_answer": result.final_answer, "verdicts": snap["verdicts"],
                  "direct_edit": snap["direct_edit"], "fraction_changed": snap["fraction_changed"],
                  "budget": snap["budget"], "edits": snap["edits"], "manifest": patch["manifest"],
                  "usage": [t.usage for t in result.turns]}
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        try:  # the report agent; a failure here must not lose the run
            from weightpatch.report import write_report

            if (out / "worker" / "patch" / "patch.safetensors").exists():
                _layer_record(out, spec["model"]["drifted"])
            report["report"] = write_report(out, client, humanizer=Path("/app/skills/humanizer/SKILL.md").read_text())
        except Exception as e:
            print("report agent failed:", type(e).__name__, str(e)[:300], flush=True)
    finally:
        handle.close()
        _keep_login(client)
        runs.commit()
    return json.loads(json.dumps(report, default=str))


# ------------------------------------------------------ report and publish
CPU_KW = dict(memory=49152, timeout=3 * 3600, secrets=[hf], volumes={"/cache": cache, "/runs": runs, "/codex": codex})


def _layer_record(run: Path, released: str) -> list:
    """edits_by_layer.json: what the stored patch changes, per tensor and column."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from weightpatch.report import layer_record

    patch = load_file(str(run / "worker" / "patch" / "patch.safetensors"))
    base = AutoModelForCausalLM.from_pretrained(released, dtype=torch.bfloat16)
    rows = layer_record({n: p.detach() for n, p in base.named_parameters() if n in patch}, patch)
    (run / "edits_by_layer.json").write_text(json.dumps(rows, indent=1))
    return rows


@app.function(**CPU_KW)
def report(name: str, released: str = "CohereLabs/tiny-aya-global", model: str = "gpt-6-astra",
           effort: str = "high") -> dict:
    """The report agent on a finished run: records the edit per layer, then
    Astra (tools off) writes REPORT.md from the evidence and a second pass
    edits it with the humanizer skill."""
    from weightpatch.model_client import CodexExecClient
    from weightpatch.report import write_report

    run = Path(f"/runs/{name}")
    if (run / "worker" / "patch" / "patch.safetensors").exists():
        rows = _layer_record(run, released)
        print("LAYERS", [(r["layer"], r.get("columns_changed"), r["entries_changed"]) for r in rows], flush=True)
    client = CodexExecClient(model=model, effort=effort, auth_file=f"{LOGIN_HOME}/auth.json")
    try:
        out = write_report(run, client, humanizer=Path("/app/skills/humanizer/SKILL.md").read_text())
    finally:
        _keep_login(client)
        runs.commit()
    print("REPORT", json.dumps({k: v for k, v in out.items() if k != "usage"}), flush=True)
    return out


MODEL_CARD = """\
---
license: cc-by-nc-4.0
base_model: {released}
library_name: transformers
tags: [weightpatch, weight-editing, safetune, circuitkit, auditkit, lexsi]
---
# {title}

`{released}` with {entries:,} weights changed ({fraction:.4%} of the model) in {tensors} tensors.
The edit was made by an agent (GPT-6 Astra, its own tools off) working inside the WeightPatch
harness with Lexsi Labs' SafeTune, CircuitKIT and AuditKIT, and was accepted by a grader that
measures on prompts the agent never saw. No gradient training was used.

Code: https://github.com/{github}. Full trace of the run: https://huggingface.co/datasets/{dataset}.

## Files
- `model.safetensors`, tokenizer and config: the edited model, loadable with `transformers`.
- `weightpatch/patch.safetensors`: only the changed tensors. `weightpatch/manifest.json`: the edits and the
  evidence the agent recorded for each. `weightpatch/edits_by_layer.json`: every changed column.
- `weightpatch/REPORT.md`: the run report, also below.

## Licence
Tiny Aya is released by Cohere Labs under CC BY-NC 4.0; this derivative keeps that licence.

---
{report}
"""


@app.function(**CPU_KW)
def publish(name: str, model_repo: str, dataset_repo: str, title: str, released: str = "CohereLabs/tiny-aya-global",
            github: str = "perfect7613/weightpatch", extra_runs: str = "", collection: str = "WeightPatch",
            private: bool = False) -> dict:
    """Put a finished run on Hugging Face: the edited model (full weights plus
    the patch), and a dataset with the run's trace, verdicts, atlas and report.
    Both go into collections named ``collection``."""
    import torch
    from huggingface_hub import HfApi
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api, run = HfApi(token=os.environ["HF_TOKEN"]), Path(f"/runs/{name}")
    owner = model_repo.split("/")[0]
    manifest = json.loads((run / "worker" / "patch" / "manifest.json").read_text())
    patch = load_file(str(run / "worker" / "patch" / "patch.safetensors"))
    report_md = (run / "REPORT.md").read_text() if (run / "REPORT.md").exists() else ""

    # ---- the model
    model = AutoModelForCausalLM.from_pretrained(released, dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    changed = 0
    with torch.no_grad():
        for n, t in patch.items():
            changed += int((params[n] != t.to(params[n].dtype)).sum())
            params[n].copy_(t.to(params[n].dtype))
    if changed != manifest["entries_changed"]:
        raise RuntimeError(f"patch changes {changed} entries, manifest says {manifest['entries_changed']}")
    out = Path("/tmp/publish-model")
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(released).save_pretrained(out)
    (out / "weightpatch").mkdir()
    for f in (run / "worker" / "patch" / "patch.safetensors", run / "worker" / "patch" / "manifest.json",
              run / "edits_by_layer.json", run / "REPORT.md", run / "worker" / "verdicts.jsonl"):
        if f.exists():
            shutil.copy2(f, out / "weightpatch" / f.name)
    (out / "README.md").write_text(MODEL_CARD.format(
        released=released, title=title, entries=changed, fraction=manifest["fraction_changed"], tensors=len(patch),
        github=github, dataset=dataset_repo, report=report_md))
    api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=model_repo, repo_type="model",
                      commit_message=f"WeightPatch run {name}")

    # ---- the dataset: traces and logs of this run and any others named
    stage = Path("/tmp/publish-data")
    shutil.rmtree(stage, ignore_errors=True)
    names = [name] + [x for x in extra_runs.split(",") if x]
    for r in names:
        src, dst = Path(f"/runs/{r}"), stage / "runs" / r
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("circuitkit", "ckpt-*", "cache", "*.pt"))
        for ev in dst.rglob("events.jsonl"):  # session ids of the model provider are not part of the record
            ev.write_text(re.sub(r'"thread_id": "[^"]*"', '"thread_id": "redacted"', ev.read_text()))
    for extra in ("profile/profile.json",):
        if Path("/runs", extra).exists():
            (stage / Path(extra).parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path("/runs", extra), stage / extra)
    shutil.copytree("/app/configs", stage / "configs")
    shutil.copytree("/app/graders", stage / "graders")
    hits = [str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() and p.suffix != ".safetensors"
            and re.search(r"hf_[A-Za-z0-9]{20}|refresh_token|access_token|@gmail\.com", p.read_text(errors="ignore"))]
    if hits:
        raise RuntimeError(f"refusing to publish: possible credentials or personal details in {hits[:5]}")
    (stage / "README.md").write_text(
        "---\nlicense: mit\ntags: [weightpatch, agent-traces, weight-editing, lexsi]\n---\n"
        f"# WeightPatch runs: traces, logs and results\n\n"
        f"Everything stored for the runs of the WeightPatch harness (https://github.com/{github}).\n\n"
        "Per run, under `runs/<name>/`:\n"
        "- `REPORT.md`: the run report, written by the report agent from `report_evidence.md`.\n"
        "- `report.json`: outcome, grader verdicts, budget and token use.\n"
        "- `edits_by_layer.json`: every changed tensor and column, with sizes.\n"
        "- `trace/turns/tNNN/`: for each turn, the message sent to the agent (`prompt.txt`), its reply with its "
        "explanation (`answer.md`), the code it ran (`code.py`), what the code printed (`output.txt`), the raw "
        "model events (`events.jsonl`) and token use (`run.json`).\n"
        "- `trace/notebook.md`, `trace/tool_calls.jsonl`: the agent's notes and every tool call with its cost.\n"
        "- `worker/verdicts.jsonl`: each grader verdict in full. `worker/patch/`: the changed tensors and manifest.\n"
        "- `worker/atlas/`: the model atlas the agent started from (architecture, wiring, diagram, profile, findings).\n\n"
        "`configs/` and `graders/` are the episode and grader specs. `profile/profile.json` is the measured "
        "baseline of the released model.\n\n"
        f"Runs here: {', '.join(names)}. The edited model: https://huggingface.co/{model_repo}.\n\n"
        "The traces quote prompts from public safety benchmarks (HarmBench, OR-Bench) and short model answers to "
        "them. The grader's sealed prompts are not included.\n")
    api.create_repo(dataset_repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=dataset_repo, repo_type="dataset",
                      commit_message=f"WeightPatch runs: {', '.join(names)}")

    # ---- collections
    links = {}
    for kind, repo, label in (("model", model_repo, f"{collection}: models"), ("dataset", dataset_repo,
                                                                              f"{collection}: datasets and traces")):
        col = api.create_collection(title=label, namespace=owner, private=private, exists_ok=True,
                                    description="WeightPatch, Lexsi Labs AI Safety and Alignment Hackathon")
        api.add_collection_item(col.slug, item_id=repo, item_type=kind, exists_ok=True)
        links[kind + "_collection"] = f"https://huggingface.co/collections/{col.slug}"
    out_info = {"model": f"https://huggingface.co/{model_repo}", "dataset": f"https://huggingface.co/datasets/{dataset_repo}",
                **links, "entries_changed": changed}
    print("PUBLISHED", json.dumps(out_info), flush=True)
    return out_info


RUN_CARD = """\
---
license: mit
tags: [weightpatch, agent-traces, weight-editing, lexsi]
---
# {title}

{summary}

One run of the WeightPatch harness (https://github.com/{github}): an agent (GPT-6 Astra, its own
tools off) editing the weights of `{released}` with Lexsi Labs' SafeTune, CircuitKIT and AuditKIT
as tools, judged by a grader on prompts it never saw. This repo holds that one run and nothing else.

## Files
- `REPORT.md`: the run report, written by the report agent from `report_evidence.md`.
- `report.json`: outcome, grader verdicts, budget and token use.
- `edits_by_layer.json`: every changed tensor and column, with sizes (when the run kept an edit).
- `trace/turns/tNNN/`: per turn, the message sent to the agent (`prompt.txt`), its reply with its
  explanation (`answer.md`), the code it ran (`code.py`), what the code printed (`output.txt`), the raw
  model events (`events.jsonl`) and token use (`run.json`).
- `trace/notebook.md`, `trace/tool_calls.jsonl`: the agent's notes and every tool call with its cost.
- `worker/verdicts.jsonl`: each grader verdict in full. `worker/patch/`: the changed tensors and manifest.
- `worker/atlas/`: the model atlas the agent started from (architecture, wiring, diagram, profile, findings).
- `episode.yaml`, `grader.yaml`: the specs this run used, when present; otherwise see the first message in `trace/turns/t001/prompt.txt` and the GitHub history.

The trace quotes prompts from public benchmarks and short model answers to them. The grader's
sealed prompts are not included.
{links}
"""


def _collect(api, owner: str, title: str, repo: str, kind: str, private: bool) -> str:
    col = api.create_collection(title=title, namespace=owner, private=private, exists_ok=True,
                                description="WeightPatch, Lexsi Labs AI Safety and Alignment Hackathon")
    api.add_collection_item(col.slug, item_id=repo, item_type=kind, exists_ok=True)
    return f"https://huggingface.co/collections/{col.slug}"


def _check_clean(stage: Path) -> None:
    hits = [str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() and p.suffix != ".safetensors"
            and re.search(r"hf_[A-Za-z0-9]{20}|refresh_token|access_token|@gmail\.com", p.read_text(errors="ignore"))]
    if hits:
        raise RuntimeError(f"refusing to publish: possible credentials or personal details in {hits[:5]}")


@app.function(**CPU_KW)
def publish_run(name: str, repo: str, title: str, summary: str, spec: str = "", grader: str = "",
                links: str = "", released: str = "CohereLabs/tiny-aya-global", github: str = "perfect7613/weightpatch",
                collection: str = "WeightPatch: datasets and traces", private: bool = False) -> dict:
    """One run, one dataset repo: its trace, verdicts, atlas, patch and report."""
    from huggingface_hub import HfApi

    api, stage = HfApi(token=os.environ["HF_TOKEN"]), Path("/tmp/publish-run")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(f"/runs/{name}", stage, ignore=shutil.ignore_patterns("circuitkit", "ckpt-*", "cache", "*.pt"))
    for ev in stage.rglob("events.jsonl"):  # session ids of the model provider are not part of the record
        ev.write_text(re.sub(r'"thread_id": "[^"]*"', '"thread_id": "redacted"', ev.read_text()))
    for src, dst in ((spec, "episode.yaml"), (grader, "grader.yaml")):
        if src and Path("/app", src).exists():
            shutil.copy2(Path("/app", src), stage / dst)
    _check_clean(stage)
    (stage / "README.md").write_text(RUN_CARD.format(title=title, summary=summary, github=github, released=released,
                                                     links=("\n" + links) if links else ""))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="dataset", commit_message=f"WeightPatch run {name}")
    out = {"dataset": f"https://huggingface.co/datasets/{repo}",
           "collection": _collect(api, repo.split("/")[0], collection, repo, "dataset", private)}
    print("PUBLISHED", json.dumps(out), flush=True)
    return out


@app.function(**CPU_KW)
def publish_files(repo: str, kind: str, title: str, body: str, files: str, collection: str,
                  license: str = "mit", base_model: str = "", private: bool = False) -> dict:
    """A small repo of its own: named files from the runs volume (``src=dst`` pairs,
    comma separated) with a card. Used for a patch-only model repo and for the
    baseline measurements."""
    from huggingface_hub import HfApi

    api, stage = HfApi(token=os.environ["HF_TOKEN"]), Path("/tmp/publish-files")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for pair in files.split(","):
        src, dst = pair.split("=")
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path("/runs", src), stage / dst)
    _check_clean(stage)
    head = f"---\nlicense: {license}\n" + (f"base_model: {base_model}\n" if base_model else "") + \
           "tags: [weightpatch, weight-editing, lexsi]\n---\n"
    (stage / "README.md").write_text(f"{head}# {title}\n\n{body}\n")
    api.create_repo(repo, repo_type=kind, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type=kind, commit_message=title)
    out = {"repo": repo, "collection": _collect(api, repo.split("/")[0], collection, repo, kind, private)}
    print("PUBLISHED", json.dumps(out), flush=True)
    return out


@app.function(secrets=[hf], timeout=600)
def uncollect(collection_slug: str, repo: str) -> list:
    """Take a repo out of a collection (the repo itself is left alone)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    for item in api.get_collection(collection_slug).items:
        if item.item_id == repo:
            api.delete_collection_item(collection_slug, item.item_object_id)
    return [i.item_id for i in api.get_collection(collection_slug).items]


# ------------------------------------------------------- remote worker (kept)
@app.cls(scaledown_window=20 * 60, max_containers=1, **GPU_KW)
class Worker:
    """The worker protocol as a Modal class, for a control side that runs
    elsewhere (``connect("modal", spec)``)."""

    @modal.enter()
    def start(self):
        from weightpatch.worker.server import Sessions

        self.sessions = Sessions()

    @modal.method()
    def open(self, session_id: str, spec: dict) -> dict:
        return self.sessions.call(session_id, "open", {"spec": {**spec, "out": f"/runs/{session_id}"}})

    @modal.method()
    def call(self, session_id: str, method: str, payload: dict) -> dict:
        out = self.sessions.call(session_id, method, payload)
        if method in ("export", "snapshot", "close"):
            runs.commit()
        return out


@app.local_entrypoint()
def grade(spec: str = "configs/episodes/tiny-aya-d1-quick.yaml", which: str = "oracle"):
    """modal run -m weightpatch.worker.modal_app::grade --which oracle"""
    print(json.dumps(grade_reference.remote(Path(spec).read_text(), which=which), indent=2, default=str))


@app.local_entrypoint()
def main(spec: str = "configs/episodes/tiny-aya-d1.yaml", name: str = "episode", model: str = "", effort: str = ""):
    """modal run -m weightpatch.worker.modal_app --spec <file> --name <run>"""
    report = episode.remote(Path(spec).read_text(), name=name, model=model, effort=effort)
    print(json.dumps({k: v for k, v in report.items() if k not in ("usage", "manifest")}, indent=2, default=str))
