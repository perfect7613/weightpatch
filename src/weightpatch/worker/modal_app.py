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
        "accelerate", "hf_transfer", "sentencepiece", "protobuf",
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
    finally:
        handle.close()
        _keep_login(client)
        runs.commit()
    return json.loads(json.dumps(report, default=str))


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
