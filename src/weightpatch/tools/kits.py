"""Lexsi-kit tools: SafeTune and CircuitKIT, registered into the same tool box.

This is where the agent's feedback channels come from. A model editing weights
blind has one signal (does the output change). With these tools it has several
that agree or disagree with each other, each pointing at specific weights:

- SafeTune localizers: the refusal direction per layer, safety neurons by
  weight alignment or by activation contrast, and a native EAP / EAP-IG
  safety circuit down to single attention heads.
- CircuitKIT discovery: a circuit for a registered safety task on a
  TransformerLens copy of the candidate (Cohere2 is supported through
  CircuitKIT's own compatibility port), scored for faithfulness.
- SafeTune recover methods: ReSta, C-DeltaTheta, LoX, NLSR, MSCP and safety
  vector restore, applied to the candidate with the session's edits in place.

Two rules keep them inside the harness's guarantees:

1. A localizer returns a *circuit handle* (``"circuit:3"``) plus plain numbers.
   The handle is what ``repair.ctheta`` takes, so CircuitKIT and SafeTune
   results feed the repair without the agent re-typing unit lists, and the
   agent can also build its own handle with ``localize.make_circuit``.
2. A repair method runs on the borrowed live model and its result is staged
   through :meth:`EditSession.adopt`. Protected tensors stay untouched, the
   evidence rule holds, the edit is revertible, and the candidate moves to the
   uncapped arm so it is never passed off as a tiny direct edit.

Every kit import is lazy. A missing kit removes its tools from the box; it
never breaks the run.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import ToolBox
from .native import ToolContext

_UNIT_RE = re.compile(r"L(\d+)\.(.+)\.(\d+)$")


def available_kits() -> dict[str, bool]:
    """Which Lexsi kits are importable here."""
    return {k: importlib.util.find_spec(k) is not None for k in ("safetune", "circuitkit", "auditkit")}


@dataclass
class KitContext:
    """What the kit tools need beyond the native :class:`ToolContext`."""

    ctx: ToolContext
    base: Mapping[str, torch.Tensor] | None = None  # the checkpoint the aligned model was tuned from
    workdir: Path | None = None  # scratch space on the worker for CircuitKIT artifacts
    refusal_token: str = " I"  # first token of a refusal / of a compliant answer,
    compliance_token: str = " Sure"  # for CircuitKIT's logit-difference metric
    circuits: dict[str, Any] = field(default_factory=dict)
    _modules: dict[str, Any] = field(default_factory=dict)
    _tl: dict[str, Any] = field(default_factory=dict)

    # ---- handles -------------------------------------------------------
    def keep(self, circuit: Any, origin: str) -> str:
        handle = f"circuit:{len(self.circuits) + 1}"
        self.circuits[handle] = {"info": circuit, "origin": origin}
        return handle

    def circuit(self, handle: str | None):
        if handle is None:
            return None
        if handle not in self.circuits:
            raise KeyError(f"no circuit {handle!r}; known: {sorted(self.circuits)}")
        return self.circuits[handle]["info"]

    # ---- models --------------------------------------------------------
    @property
    def runner(self):
        runner = self.ctx.need_runner()
        if not hasattr(runner, "borrow"):
            raise RuntimeError("kit tools need a runner that can lend its model (HFRunner)")
        return runner

    def module(self, which: str):
        """The aligned or base checkpoint as a module (built once, kept)."""
        if which not in self._modules:
            state = self.ctx.need_reference() if which == "aligned" else self.base
            if state is None:
                raise RuntimeError(f"this episode has no {which} checkpoint")
            m = copy.deepcopy(self.runner.model)
            params = dict(m.named_parameters())
            with torch.no_grad():
                for name, t in state.items():
                    if name in params:
                        params[name].copy_(t.to(device=params[name].device, dtype=params[name].dtype))
            self._modules[which] = m.eval()
        return self._modules[which]

    def free_hooked(self) -> None:
        """Drop the TransformerLens copies CircuitKIT ran on."""
        if self._tl:
            self._tl.clear()
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self) -> None:
        """Drop the reference modules and TransformerLens copies."""
        self._modules.clear()
        self._tl.clear()

    def prompts(self, n: int) -> tuple[list[str], list[str]]:
        harmful, benign = list(self.ctx.dev.harmful[:n]), list(self.ctx.dev.benign[:n])
        k = min(len(harmful), len(benign))
        if k == 0:
            raise RuntimeError("this episode has no dev prompts")
        return harmful[:k], benign[:k]


def units_of(info: Any) -> dict[int, list[int]]:
    """A CircuitInfo's neuron units as {layer: [unit, ...]}. Head and block
    nodes (which have no neuron index) are not included."""
    out: dict[int, list[int]] = {}
    for uid in getattr(info.safety_units, "unit_ids", []) or []:
        m = _UNIT_RE.match(str(uid))
        if m:
            out.setdefault(int(m.group(1)), []).append(int(m.group(3)))
    return {k: sorted(v) for k, v in sorted(out.items())}


_ACT_MODULES = ("mlp.act_fn", "mlp.act")


def _as_down_proj(info: Any) -> None:
    """SafeTune's activation method may read neurons at the MLP's activation
    (``mlp.act_fn``). Neuron i there is column i of ``mlp.down_proj``, which is
    the weight a repair can act on, so name the units by that module."""
    def fix(x):
        if isinstance(x, str):
            for a in _ACT_MODULES:
                x = x.replace(a, "mlp.down_proj") if (x == a or f".{a}." in x) else x
            return x
        if isinstance(x, Mapping):
            return {fix(k): v for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(fix(v) for v in x)
        return x

    su, ls = info.safety_units, info.layer_suggestions
    for obj, attrs in ((su, ("module_names", "unit_ids", "activation_correlation")), (ls, ("target_modules",))):
        for attr in attrs:
            if getattr(obj, attr, None) is not None:
                setattr(obj, attr, fix(getattr(obj, attr)))


def _summary(info: Any) -> dict:
    su, ls = info.safety_units, info.layer_suggestions
    ids = list(getattr(su, "unit_ids", []) or [])
    return {"layers": sorted(set(su.layer_indices or [])), "modules": list(su.module_names or []),
            "n_units": len(ids), "first_units": ids[:12],
            "suggested_modules": list(getattr(ls, "target_modules", []) or [])}


def register_kit_tools(box: ToolBox, kit: KitContext) -> dict[str, bool]:
    """Register every tool whose kit is installed. Returns what was found."""
    have = available_kits()
    if have["safetune"]:
        _register_safetune(box, kit)
    if have["circuitkit"] and have["safetune"]:
        _register_circuitkit(box, kit)
    return have


# ======================================================================
# SafeTune
# ======================================================================
def _register_safetune(box: ToolBox, kit: KitContext) -> None:
    ctx, s = kit.ctx, kit.ctx.session

    def _on(which: str):
        """The model a localizer reads: the candidate (with edits) or a reference."""
        if which == "candidate":
            return kit.runner.borrow()
        from contextlib import nullcontext

        return nullcontext(kit.module(which))

    # ---------------- localize ----------------
    def safety_direction(n: int = 32, pick_layer: int | None = None, on: str = "candidate") -> dict:
        """SafeTune refusal direction (difference of means, with SafeTune's layer
        selection sweep). `on` is "candidate", "aligned" or "base".
        Returns {layer, direction [d], per_layer {layer: [d]}}. Compare the
        candidate's direction with the aligned model's: where they disagree is
        where the drift sits."""
        from safetune.interventions.steer.refusal_direction import (
            RefusalDirectionConfig, extract_refusal_direction)

        harmful, benign = kit.prompts(n)
        cfg = RefusalDirectionConfig(pick_layer=pick_layer, n_val=min(16, max(1, len(harmful) // 2)))
        with _on(on) as model:
            direction, layer, per_layer = extract_refusal_direction(
                model, kit.runner.tokenizer, harmful_prompts=harmful, harmless_prompts=benign, config=cfg)
        out = {"layer": int(layer), "direction": direction.detach().float().cpu(),
               "per_layer": {int(k): v.detach().float().cpu() for k, v in per_layer.items()}}
        ctx.findings.append({"tool": "localize.safety_direction", "on": on, "layer": int(layer), "n": len(harmful)})
        return out

    def safety_neurons(method: str = "activation", top_k: int = 16, layers: Sequence[int] | None = None,
                       module: str = "mlp.down_proj", on: str = "aligned", n: int = 32) -> dict:
        """SafeTune safety neurons, the fine-grained rung. method="weight" ranks
        MLP neurons by how strongly their output weights write the refusal
        direction; method="activation" ranks them by how differently they fire on
        harmful and benign prompts. `on` is "aligned" (where the safety lives),
        "candidate" or "base". Returns {circuit, units {layer: [neuron]}, ...};
        pass `circuit` to repair.ctheta, or use `units` as column indices of
        model.layers.<layer>.mlp.down_proj.weight for a direct edit."""
        from safetune.instrumentation.interpret.safety_neurons import safety_circuit_info

        harmful, benign = kit.prompts(n)
        with _on(on) as model:
            info = safety_circuit_info(
                model, kit.runner.tokenizer, harmful, benign, top_k_per_layer=top_k, target_module=module,
                target_layers=list(layers) if layers is not None else None, method=method)
        _as_down_proj(info)
        handle = kit.keep(info, f"safetune.safety_neurons[{method},{on}]")
        out = {"circuit": handle, "units": units_of(info), **_summary(info)}
        ctx.findings.append({"tool": "localize.safety_neurons", "method": method, "on": on,
                             "circuit": handle, "units": out["units"]})
        return out

    def eap_circuit(method: str = "eap-ig", top_k: int = 64, n: int = 16, granularity: str = "head",
                    ig_steps: int = 5, on: str = "candidate") -> dict:
        """SafeTune native EAP / EAP-IG safety circuit: attribution of the
        comply-versus-refuse logit difference to every attention head and MLP
        block. Attribution uses gradients inside this tool; nothing is trained.
        Returns {circuit, scores {node: score} for the top nodes, ...}. Node
        names are blocks.<L>.attn.head<H> and blocks.<L>.mlp."""
        from safetune.instrumentation.interpret.eap import EAPSafetyCircuitConfig, eap_safety_circuit

        harmful, benign = kit.prompts(n)
        cfg = EAPSafetyCircuitConfig(method=method, top_k_edges=top_k, granularity=granularity, ig_steps=ig_steps,
                                     batch_size=min(8, len(harmful)))
        with _on(on) as model, torch.enable_grad():
            info = eap_safety_circuit(model, harmful, benign, cfg, tokenizer=kit.runner.tokenizer)
        handle = kit.keep(info, f"safetune.eap[{method},{on}]")
        su = info.safety_units
        raw = su.activation_correlation or {}
        scores = ({str(k): float(v) for k, v in raw.items()} if isinstance(raw, Mapping)
                  else dict(zip(list(su.unit_ids or []), [float(x) for x in raw])))
        top = dict(sorted(scores.items(), key=lambda kv: -abs(kv[1]))[:24])
        ctx.findings.append({"tool": "localize.eap_circuit", "method": method, "on": on, "circuit": handle, "top": top})
        return {"circuit": handle, "scores": top, **_summary(info)}

    def make_circuit(units: Mapping[int, Sequence[int]], module: str = "mlp.down_proj") -> dict:
        """Build a circuit handle from your own localization: {layer: [neuron,
        ...]} for one module. Lets repair.ctheta act on exactly the neurons you
        found, instead of the whole model."""
        from safetune.core.circuit_kit.adapter import CircuitInfo, LayerModuleSuggestions, SafetyRelevantUnits

        layers = sorted(int(k) for k in units)
        ids = [f"L{layer}.{module}.{int(u)}" for layer in layers for u in units[layer]]
        info = CircuitInfo(
            safety_units=SafetyRelevantUnits(layer_indices=layers, module_names=[module], unit_ids=ids,
                                             activation_correlation=[1.0] * len(ids)),
            layer_suggestions=LayerModuleSuggestions(target_modules=[module], layer_subset=layers,
                                                     priority=[1.0] * len(layers)),
            metadata={"source": "agent"})
        handle = kit.keep(info, "agent")
        return {"circuit": handle, "units": units_of(info), **_summary(info)}

    def circuit_units(circuit: str) -> dict:
        """What a circuit handle holds: origin, layers, modules and neuron units."""
        info = kit.circuit(circuit)
        return {"origin": kit.circuits[circuit]["origin"], "units": units_of(info), **_summary(info)}

    def circuit_overlap(a: str, b: str) -> dict:
        """Agreement between two circuits found by different methods (Jaccard
        overlap of their units, and the shared units per layer). Units that two
        independent methods both name are the ones worth editing."""
        ua, ub = units_of(kit.circuit(a)), units_of(kit.circuit(b))
        shared = {layer: sorted(set(ua.get(layer, [])) & set(ub.get(layer, [])))
                  for layer in sorted(set(ua) | set(ub))}
        na = sum(len(v) for v in ua.values())
        nb = sum(len(v) for v in ub.values())
        ns = sum(len(v) for v in shared.values())
        return {"jaccard": ns / max(na + nb - ns, 1), "shared": {k: v for k, v in shared.items() if v},
                "only_a": na - ns, "only_b": nb - ns}

    fp = {"forward_passes": 2}
    box.register("localize", safety_direction, cost={"forward_passes": 3}, timed=True)
    box.register("localize", safety_neurons, cost=fp, timed=True)
    box.register("localize", eap_circuit, cost={"forward_passes": 4}, timed=True)
    box.register("localize", make_circuit)
    box.register("localize", circuit_units)
    box.register("localize", circuit_overlap)

    # ---------------- repair ----------------
    def _apply(method: str, build, evidence: str, **params) -> dict:
        if not evidence.strip():
            raise ValueError("cite your evidence: which localization result says this method fits")
        runner = kit.runner
        kit.free_hooked()  # a TransformerLens copy and a repair do not fit on the GPU together
        with runner.borrow() as model:
            trainer = build(model)
            out = trainer.apply()
            if isinstance(out, torch.nn.Module) and out is not model:
                # the method returned a new module: read the result from it
                new = {n: p.detach().float().cpu() for n, p in out.named_parameters()}
            else:
                new = runner.changed_since_borrow()
        report = s.adopt(method, new, evidence)
        report["params"] = params
        ctx.findings.append({"tool": f"repair.{method}", **{k: v for k, v in report.items()}})
        return report

    def resta(evidence: str, alpha: float = 0.25, dare: bool = True, dare_drop_rate: float | None = None) -> dict:
        """SafeTune ReSta: add back the safety vector (aligned minus base), scaled
        by alpha, with DARE sparsification of the fine-tuning delta. Use when the
        drift is spread across the model. Tiny Aya wants alpha near 0.25."""
        from safetune.runner.recover import ReStaTrainer

        return _apply("resta", lambda m: ReStaTrainer(
            m, base_model=kit.module("base"), aligned_model=kit.module("aligned"), alpha=alpha, dare=dare,
            dare_drop_rate=dare_drop_rate), evidence, alpha=alpha, dare=dare, dare_drop_rate=dare_drop_rate)

    def ctheta(evidence: str, strength: float = 1.0, circuit: str | None = None) -> dict:
        """SafeTune C-DeltaTheta: shift the candidate along (aligned minus base),
        restricted to a circuit. Pass a `circuit` handle from localize.* so only
        the localized neurons, heads and layers move; without one it moves
        everything. Use when localization is confident."""
        from safetune.runner.recover import CThetaTrainer

        info = kit.circuit(circuit)

        class _Bound:  # CThetaTrainer.apply takes the circuit at apply time
            def __init__(self, m):
                self.t = CThetaTrainer(m, base_model=kit.module("base"), aligned_model=kit.module("aligned"),
                                       strength=strength)

            def apply(self):
                return self.t.apply(circuit_info=info, strength=strength)

        return _apply("ctheta", _Bound, evidence, strength=strength, circuit=circuit)

    def lox(evidence: str, rank: int = 8, extrapolation_factor: float = 0.3) -> dict:
        """SafeTune LoX: extrapolate the candidate along the top-`rank` safety
        subspace of (aligned minus base). Use when the drift is low rank
        (inspect.svd_of_diff shows a few dominant singular values)."""
        from safetune.runner.recover import LoXTrainer

        return _apply("lox", lambda m: LoXTrainer(
            m, base_model=kit.module("base"), aligned_model=kit.module("aligned"), rank=rank,
            extrapolation_factor=extrapolation_factor), evidence, rank=rank,
            extrapolation_factor=extrapolation_factor)

    def safety_vector_restore(evidence: str, alpha: float = 1.0, rank: int = 8) -> dict:
        """SafeTune safety vector restore: project the candidate's departure from
        the aligned model onto its top-`rank` subspace and undo `alpha` of it.
        Needs only the aligned checkpoint."""
        from safetune.runner.recover import SafetyVectorRestoreTrainer

        return _apply("safety_vector_restore", lambda m: SafetyVectorRestoreTrainer(
            m, aligned_model=kit.module("aligned"), alpha=alpha, rank=rank), evidence, alpha=alpha, rank=rank)

    def nlsr(evidence: str, blend: float = 0.5) -> dict:
        """SafeTune NLSR: transplant safety-critical neurons from the aligned
        model into the candidate, blended by `blend`. Needs only the aligned
        checkpoint. Use when a small set of neurons carries the drift."""
        from safetune.runner.recover import NLSRTrainer

        return _apply("nlsr", lambda m: NLSRTrainer(m, donor_state=dict(ctx.need_reference()), blend=blend),
                      evidence, blend=blend)

    def mscp(evidence: str, coefficient: float = 0.05) -> dict:
        """SafeTune MSCP: fine-grained safety-neuron projection using the aligned
        and base checkpoints, scaled by `coefficient`."""
        from safetune.runner.recover import MSCPTrainer

        if kit.base is None:
            raise RuntimeError("this episode has no base checkpoint")
        return _apply("mscp", lambda m: MSCPTrainer(
            m, aligned_state=dict(ctx.need_reference()), base_state=dict(kit.base), coefficient=coefficient),
            evidence, coefficient=coefficient)

    def methods() -> list[dict]:
        """The repair methods applied so far, in order, with how much each moved."""
        rows: dict[str, dict] = {}
        for e in s.list_edits():
            if e.method:
                r = rows.setdefault(e.method, {"method": e.method, "tensors": 0, "entries": 0})
                r["tensors"] += 1
                r["entries"] += e.entries_changed
        return list(rows.values())

    for fn in (resta, ctheta, lox, safety_vector_restore, nlsr, mscp):
        box.register("repair", fn, timed=True)
    box.register("repair", methods)


# ======================================================================
# CircuitKIT
# ======================================================================
SAFETY_TASKS = ("harmbench", "advbench", "jailbreakbench", "sorrybench")
_CK_HEAD, _CK_MLP = re.compile(r"a(\d+)\.h(\d+)$"), re.compile(r"m(?:lp)?\s*(\d+)$")


def _ck_names(scores: Mapping[str, float]) -> dict[str, float]:
    """CircuitKIT score keys in the spelling SafeTune reads (A<layer>.<head>,
    'MLP <layer>'). Neuron-level runs report scores per head and MLP too."""
    out: dict[str, float] = {}
    for name, score in scores.items():
        if m := _CK_HEAD.match(name):
            name = f"A{m.group(1)}.{m.group(2)}"
        elif m := _CK_MLP.match(name):
            name = f"MLP {m.group(1)}"
        out[name] = float(score)
    return out


def _register_circuitkit(box: ToolBox, kit: KitContext) -> None:
    ctx = kit.ctx

    def _workdir() -> Path:
        if kit.workdir is None:
            raise RuntimeError("this episode has no worker scratch directory for CircuitKIT")
        kit.workdir = Path(kit.workdir).resolve()
        kit.workdir.mkdir(parents=True, exist_ok=True)
        return kit.workdir

    def _hooked(which: str):
        """A TransformerLens copy of the candidate or a reference. CircuitKIT's
        Cohere2 port is applied when circuitkit is imported, so import it first."""
        import circuitkit as ck

        runner = kit.runner
        key = which if which != "candidate" else f"candidate:{hash(tuple(sorted(id(t) for t in ctx.session.changed_tensors().values())))}"
        if key in kit._tl:
            return kit._tl[key]
        path = _workdir() / f"ckpt-{which}"
        if which == "candidate":
            with runner.borrow() as model:
                model.save_pretrained(path)
        else:
            kit.module(which).save_pretrained(path)
        runner.tokenizer.save_pretrained(path)
        dtype = "float32" if next(runner.model.parameters()).device.type == "cpu" else "bfloat16"
        kit._tl = {k: v for k, v in kit._tl.items() if not k.startswith("candidate:")}  # one candidate copy at a time
        kit._tl[key] = ck.load_model(str(path), dtype=dtype)
        return kit._tl[key]

    def _register_task(task: str, n: int) -> str:
        """Make `task` known to CircuitKIT. "dev" is built from this episode's
        own contrast prompts, so the circuit is about the same behaviour the
        other localizers measured (and no gated dataset is needed)."""
        from circuitkit.tasks import safety_datasets as sd

        if task in SAFETY_TASKS:
            getattr(sd, f"register_{task}")(max_records=max(n, 64))
            return task
        if task != "dev":
            return task  # any task already registered with CircuitKIT
        from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
        from circuitkit.data.normalized_task import NormalizedTaskSpec
        from circuitkit.tasks.registry import is_task_registered, register_task

        name = f"weightpatch_dev_{id(kit) & 0xFFFFFF:x}"
        if not is_task_registered(name):
            harmful, benign = kit.prompts(max(n, 8))
            ds = SafetyPromptAdapter().adapt(
                [{"prompt": p} for p in harmful], name=name, source="episode dev set", benign_controls=benign,
                pairing_mode="harmful_vs_benign", refusal_token=kit.refusal_token,
                compliance_token=kit.compliance_token)
            register_task(NormalizedTaskSpec(ds, name=name))
        return name

    def discover_circuit(task: str = "dev", algorithm: str = "eap-ig", level: str = "node",
                         n_examples: int = 64, sparsity: float = 0.3, scope: str = "both",
                         on: str = "candidate") -> dict:
        """CircuitKIT circuit discovery: which attention heads and MLPs, or with
        level="neuron" which neurons, carry the refuse-versus-comply behaviour.
        task="dev" uses this episode's harmful and benign dev prompts; harmbench,
        advbench, jailbreakbench and sorrybench use those public sets.
        `on` is "candidate", "aligned" or "base". Returns {circuit, top_nodes
        {node: score}, n_nodes}. Node names: A<layer>.<head>, MLP <layer>,
        L<layer>.<neuron>. Run it on the aligned model and on the candidate and
        compare with localize.circuit_overlap."""
        import circuitkit as ck
        from safetune.core.circuit_kit.adapter import circuit_info_from_node_scores

        # CircuitKIT caches normalised task data under the working directory, at
        # registration and again at discovery: do both from the worker's scratch
        with contextlib.chdir(_workdir()), torch.enable_grad():
            task = _register_task(task, n_examples)
            out_path = _workdir() / f"{task}-{algorithm}-{level}-{on}-{len(kit.circuits) + 1}.pt"
            circuit = ck.discover(_hooked(on), task, algorithm=algorithm, level=level, n_examples=n_examples,
                                  sparsity=sparsity, scope=scope, output_path=str(out_path))
        if not circuit.scores:
            raise RuntimeError("CircuitKIT returned a circuit with no node scores")
        info = circuit_info_from_node_scores(_ck_names(circuit.scores), top_fraction=max(1.0 - sparsity, 0.05))
        handle = kit.keep(info, f"circuitkit.{algorithm}[{task},{level},{on}]")
        kit.circuits[handle]["ck"] = circuit
        kit.circuits[handle]["task"] = task
        top = {k: float(v) for k, v in circuit.top_nodes(24).items()}
        ctx.findings.append({"tool": "localize.discover_circuit", "task": task, "algorithm": algorithm,
                             "level": level, "on": on, "circuit": handle, "top": top})
        return {"circuit": handle, "top_nodes": top, "n_nodes": len(circuit), "units": units_of(info)}

    def circuit_faithfulness(circuit: str, n_examples: int = 64, on: str = "candidate") -> dict:
        """CircuitKIT faithfulness of a discovered circuit: patching_score and
        ablation_score in [0, 1] (1 = the circuit alone reproduces the full
        model's behaviour on the task). A low score means the circuit is not the
        mechanism; do not edit on its evidence."""
        import circuitkit as ck

        entry = kit.circuits.get(circuit)
        if entry is None or "ck" not in entry:
            raise KeyError(f"{circuit!r} was not produced by localize.discover_circuit")
        with contextlib.chdir(_workdir()):
            report = ck.faithfulness(_hooked(on), entry["ck"], entry["task"], n_examples=n_examples)
        out = {"patching_score": float(report.patching_score), "ablation_score": float(report.ablation_score)}
        ctx.findings.append({"tool": "localize.circuit_faithfulness", "circuit": circuit, **out})
        return out

    box.register("localize", discover_circuit,
                 cost=lambda n_examples=64, **_: {"forward_passes": 2 + n_examples // 32}, timed=True)
    box.register("localize", circuit_faithfulness,
                 cost=lambda n_examples=64, **_: {"forward_passes": 2 + n_examples // 32}, timed=True)


__all__ = ["KitContext", "register_kit_tools", "available_kits", "units_of", "SAFETY_TASKS"]
