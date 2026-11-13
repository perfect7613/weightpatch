"""GRAFT: gradient routing into two MLP adapters, on one GPU.

GRAFT (Ward et al., arXiv 2610.17062, code at github.com/jnward/GRAFT, MIT
licence) trains two small SwiGLU branches beside every MLP: RETAIN and FORGET.
A monitor labels each training example and the label decides where its
gradient goes:

    route       FORGET forward   RETAIN grad   FORGET grad
    anchor            0               1             0      known clean
    unflagged         1               1             1      everything else
    flagged           1               0           kappa    monitor says unsafe

So whatever the unsafe examples teach lands in FORGET, which is dropped at
deployment. The split-moment optimizer makes the kappa-times step real under
Adam (Adam would otherwise normalise it away).

This module is a compact single-process port of the SFT recipe: the adapter
and the optimizer follow GRAFT's ``sft/scripts/gr/adapter.py`` and
``split_moment.py``. It adds one thing the harness needs, :func:`merge`:

A SwiGLU branch beside a SwiGLU MLP is exactly extra neurons in that MLP. So
the trained adapters fold into an ordinary, slightly wider checkpoint whose
last ``d_forget`` neurons per layer *are* the FORGET branch. That gives

- a deployable prevention recipe (merge with FORGET dropped), and
- drift tier D2: merge with FORGET on, and the drifted model has a
  neuron-level answer key. The oracle repair is "zero these columns".

Training here uses gradients, as any fine-tuning does. It builds models for
the agent to repair; it never runs inside the agent's kernel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

ANCHOR, UNFLAGGED, FLAGGED = "anchor", "unflagged", "flagged"


@dataclass
class GraftConfig:
    d_retain: int = 32
    d_forget: int = 32
    lora_alpha: float = 32.0  # adapters are scaled to match an rsLoRA of this alpha
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4
    epochs: int = 5
    window: int = 8  # examples per optimizer step (each gets its own backward)
    kappa: float = 2.0  # FORGET's step multiplier on flagged examples
    anchor_fraction: float = 0.01  # share of unflagged examples promoted to anchors
    max_len: int = 256
    grad_clip: float | None = 1.0
    seed: int = 0

    def routes(self) -> dict[str, tuple[float, float, float]]:
        return {ANCHOR: (0.0, 1.0, 0.0), UNFLAGGED: (1.0, 1.0, 1.0), FLAGGED: (1.0, 0.0, self.kappa)}


@dataclass
class GraftExample:
    prompt: str
    response: str = ""  # loss is taken on the response when there is one
    route: str = UNFLAGGED


# ---------------------------------------------------------------- adapter
class DualBranch(nn.Module):
    """A frozen MLP plus RETAIN and FORGET SwiGLU branches.

    out = base(x) + retain_scale * s_r * retain(x) + forget_scale * s_f * forget(x)

    ``s_r`` and ``s_f`` are fixed variance-matching constants (the output
    variance of an rsLoRA with the same alpha); the two ``*_scale`` values are
    the runtime switches that routing and deployment flip.
    """

    def __init__(self, base_mlp: nn.Module, hidden: int, d_retain: int, d_forget: int, lora_alpha: float = 32.0):
        super().__init__()
        self.base_mlp = base_mlp
        ref = next(base_mlp.parameters())
        mk = lambda i, o: nn.Linear(i, o, bias=False, device=ref.device, dtype=ref.dtype)  # noqa: E731
        self.gate_retain, self.up_retain, self.down_retain = mk(hidden, d_retain), mk(hidden, d_retain), mk(d_retain, hidden)
        self.gate_forget, self.up_forget, self.down_forget = mk(hidden, d_forget), mk(hidden, d_forget), mk(d_forget, hidden)
        nn.init.zeros_(self.down_retain.weight)  # a fresh adapter is a no-op
        nn.init.zeros_(self.down_forget.weight)
        self.s_retain = lora_alpha * math.sqrt(1.0 / (0.5 * d_retain))
        self.s_forget = lora_alpha * math.sqrt(1.0 / (0.5 * d_forget))
        self.retain_scale, self.forget_scale = 1.0, 1.0
        self.act = getattr(base_mlp, "act_fn", nn.functional.silu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_mlp(x)
        if self.retain_scale:
            out = out + (self.retain_scale * self.s_retain) * self.down_retain(self.act(self.gate_retain(x)) * self.up_retain(x))
        if self.forget_scale:
            out = out + (self.forget_scale * self.s_forget) * self.down_forget(self.act(self.gate_forget(x)) * self.up_forget(x))
        return out


def _layers(model):
    body = getattr(model, model.base_model_prefix, model)
    return body.layers


def inject(model, cfg: GraftConfig) -> list[DualBranch]:
    """Wrap every layer's MLP and freeze everything but the adapters. Every
    layer is wrapped, because :func:`merge` widens every MLP by the same amount."""
    hidden = int(model.config.hidden_size)
    for p in model.parameters():
        p.requires_grad_(False)
    out = []
    for layer in _layers(model):
        if isinstance(layer.mlp, DualBranch):
            raise RuntimeError("adapters are already injected")
        layer.mlp = DualBranch(layer.mlp, hidden, cfg.d_retain, cfg.d_forget, cfg.lora_alpha)
        out.append(layer.mlp)
    for a in out:
        for name, p in a.named_parameters():
            p.requires_grad_(not name.startswith("base_mlp."))
    return out


def adapters(model) -> list[DualBranch]:
    return [m for m in model.modules() if isinstance(m, DualBranch)]


def set_scales(model, retain: float = 1.0, forget: float = 1.0) -> None:
    for a in adapters(model):
        a.retain_scale, a.forget_scale = retain, forget


def _role_params(model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    retain, forget = [], []
    for a in adapters(model):
        retain += [a.gate_retain.weight, a.up_retain.weight, a.down_retain.weight]
        forget += [a.gate_forget.weight, a.up_forget.weight, a.down_forget.weight]
    return retain, forget


# -------------------------------------------------------------- optimizer
class SplitMomentAdamW(torch.optim.AdamW):
    """AdamW whose first moment follows the routed gradient (``p.grad``) and
    whose second moment follows the natural, pre-routing gradient
    (``p._natural_grad``). Feeding a kappa-times gradient into both moments
    would cancel; splitting them makes the realised step scale with kappa.

    Param groups carry ``role`` ("retain" or "forget"). :meth:`set_window`
    gives, for the next step, each role's participation (scales the natural
    gradient) and whether it had any examples (an idle role is frozen: no
    moment update, no step count, no weight decay)."""

    def set_window(self, participation: dict[str, float], active: dict[str, bool]) -> None:
        self._window = (participation, active)

    @torch.no_grad()
    def step(self, closure=None):
        participation, active = getattr(self, "_window", ({}, {}))
        for group in self.param_groups:
            role = group.get("role")
            if role is not None and not active.get(role, True):
                continue
            c = float(participation.get(role, 1.0))
            (b1, b2), lr, eps, wd = group["betas"], group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                natural = getattr(p, "_natural_grad", None)
                if natural is None:
                    if role is not None:
                        raise RuntimeError("a routed parameter has no natural gradient: the capture hook did not fire")
                    natural = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)  # fp32 moments even for bf16 weights
                    st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                if wd:
                    p.mul_(1.0 - lr * wd)
                gm, gv = p.grad.float(), natural.float() * c
                st["m"].mul_(b1).add_(gm, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(gv, gv, value=1 - b2)
                denom = (st["v"].sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
                p.add_((st["m"] / denom).to(p.dtype), alpha=-lr / (1 - b1 ** t))
        self._window = ({}, {})
        return None


def _route_hooks(params: Sequence[nn.Parameter], scale: float) -> list:
    """Record the natural gradient, hand on the routed one."""
    def mk(p):
        def hook(g):
            prior = getattr(p, "_natural_grad", None)
            if prior is None:
                p._natural_grad = g.detach().clone()
            else:
                prior.add_(g.detach())
            return g if scale == 1.0 else g * scale
        return hook
    return [p.register_hook(mk(p)) for p in params]


# ---------------------------------------------------------------- training
def assign_anchors(examples: Sequence[GraftExample], fraction: float, seed: int = 0) -> list[GraftExample]:
    """Promote a share of the unflagged examples to anchors. Anchors are the
    examples known to be clean; they train RETAIN with FORGET switched off, so
    RETAIN learns to do the task without leaning on FORGET."""
    rng = random.Random(seed)
    idx = [i for i, e in enumerate(examples) if e.route == UNFLAGGED]
    chosen = set(rng.sample(idx, max(1, round(len(idx) * fraction)))) if idx and fraction > 0 else set()
    return [GraftExample(e.prompt, e.response, ANCHOR) if i in chosen else e for i, e in enumerate(examples)]


def _encode(tokenizer, ex: GraftExample, max_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    if ex.response and getattr(tokenizer, "chat_template", None):
        head = tokenizer.apply_chat_template([{"role": "user", "content": ex.prompt}], tokenize=False,
                                             add_generation_prompt=True)
        p_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(ex.response, add_special_tokens=False)["input_ids"]
    else:
        p_ids = tokenizer(ex.prompt)["input_ids"]
        r_ids = tokenizer(" " + ex.response, add_special_tokens=False)["input_ids"] if ex.response else []
    if tokenizer.eos_token_id is not None and r_ids:
        r_ids = r_ids + [tokenizer.eos_token_id]
    ids = (p_ids + r_ids)[-max_len:]
    labels = ([-100] * len(p_ids) + r_ids)[-max_len:] if r_ids else list(ids)
    return torch.tensor([ids], device=device), torch.tensor([labels], device=device)


def train(model, tokenizer, examples: Sequence[GraftExample], cfg: GraftConfig | None = None,
          on_step: Callable[[dict], None] | None = None) -> dict:
    """Train RETAIN and FORGET with three-way routing. Injects adapters if the
    model has none. Returns per-route mean losses of the last epoch."""
    cfg = cfg or GraftConfig()
    if not adapters(model):
        inject(model, cfg)
    retain, forget = _role_params(model)
    opt = SplitMomentAdamW(
        [{"params": retain, "role": "retain"}, {"params": forget, "role": "forget"}],
        lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, foreach=False, fused=False)
    routes, device = cfg.routes(), next(model.parameters()).device
    data = assign_anchors(list(examples), cfg.anchor_fraction, cfg.seed)
    rng, step, last = random.Random(cfg.seed), 0, {}
    model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        losses: dict[str, list[float]] = {ANCHOR: [], UNFLAGGED: [], FLAGGED: []}
        for start in range(0, len(order), cfg.window):
            window = [data[i] for i in order[start:start + cfg.window]]
            for p in retain + forget:
                p._natural_grad = None
            n_forward_on = 0
            for ex in window:
                f_forward, r_grad, f_grad = routes[ex.route]
                set_scales(model, retain=1.0, forget=f_forward)
                n_forward_on += f_forward > 0
                hooks = _route_hooks(retain, r_grad) + _route_hooks(forget, f_grad)
                try:
                    ids, labels = _encode(tokenizer, ex, cfg.max_len, device)
                    loss = model(input_ids=ids, labels=labels).loss
                    (loss / len(window)).backward()
                    losses[ex.route].append(float(loss.detach()))
                finally:
                    for h in hooks:
                        h.remove()
                    set_scales(model, 1.0, 1.0)
            for p in retain + forget:  # anchors never reach FORGET: give it exact zeros
                if p._natural_grad is None:
                    p._natural_grad = torch.zeros_like(p)
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            if cfg.grad_clip:
                total = torch.nn.utils.clip_grad_norm_(retain + forget, cfg.grad_clip)
                coef = cfg.grad_clip / (float(total) + 1e-6)
                if coef < 1.0:  # one shared clipping event for both moment sources
                    for p in retain + forget:
                        p._natural_grad.mul_(coef)
            # FORGET sees only the forward-on examples: rescale so it steps at RETAIN's per-example rate
            opt.set_window({"retain": 1.0, "forget": len(window) / n_forward_on if n_forward_on else 1.0},
                           {"retain": True, "forget": n_forward_on > 0})
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if on_step:
                on_step({"epoch": epoch, "step": step, **{k: (sum(v) / len(v) if v else None) for k, v in losses.items()}})
        last = {k: (sum(v) / len(v) if v else None) for k, v in losses.items()}
    for p in retain + forget:
        p._natural_grad = None
    model.eval()
    return {"steps": step, "loss": last, "routes": {r: sum(e.route == r for e in data) for r in routes}}


# ------------------------------------------------------------------ merge
def merge(model, forget_scale: float = 1.0, retain_scale: float = 1.0, drop_forget: bool = False):
    """Fold the adapters into a plain checkpoint with a wider MLP.

    Returns ``(state, config, key)``. Per layer the MLP gains ``d_retain``
    neurons, then ``d_forget`` neurons (unless ``drop_forget``); ``key`` names
    them. With ``forget_scale=0`` the FORGET neurons are present but write
    nothing, which gives a same-shape "aligned" twin of the FORGET-on model.
    """
    ads = adapters(model)
    if not ads:
        raise RuntimeError("the model has no GRAFT adapters to merge")
    state = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
             if "_retain." not in n and "_forget." not in n}
    state = {n.replace(".mlp.base_mlp.", ".mlp."): t for n, t in state.items()}
    key = {"retain_neurons": {}, "forget_neurons": {}, "d_retain": ads[0].down_retain.in_features,
           "d_forget": 0 if drop_forget else ads[0].down_forget.in_features}
    prefix = next(n for n in state if ".mlp.down_proj.weight" in n).split("layers.")[0]
    for i, a in enumerate(ads):
        base = f"{prefix}layers.{i}.mlp."
        width = state[base + "down_proj.weight"].shape[1]
        f = lambda lin: lin.weight.detach().float().cpu()  # noqa: E731
        gates, ups = [state[base + "gate_proj.weight"], f(a.gate_retain)], [state[base + "up_proj.weight"], f(a.up_retain)]
        downs = [state[base + "down_proj.weight"], retain_scale * a.s_retain * f(a.down_retain)]
        key["retain_neurons"][i] = list(range(width, width + key["d_retain"]))
        if not drop_forget:
            gates.append(f(a.gate_forget))
            ups.append(f(a.up_forget))
            downs.append(forget_scale * a.s_forget * f(a.down_forget))
            start = width + key["d_retain"]
            key["forget_neurons"][i] = list(range(start, start + key["d_forget"]))
        state[base + "gate_proj.weight"] = torch.cat(gates, dim=0)
        state[base + "up_proj.weight"] = torch.cat(ups, dim=0)
        state[base + "down_proj.weight"] = torch.cat(downs, dim=1)
    config = model.config.to_dict()
    config["intermediate_size"] = int(config["intermediate_size"]) + key["d_retain"] + key["d_forget"]
    key["tensor"] = prefix + "layers.{layer}.mlp.down_proj.weight"
    return state, config, key


def build_merged(config: dict, state: dict, like=None):
    """A plain model from :func:`merge`'s output."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(config["model_type"], **{k: v for k, v in config.items() if k != "model_type"})
    model = AutoModelForCausalLM.from_config(cfg)
    missing = model.load_state_dict(state, strict=False)
    bad = [k for k in missing.unexpected_keys] + [k for k in missing.missing_keys if "lm_head" not in k]
    if bad:
        raise RuntimeError(f"merged state does not fit the widened config: {bad[:5]}")
    if like is not None:
        ref = next(like.parameters())
        model = model.to(device=ref.device, dtype=ref.dtype)
    return model.eval()


def calibrate_forget_scale(harm_at: Callable[[float], float], target: float,
                           scales: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)) -> dict:
    """Pick the FORGET scale whose measured harm rate is closest to ``target``.
    ``harm_at(scale)`` evaluates the merged model at that scale. Used to plant
    D2 drift of a chosen severity (e.g. matching the D3 fine-tune)."""
    rows = [{"scale": float(sc), "harm": float(harm_at(float(sc)))} for sc in scales]
    best = min(rows, key=lambda r: abs(r["harm"] - target))
    return {"scale": best["scale"], "harm": best["harm"], "target": target, "sweep": rows}


__all__ = ["GraftConfig", "GraftExample", "DualBranch", "SplitMomentAdamW", "inject", "adapters", "set_scales",
           "assign_anchors", "train", "merge", "build_merged", "calibrate_forget_scale",
           "ANCHOR", "UNFLAGGED", "FLAGGED"]
