"""The Modal worker: one L40S, the Lexsi kits pinned, weights kept warm.

    modal secret create huggingface HF_TOKEN=hf_...      # once; gated models need it
    modal deploy -m weightpatch.worker.modal_app          # once per code change

Then ``connect("modal", spec)`` from the control side. A container holds its
sessions in memory, so ``max_containers=1`` keeps every call of an episode on
the container that has its weights. Tiny Aya is 3.35B parameters: the live
model, the borrow snapshot and two reference modules fit in the L40S's 48 GB
in bfloat16.
"""

from __future__ import annotations

import modal

APP_NAME = "weightpatch-worker"
GPU = "L40S"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.10", "transformers>=5.15,<6", "safetensors", "numpy", "pyyaml", "datasets", "peft",
        "accelerate", "hf_transfer",
        # Lexsi's stack, pinned to the versions this harness was verified against
        "safetune>=0.1.12,<0.2", "circuitkit==0.2.3", "auditkit==1.1.11",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("weightpatch")
)

app = modal.App(APP_NAME, image=image)
cache = modal.Volume.from_name("weightpatch-cache", create_if_missing=True)  # model downloads
runs = modal.Volume.from_name("weightpatch-runs", create_if_missing=True)    # atlas, patches, sealed pool


@app.cls(gpu=GPU, memory=65536, timeout=4 * 3600, scaledown_window=20 * 60, max_containers=1,
         volumes={"/cache": cache, "/runs": runs}, secrets=[modal.Secret.from_name("huggingface")])
class Worker:
    @modal.enter()
    def start(self):
        from weightpatch.worker.server import Sessions

        self.sessions = Sessions()

    @modal.method()
    def open(self, session_id: str, spec: dict) -> dict:
        spec = {**spec, "out": f"/runs/{session_id}"}
        return self.sessions.call(session_id, "open", {"spec": spec})

    @modal.method()
    def call(self, session_id: str, method: str, payload: dict) -> dict:
        out = self.sessions.call(session_id, method, payload)
        if method in ("export", "snapshot", "close"):
            runs.commit()
        return out
