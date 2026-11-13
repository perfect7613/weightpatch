"""Provider adapters: three ways to reach a worker, one interface.

    handle = connect("modal", spec)        # or "local", or "http"
    handle.call("run", code="print(tools.inspect.names('*down_proj*')[:3])")
    handle.close()

``local``  the worker runs in this process. Tests, the CPU toy, a workstation
           with its own GPU.
``modal``  a Modal container with one L40S. Deploy once with
           ``modal deploy -m weightpatch.worker.modal_app``; the container
           keeps the session's weights in memory between calls.
``http``   any machine running ``weightpatch worker serve``: RunPod, Lambda,
           a university cluster, a box under a desk. This is the "bring your
           own GPU" path, and the one the web app uses for user-connected
           providers.

Adding a provider is one class with ``open(spec) -> WorkerHandle``.
"""

from __future__ import annotations

import json
import urllib.request
import uuid
from typing import Any, Mapping, Protocol


class WorkerHandle(Protocol):
    def call(self, method: str, **payload: Any) -> dict: ...
    def close(self) -> None: ...


# ------------------------------------------------------------------ local
class _LocalHandle:
    def __init__(self, session):
        self.session = session

    def call(self, method: str, **payload: Any) -> dict:
        return self.session.handle(method, payload)

    def close(self) -> None:
        self.session.close()


class LocalProvider:
    name = "local"

    def open(self, spec: Mapping[str, Any]) -> WorkerHandle:
        from .session import WorkerSession

        return _LocalHandle(WorkerSession(spec))


# ------------------------------------------------------------------ modal
class _ModalHandle:
    def __init__(self, worker, session_id: str):
        self.worker, self.session_id = worker, session_id

    def call(self, method: str, **payload: Any) -> dict:
        return self.worker.call.remote(self.session_id, method, payload)

    def close(self) -> None:
        self.worker.call.remote(self.session_id, "close", {})


class ModalProvider:
    """A deployed ``weightpatch-worker`` Modal app (one L40S per container)."""

    name = "modal"

    def __init__(self, app: str = "weightpatch-worker", cls: str = "Worker", environment: str | None = None):
        self.app, self.cls, self.environment = app, cls, environment

    def open(self, spec: Mapping[str, Any]) -> WorkerHandle:
        import modal

        worker = modal.Cls.from_name(self.app, self.cls, environment_name=self.environment)()
        session_id = uuid.uuid4().hex[:12]
        worker.open.remote(session_id, dict(spec))
        return _ModalHandle(worker, session_id)


# ------------------------------------------------------------------- http
class _HttpHandle:
    def __init__(self, url: str, token: str, session_id: str, timeout: float):
        self.url, self.token, self.session_id, self.timeout = url.rstrip("/"), token, session_id, timeout

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url + "/call", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        if "error" in out:
            raise RuntimeError(f"worker error: {out['error']}")
        return out["result"]

    def call(self, method: str, **payload: Any) -> dict:
        return self._post({"session": self.session_id, "method": method, "payload": payload})

    def close(self) -> None:
        self._post({"session": self.session_id, "method": "close", "payload": {}})


class HttpProvider:
    """A worker started with ``weightpatch worker serve --token ...``."""

    name = "http"

    def __init__(self, url: str, token: str, timeout: float = 3600.0):
        self.url, self.token, self.timeout = url, token, timeout

    def open(self, spec: Mapping[str, Any]) -> WorkerHandle:
        handle = _HttpHandle(self.url, self.token, uuid.uuid4().hex[:12], self.timeout)
        handle._post({"session": handle.session_id, "method": "open", "payload": {"spec": dict(spec)}})
        return handle


PROVIDERS = {"local": LocalProvider, "modal": ModalProvider, "http": HttpProvider}


def connect(provider: str, spec: Mapping[str, Any], **options: Any) -> WorkerHandle:
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[provider](**options).open(spec)


__all__ = ["WorkerHandle", "LocalProvider", "ModalProvider", "HttpProvider", "PROVIDERS", "connect"]
