"""``weightpatch worker serve``: a worker any GPU machine can run.

One POST endpoint, bearer-token auth, JSON in and out. Sessions are kept in
memory by id. It is deliberately the standard library only: the machine needs
Python, the kits and this package, nothing else. Put it behind TLS (a tunnel
or a reverse proxy) when the control side is not on the same network.
"""

from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .session import WorkerSession


class Sessions:
    """Session table shared by every transport (HTTP here, Modal's class too)."""

    def __init__(self):
        self.table: dict[str, WorkerSession] = {}
        self.lock = threading.Lock()  # one GPU: one call at a time

    def call(self, session_id: str, method: str, payload: dict) -> dict:
        with self.lock:
            if method == "open":
                self.table[session_id] = WorkerSession(payload["spec"])
                return {"opened": session_id, "model": self.table[session_id].model_name}
            if session_id not in self.table:
                raise KeyError(f"no session {session_id!r} on this worker")
            out = self.table[session_id].handle(method, payload)
            if method == "close":
                del self.table[session_id]
            return out


def make_server(host: str, port: int, token: str, sessions: Sessions | None = None) -> ThreadingHTTPServer:
    if not token:
        raise ValueError("refusing to serve without a token")
    sessions = sessions or Sessions()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            given = self.headers.get("Authorization", "")
            if not hmac.compare_digest(given, f"Bearer {token}"):
                return self._send(401, {"error": "unauthorized"})
            if self.path != "/call":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                result = sessions.call(body["session"], body["method"], body.get("payload") or {})
                self._send(200, {"result": result})
            except Exception as e:  # the caller needs the reason, not a dropped connection
                self._send(200, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):  # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "0.0.0.0", port: int = 8765, token: str = "") -> None:
    server = make_server(host, port, token)
    print(f"weightpatch worker on http://{host}:{port} (POST /call)", flush=True)
    server.serve_forever()
