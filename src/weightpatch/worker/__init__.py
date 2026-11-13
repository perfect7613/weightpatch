"""Worker: everything that has to sit next to the weights.

The harness has two halves. The control side talks to the model provider
(Codex CLI on the user's ChatGPT subscription) and holds no weights. The
worker side holds the checkpoints, the kernel, the tool box and the grader,
and runs on whatever GPU the user connects.

Between them is one small protocol (:class:`WorkerSession.handle`): a method
name and a JSON payload in, a JSON dict out.

    open      build the session from an episode spec
    describe  tool reference, atlas summary, run facts, budget
    run       execute one code block in the kernel
    snapshot  tool calls, notebook, edits, findings, budget
    export    write the patch and manifest
    close     free the GPU

A provider adapter is a way to reach a worker: in-process (``local``), a
Modal L40S container (``modal``), or any machine running
``weightpatch worker serve`` (``http``: RunPod, Lambda, a lab server).
"""

from .providers import PROVIDERS, HttpProvider, LocalProvider, ModalProvider, WorkerHandle, connect
from .remote import RemoteKernel, RemoteToolBox, remote_episode_parts
from .session import WorkerSession

__all__ = ["WorkerSession", "WorkerHandle", "LocalProvider", "ModalProvider", "HttpProvider", "PROVIDERS",
           "connect", "RemoteKernel", "RemoteToolBox", "remote_episode_parts"]
