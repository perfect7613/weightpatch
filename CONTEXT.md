# WeightPatch: domain language

Use these words exactly, in code, docs and prompts.

| Term | Meaning |
|---|---|
| **Checkpoint** | A model's full set of tensors, identified by its content hash. |
| **Base / aligned / drifted** | The three reference checkpoints of a repair: pretrained, safety-aligned, and the fine-tune whose behaviour drifted. The drifted checkpoint is the one under repair. |
| **Edit** | One replaced tensor in an edit session, with the evidence that justifies it. |
| **Edit session** | The drifted checkpoint as an immutable baseline plus the agent's edits. The only way weights change. |
| **Edit size** | Entries that differ from the baseline after casting to the deployment dtype (bfloat16). The grader recomputes it. |
| **Direct edit** | An edit to neuron columns or head slices written by the agent, under the 0.1% cap. Contrast with a **repair method** (ReSta, C-ΔΘ, LoX ...) that may touch whole modules. |
| **Patch** | The exported result of an edit session: changed tensors and a manifest. |
| **Weight kernel** | The persistent Python session the agent acts through. Holds `W`, `tools`, `note`. Never holds the model. |
| **Tool box** | The agent's tools in five groups: inspect, localize, repair, edit, measure. An experiment arm is a tool box with groups removed. |
| **Rung** | A level of feedback the agent is given: weights only, kernel, forward passes, SafeTune localizers, CircuitKIT circuits, neuron-level localization. |
| **Forward pass** | One tool call that runs the model on a batch. The scarce resource. |
| **Budget** | The ledger of code runs, forward passes, dev prompts, GPU seconds and submissions for one episode. |
| **Turn** | One message to the model and its reply, holding at most one Python block. |
| **Episode** | One repair attempt: turns until a final answer, the turn limit, or the budget. |
| **Handoff** | Restarting the conversation mid-episode with the task, the notebook and the current edits. The kernel is unchanged. |
| **Tool-free** | A turn whose event stream shows the model used none of its own built-in tools. An episode with a turn that is not tool-free is void. |
| **Model atlas** | Per-checkpoint context built by code: card, name map, quirks, findings. Agents read it first. |
| **Finding** | One logged localization result, keyed by checkpoint hash. |
| **Recipe** | A supported goal (safety-drift repair, over-refusal repair ...) with its grader spec and data. |
| **Grader spec** | The versioned YAML that defines how a candidate is judged. Frozen and hashed at episode start. |
| **Candidate** | A checkpoint submitted for grading. |
| **Verdict** | The grader's decision: accepted, honest failure, or integrity failure. |
| **Honest failure** | A candidate that missed a target or lost capability. Its stage and numbers go back to the agent. |
| **Integrity failure** | A candidate caught by an integrity check, hack detector or cheat review. The agent sees a coarse reason only. |
| **Quarantine log** | Where integrity-failure details go. Humans read it; agents never do. |
| **Dev set** | Prompts the agent may see. **Held-out pool** and **dynamic set** are sealed inside the grader. |
| **Drift tier** | D1 abliteration, D2 GRAFT-planted (neuron-level answer key), D3 contaminated fine-tune. |
| **Method edit** | An edit staged by a repair method through `EditSession.adopt`. Uncapped, revertible, and it moves the candidate out of the direct-edit arm. |
| **Circuit handle** | The name (`circuit:3`) a localizer returns for a circuit held on the worker. Repair tools take handles, so SafeTune and CircuitKIT results feed each other. |
| **Worker** | The half of the harness that sits next to the weights: checkpoints, kernel, tool box, grader. Reached through the worker protocol. |
| **Provider** | A way to reach a worker: local, Modal, or HTTP to any GPU machine. |
| **Suite** | A fixed, ordered list of items of one kind (harm, refusal, capability). Every model sees the same items, so comparisons are paired. |
| **Judge** | An AuditKIT guard model that labels an answer harmful or not. Two judges per harm suite. |
| **RETAIN / FORGET** | GRAFT's two adapters. Flagged examples train FORGET; FORGET is dropped at deployment, or left on to plant D2 drift. |
| **Oracle repair** | The repair the answer key implies. It bounds what any repair can reach on that tier. |
| **Answer key** | The known cause of a planted drift, used to score localization and repair exactly. |
