# Rewind SDK

**A transactional sandbox runtime for AI coding agents.** Run agent-generated code in an isolated container, checkpoint filesystem and conversation state together, and roll both back atomically when something breaks.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)

> **Status:** early prototype. The core engine works and is tested; the framework integration surface is currently LangGraph-only. Treat this as a v0 you can build against, not a finished product. See [Limitations](#known-limitations) before you rely on it.

---

## The Problem

Agents that write and execute code need somewhere to do that safely, and a way to recover when they fail. Two specific failures keep coming up:

- **Filesystem damage.** An agent edits files, runs a destructive command, or otherwise leaves a workspace in a half-broken state with no way back except git or a manual backup.
- **Schema corruption after a crash.** When an agent's tool call fails mid-execution, you're often left with an assistant message requesting a tool call that has no matching tool response. Strict providers (Gemini, OpenAI) will reject that message history outright on the next call, even though the failure already happened and you just want to continue.

Rewind addresses both by tying filesystem snapshots and conversation history to the same checkpoint, so rolling back one always rolls back the other.

---

## What It Actually Does

Rewind boots an Alpine Linux Docker container, mounts your host workspace into it **read-only**, and gives the agent a writable OverlayFS layer to work in. Checkpointing is a layer-stacking operation (fast — no file copying), and rollback discards layers back to a chosen point. Separately, it keeps a parallel in-memory record of your conversation messages, snapshotted at the same checkpoint label, so a filesystem rollback and a memory rollback always happen together via one call.

```python
from rewind_sdk import session

with session("agent", workspace="./src", auto_commit=True) as sess:
    sess.checkpoint("stable")

    sess.write_file("auth.py", new_implementation)
    try:
        sess.run_tests("pytest")
    except RuntimeError:
        sess.rollback("stable")
    # If this block exits without raising and auto_commit=True,
    # the workspace is streamed back to ./src on the host.
```

**Filesystem and message history are restored together, with one call.** That's the actual core mechanic — everything else in this SDK is built around making that pairing convenient.

---

## Verified Features

These are implemented and covered by the test suite or directly traceable in source:

- **OverlayFS checkpoints** — instant, layer-based snapshots of the sandbox filesystem (`engine.py`)
- **Paired memory rollback** — message history is truncated to match a filesystem checkpoint in one call
- **Dangling tool-call cleanup** — if the checkpoint point falls right after an assistant message that initiated a tool call, that message is automatically dropped too, so you don't hand a broken message history back to a strict-schema provider
- **Auto-checkpoint before tool calls** — `on_tool_call()` snapshots state automatically when wired into your tool functions
- **Auto-rollback on exception or test failure** — `auto_rollback("exception", "test_failure", ...)` triggers a rollback automatically inside `run()` / `run_tests()` and inside LangGraph's `invoke`/`stream`
- **Two-phase commit to host** — host files are only touched if a session block exits without raising and `auto_commit=True` is set; otherwise nothing is written back
- **LangGraph adapter** — `wrap_langgraph()` wraps a compiled graph's `invoke`/`stream` to keep memory synced and to trigger rollback on unhandled exceptions
- **CLI and MCP server** — `rewind_cli.py` and `mcp_server.py` expose the same session operations as a command-line tool and as MCP tools, respectively

---

## Installation

**Requirements:** Python 3.9+, Docker running locally.

This is not yet published to PyPI. Install from source, editable:

```bash
git clone <your-repo-url>
cd rewind
pip install -e .
```

```python
from rewind_sdk import session  # package name is rewind_sdk, not rewind
```

---

## Quick Start

```python
from rewind_sdk import session

with session("agent", workspace="./workspace") as sess:
    sess.write_file("script.py", "print('hello')")
    output = sess.run("python3 script.py")
    print(output)
# By default destroy_on_exit=True and auto_commit=False:
# the container is torn down on exit and nothing is written back
# to ./workspace. Pass auto_commit=True if you want the result persisted.
```

### Checkpoint and roll back

```python
with session("agent", workspace="./src", auto_commit=True) as sess:
    sess.checkpoint("stable")

    sess.write_file("config.py", risky_change)
    try:
        sess.run_tests()  # raises RuntimeError on non-zero exit, e.g. pytest failure
    except RuntimeError:
        sess.rollback("stable")
```

### Sync conversation memory

```python
with session("agent", workspace="./src") as sess:
    messages = [
        {"role": "user", "content": "Find the bug"},
        {"role": "assistant", "content": "Found it in auth.py"},
    ]
    sess.sync_memory(messages)
    restored = sess.get_messages()
```

---

## Automated State Management

### Auto-checkpoint

```python
sess.auto_checkpoint(trigger="before_tool_call", keep_last=10)
```

`trigger="before_tool_call"` is the only trigger currently implemented. `keep_last` trims the SDK's own convenience label history (`_auto_labels`) — it does **not** delete the underlying OverlayFS checkpoint layers, which remain on disk regardless of this setting. If you're watching container disk usage, this parameter won't help; there's currently no automatic checkpoint-layer pruning.

Auto-checkpoints only fire where you explicitly call `sess.on_tool_call(...)` — typically from inside your own tool functions, or via the LangGraph adapter's `before_tool_node` hook if you wire it into your graph. It is not a global hook that activates on every tool call without integration.

### Auto-rollback

```python
sess.auto_rollback("exception", "test_failure", to="latest", test_command="pytest")
```

Two events are actually implemented: `"exception"` and `"test_failure"`. Both are checked inside `run()`, `run_tests()`, and inside the LangGraph adapter's `invoke`/`stream` exception handling. There is no `"validation_error"` or `"timeout"` event in the current code, despite what you may see suggested elsewhere — if you need either, you'll need to catch it yourself and call `sess.rollback(...)` directly.

```python
if sess.last_auto_rollback:
    print(sess.last_auto_rollback["event"], sess.last_auto_rollback["to"])
```

---

## LangGraph Integration

```python
import threading
from rewind_sdk import session, wrap_langgraph

tool_lock = threading.Lock()
sandbox = session("agent_sandbox", workspace="./my_codebase", auto_commit=True)

@tool
def write_file(path: str, content: str) -> str:
    with tool_lock:
        sandbox.on_tool_call(tool_name="write_file")  # explicit checkpoint trigger
        sandbox.write_file(path, content)
        return f"Wrote to {path}"

with sandbox:
    sandbox.auto_checkpoint(trigger="before_tool_call")
    sandbox.auto_rollback("exception", to="latest")

    safe_agent = wrap_langgraph(base_agent, session=sandbox)
    for event in safe_agent.stream({"messages": messages}):
        pass
```

**What this buys you:** your system prompt doesn't need to mention rollbacks, checkpoints, or recovery — the message-history correction happens in `memory.py`, not in the prompt. **What it doesn't do automatically:** checkpointing before each tool call still requires you to call `sandbox.on_tool_call(...)` inside your tool implementations, as shown above. The adapter keeps memory synced and catches unhandled exceptions from `invoke`/`stream`, but it does not instrument your tools for you.

A thread lock around tool execution is recommended (and used above) because the sandbox is a single container — concurrent writes from parallel tool calls aren't serialized for you.

---

## CLI

```bash
python rewind_cli.py init ./my-project
python rewind_cli.py write src/app.py "print('hi')"
python rewind_cli.py checkpoint stable
python rewind_cli.py exec "pytest"
python rewind_cli.py rollback stable
python rewind_cli.py status
python rewind_cli.py destroy
```

Add `--json` for machine-readable output and `--quiet` to suppress stderr logging — useful if another agent is driving the CLI directly.

## MCP Server

`mcp_server.py` exposes session operations (`init_sandbox`, `execute_sandbox_command`, `write_sandbox_file`, `read_sandbox_file`, `sync_agent_memory`, `create_sandbox_checkpoint`, `rollback_sandbox_state`, `configure_auto_checkpoint`, `configure_auto_rollback`, `get_sandbox_status`) as MCP tools, for clients that want to drive a Rewind sandbox without writing Python. Requires `pip install mcp`.

---

## Known Limitations

Being direct about these now is better than someone finding them the hard way:

- **Containers run `--privileged`.** This is required for the current OverlayFS mounting approach, but it means the sandbox container has broad host-kernel access — it is not a hardened security boundary against a determined adversary. Treat it as protection against an agent's *accidental* mistakes (bad refactors, destructive commands), not as isolation against malicious code.
- **One framework integration.** Only LangGraph is supported today. The adapter pattern (`messages_to_dicts` / `dicts_to_messages`) is framework-agnostic in design, but no LangChain-only or CrewAI adapter exists yet.
- **Not on PyPI.** Install from source only, for now.
- **No automatic concurrency control inside the SDK.** If you call sandbox methods from multiple threads, you need your own lock (see the LangGraph example above) — the SDK does not serialize for you.
- **Auto-checkpoint requires manual wiring.** `on_tool_call()` needs to be called from your own tool code; it isn't injected automatically into arbitrary agent frameworks.
- **Only two auto-rollback events implemented:** `exception` and `test_failure`. Anything else needs a manual `sess.rollback(...)` call.
- **`keep_last` doesn't free disk space.** It trims label bookkeeping, not the underlying checkpoint layers.
- **Default behavior discards work.** With default arguments (`destroy_on_exit=True`, `auto_commit=False`), exiting a `with session(...)` block destroys the container and writes nothing back to the host. Pass `auto_commit=True` explicitly if you want results persisted.
- **Untested against nested/concurrent tool-call crashes.** The dangling-tool-call cleanup handles the single-message case (one assistant tool-call message immediately before the checkpoint). Behavior under deeper crash scenarios hasn't been verified.

---

## API Reference

```python
session(name="rewind_sandbox", workspace=".", *, container_name=None,
        engine=None, memory=None, destroy_on_exit=True, auto_commit=False)

sess.write_file(path, content)
sess.read_file(path) -> str
sess.run(cmd) -> str                  # raises RuntimeError on non-zero exit
sess.run_tests(cmd=None) -> str       # defaults to "pytest"

sess.sync_memory(messages, message_format="auto")
sess.get_messages(message_format="auto") -> list

sess.checkpoint(label, messages=None) -> str
sess.rollback(label="latest", patch_notes=None, message_format="auto") -> list

sess.auto_checkpoint(trigger="before_tool_call", keep_last=None)
sess.auto_rollback(*events, to="latest", test_command=None)

sess.on_tool_call(messages=None, tool_name=None)
sess.on_tool_result(messages=None, error=None)

sess.start(workspace=None, force=False)
sess.attach()
sess.destroy()
sess.status() -> dict
sess.commit()                         # manual host export; auto_commit calls this on clean exit
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Docker not running | `docker version` should return cleanly, or start Docker Desktop |
| `RuntimeError: Session not started` | Use `with session(...)` or call `.start()` first |
| Work disappeared after the `with` block | Default `auto_commit=False` — pass `auto_commit=True` |
| `"Checkpoint X already exists"` | Checkpoint labels must be unique per session; pick a new label |

---

## Contact

Built by a solo developer. Feedback and bug reports welcome.

**Email:** rewind.sdk.dev@protonmail.com

## License

MIT — see [LICENSE](LICENSE).
