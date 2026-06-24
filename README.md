# Rewind SDK

**A transactional sandbox runtime for AI coding agents.** Run agent-generated code in an isolated container, checkpoint filesystem and conversation state together, and roll both back atomically when something breaks.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rahulb0802/rewind-sdk/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/rewind-sdk.svg)](https://pypi.org/project/rewind-sdk/)
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

Rewind boots an Alpine Linux Docker container, mounts your host workspace into it **read-only**, and gives the agent a writable OverlayFS layer to work in. Checkpointing is a layer-stacking operation (fast, with no file copying), and rollback discards layers back to a chosen point. Separately, it keeps a parallel in-memory record of your conversation messages, snapshotted at the same checkpoint label, so a filesystem rollback and a memory rollback always happen together via one call.

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

**Filesystem and message history are restored together, with one call.** This is the mechanic everything in this SDK is built around, making that pairing convenient.

---

## Verified Features

These are implemented and covered by the test suite or directly traceable in source:

- **OverlayFS checkpoints** — instant, layer-based snapshots of the sandbox filesystem (`engine.py`)
- **Paired memory rollback** — message history is truncated to match a filesystem checkpoint in one call
- **Dangling tool-call cleanup** — if the checkpoint point falls right after an assistant message that initiated a tool call, that message is automatically dropped too, so you don't hand a broken message history back to a strict-schema provider
- **Auto-checkpoint before tool calls** — `on_tool_call()` snapshots state automatically; `@session.tool` injects this for decorated tools
- **Auto-rollback on exception or test failure** — `auto_rollback("exception", "test_failure", ...)` triggers rollback inside `run()` / `run_tests()` and LangGraph's `invoke`/`stream`; per-tool scoping via `rollback_on_error`
- **JSON verifier contract** — `run_tests()` runs a verifier in-container, parses `{"status": "pass"|"fail"|"unknown"}` from stdout, retries on UNKNOWN, and returns a human-readable summary on PASS
- **Verification ledger** — append-only audit log of verification, escalation, and rollback events; survives `rollback()` calls
- **Escalation on UNKNOWN** — after retries are exhausted, `mode="interactive"` prompts continue/rollback/stop on stdin; `mode="agent"` halts with `VerificationHaltError` and preserves the container
- **`@session.tool` decorator** — LangChain-compatible tools with automatic checkpointing, scoped exception rollback, and `RuntimeError` → error-string conversion for the LLM
- **Two-phase commit to host** — host files are only touched if a session block exits without raising and `auto_commit=True` is set; halt-aware `__exit__` skips commit and optionally preserves the container
- **LangGraph adapter** — `wrap_langgraph()` keeps memory synced, re-raises `VerificationHaltError`, and triggers rollback on other unhandled exceptions
- **CLI and MCP server** — `rewind_cli.py` and `mcp_server.py` expose session operations including verification ledger history

---

## Installation

**Requirements:** Python 3.9+, Docker running locally.

```bash
pip install rewind-sdk
```

> **Note:** the PyPI package name is `rewind-sdk` (hyphen), but the importable
> Python module is `rewind_sdk` (underscore); this is not a typo.

```python
from rewind_sdk import session 
```

For LangGraph integration:
```bash
pip install "rewind-sdk[langgraph]"
```

### Install from source (for contributors)

```bash
git clone https://github.com/rahulb0802/rewind-sdk.git
cd rewind_sdk
pip install -e .
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
from rewind_sdk import session, Verifier

with session("agent", workspace="./src", auto_commit=True) as sess:
    sess.checkpoint("stable")

    sess.write_file("config.py", risky_change)
    sess.auto_rollback("exception", "test_failure", to="stable",
                       verifier=Verifier(command="pytest"))
    try:
        sess.run_tests()  # PASS: formatted summary; FAIL: RuntimeError + auto-rollback
    except RuntimeError:
        pass  # already rolled back to "stable"
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

`trigger="before_tool_call"` is the only trigger currently implemented. `keep_last` trims the SDK's own convenience label history (`_auto_labels`), but does **not** delete the underlying OverlayFS checkpoint layers, which remain on disk regardless of this setting. If you're watching container disk usage, this parameter won't help; there's currently no automatic checkpoint-layer pruning.

Auto-checkpoints fire when you call `sess.on_tool_call(...)` or use `@session.tool`, which calls `on_tool_call()` automatically before each decorated tool runs. You can also wire the LangGraph adapter's `before_tool_node` hook into a custom graph. It is not a global hook that activates on every tool call without one of these integrations.

### Auto-rollback

```python
from rewind_sdk import Verifier

sess.checkpoint("known_good")  # create this BEFORE risky work begins
sess.auto_rollback(
    "exception",
    "test_failure",
    to="known_good",
    verifier=Verifier(command="pytest", retries=2, timeout=30.0),
)
```

`"test_failure"` rollback requires a `Verifier` whose command prints JSON
(`{"status": "pass"|"fail"|"unknown", ...}`) to stdout. `"exception"` fires
rollback on `run()` / `run_tests()` errors and on failures inside
`@session.tool`-decorated tools. For decorated tools, rollback is scoped per
tool via `rollback_on_error` (see below); only tools with
`rollback_on_error=True` (the default) participate.

> **Important:** `to=` should almost always be an explicit checkpoint label
> created with `sess.checkpoint(...)` *before* the risky operation, not the
> default `"latest"`. Auto-checkpoints are taken immediately *before* each
> tool call, meaning the most recent auto-checkpoint can already contain
> the very change that caused the failure you're trying to recover from.
> `to="latest"` rolls back to that checkpoint, not to a known-good state.

```python
if sess.last_auto_rollback:
    print(sess.last_auto_rollback["event"], sess.last_auto_rollback["to"])
```

### `@session.tool` decorator

The `@session.tool` decorator wraps a function as a LangChain-compatible tool with
automatic `on_tool_call()` bookkeeping. When a tool raises `RuntimeError`, the
decorator converts it to an error string the LLM can read; if a rollback fired,
the string includes a `[REWIND]` notice naming the checkpoint that was restored.

```python
@sandbox.tool(rollback_on_error=False)
def run_sql(query: str) -> str:
    """Read-only query — failure should not roll back filesystem changes."""
    return sandbox.run(f"sqlite3 db.sqlite '{query}'")

@sandbox.tool(rollback_on_error=True)  # default; can be omitted
def run_script(path: str) -> str:
    """State-changing script — failure rolls back to the last checkpoint."""
    return sandbox.run(f"python3 {path}")
```

`rollback_on_error` controls whether `"exception"` auto-rollback applies to
failures inside that tool:

- **`rollback_on_error=True`** (default) — failures (including from `sandbox.run()`
  inside the tool) trigger auto-rollback when `"exception"` is configured.
- **`rollback_on_error=False`** — suppresses auto-rollback for that tool; use for
  read-only or side-effect-free operations where a failure should not discard
  other work in the sandbox.

`VerificationHaltError` (raised when a verifier returns UNKNOWN and escalation
resolves STOP) always propagates uncaught through `@session.tool` — it is a
session-level halt signal, not a tool error string.

Duplicate rollbacks to the same checkpoint are skipped automatically and
recorded in the ledger as `skipped_noop`.

---

## Verification and escalation

### Three-state verifier contract

When a `Verifier` is configured via `auto_rollback(...)`, `run_tests()` runs
`verifier.command` in the container and treats **JSON stdout** as the sole
verdict (process exit code is ignored):

```json
{"status": "pass", "summary": "All checks passed."}
{"status": "fail", "summary": "2 check(s) failed", "errors": ["..."]}
{"status": "unknown", "summary": "Verifier crashed", "error": "..."}
```

| Status | Behavior |
|--------|----------|
| **pass** | No rollback; ledger records verification; `run_tests()` returns a formatted summary string |
| **fail** | Auto-rollback to `to=` checkpoint; `run_tests()` raises `RuntimeError` (or `@session.tool` converts it to an error string the LLM can act on) |
| **unknown** | Retried up to `verifier.retries` times; if still unknown, escalation runs (see below) |

```python
from rewind_sdk import Verifier

sess.auto_rollback(
    "exception",
    "test_failure",
    to="known_good",
    verifier=Verifier(command="python3 verify.py", retries=2, retry_delay=1.0, timeout=30.0),
)

summary = sess.run_tests()  # human-readable string on PASS
```

### Verification ledger

`session.ledger` is an append-only record of verification, escalation, and
rollback events. It lives **outside** the rollback scope — `rollback()` never
touches it.

```python
for entry in sess.ledger.history():
    print(entry.event_type, entry.status, entry.resolution, entry.checkpoint)
```

### Session modes and escalation

```python
# Interactive (default): prompts [c]ontinue / [r]ollback / [s]top on stdin after UNKNOWN
sess = session("dev", workspace="./src")

# Agent / headless: UNKNOWN → STOP automatically; container preserved on halt
sess = session("agent", workspace="./src", mode="agent")
```

Override the default handler explicitly if needed:

```python
from rewind_sdk import session, stdin_escalation_handler, stop_escalation_handler

sess = session("custom", workspace="./src", escalation_handler=stdin_escalation_handler)
```

Escalation resolutions:

- **continue** — proceed without trustworthy verification; ledger records the decision; `run_tests()` still raises (UNKNOWN is never treated as PASS)
- **rollback** — revert to the `to=` checkpoint
- **stop** — raise `VerificationHaltError`; in `mode="agent"`, `__exit__` preserves the container and skips `auto_commit`

### Handling `VerificationHaltError` in agent apps

Catch halt **outside** the `with session(...)` block so `__exit__` can run
preserve-on-halt logic. Catching inside `with` and returning makes Python call
`__exit__(None, None, None)`, which skips container preservation.

```python
from rewind_sdk import session, VerificationHaltError, wrap_langgraph

sandbox = session("agent", workspace="./src", mode="agent", auto_commit=True)

try:
    with sandbox:
        sandbox.checkpoint("known_good")
        sandbox.auto_rollback("exception", "test_failure", to="known_good", verifier=...)

        @sandbox.tool
        def run_verify() -> str:
            return sandbox.run_tests()

        safe_agent = wrap_langgraph(agent, session=sandbox)
        for event in safe_agent.stream({"messages": messages}):
            ...

except VerificationHaltError as exc:
    print(exc)                              # halt details
    print(exc.checkpoint)                   # checkpoint at time of halt
    print(sandbox.engine.container_name)    # preserved container name
    print(sandbox.ledger.history())         # audit trail
```

---

## LangGraph Integration

Install with the LangGraph extra: `pip install "rewind-sdk[langgraph]"`

```python
import threading
from rewind_sdk import session, Verifier, VerificationHaltError, wrap_langgraph

tool_lock = threading.Lock()
sandbox = session("agent_sandbox", workspace="./my_codebase", mode="agent", auto_commit=True)

try:
    with sandbox:
        sandbox.auto_checkpoint(trigger="before_tool_call")
        sandbox.checkpoint("known_good")
        sandbox.auto_rollback(
            "exception",
            "test_failure",
            to="known_good",
            verifier=Verifier(command="pytest", retries=2, timeout=30.0),
        )

        @sandbox.tool(rollback_on_error=False)
        def read_file(path: str) -> str:
            """Read-only — query failures should not roll back other work."""
            with tool_lock:
                return sandbox.read_file(path)

        @sandbox.tool
        def write_file(path: str, content: str) -> str:
            """State-changing — failures trigger exception rollback."""
            with tool_lock:
                sandbox.write_file(path, content)
                return f"Wrote to {path}"

        @sandbox.tool
        def run_verify() -> str:
            """Run the configured verifier; FAIL rolls back, UNKNOWN halts."""
            with tool_lock:
                return sandbox.run_tests()

        agent = create_react_agent(llm, tools=[read_file, write_file, run_verify])
        safe_agent = wrap_langgraph(agent, session=sandbox)
        for event in safe_agent.stream({"messages": messages}):
            ...

except VerificationHaltError as exc:
    # sandbox container is preserved in mode="agent"
    ...
```

`@sandbox.tool` injects `on_tool_call()` before each tool run, so you do not
need to call it manually in decorated tools. `wrap_langgraph` keeps memory
synced and re-raises `VerificationHaltError` uncaught; other unhandled
exceptions trigger `"exception"` rollback via `on_tool_result`.

Your system prompt doesn't need to mention rollbacks, checkpoints, or recovery,
as the message-history correction happens in `memory.py`, not in the prompt.

A thread lock around tool execution is recommended because the sandbox is a
single container; concurrent writes from parallel tool calls aren't serialized
for you.

---

## CLI

> `rewind_cli.py` is included in the GitHub repo, not the PyPI package. Clone
> the repo (see [Install from source](#install-from-source-for-contributors))
> to use it.

```bash
python rewind_cli.py init ./my-project
python rewind_cli.py write src/app.py "print('hi')"
python rewind_cli.py checkpoint stable
python rewind_cli.py exec "pytest"
python rewind_cli.py rollback stable
python rewind_cli.py status
python rewind_cli.py ledger
python rewind_cli.py ledger --checkpoint stable
python rewind_cli.py destroy
```

Add `--json` for machine-readable output and `--quiet` to suppress stderr logging, which is useful if another agent is driving the CLI directly.

## MCP Server

> `mcp_server.py` is included in the GitHub repo, not the PyPI package. Clone
> the repo to use it.

`mcp_server.py` exposes session operations (`init_sandbox`, `execute_sandbox_command`, `write_sandbox_file`, `read_sandbox_file`, `sync_agent_memory`, `create_sandbox_checkpoint`, `rollback_sandbox_state`, `configure_auto_checkpoint`, `configure_auto_rollback`, `get_sandbox_status`, `get_ledger_history`) as MCP tools, for clients that want to drive a Rewind sandbox without writing Python. The MCP server uses `stop_escalation_handler` by default. Install with MCP extra: `pip install "rewind-sdk[mcp]"`.

---

## Known Limitations

Being direct and transparent (as this is still an early prototype):

- **Containers run `--privileged`.** This is required for the current OverlayFS mounting approach, but it means the sandbox container has broad host-kernel access, and it is not a hardened security boundary against a determined adversary. Treat it as protection against an agent's *accidental* mistakes (bad refactors, destructive commands), not as isolation against malicious code.
- **One framework integration.** Only LangGraph is supported today. The adapter pattern (`messages_to_dicts` / `dicts_to_messages`) is framework-agnostic in design, but no LangChain-only or CrewAI adapter exists yet.
- **No automatic concurrency control inside the SDK.** If you call sandbox methods from multiple threads, you need your own lock (see the LangGraph example above); the SDK does not serialize for you.
- **Exception rollback is opt-in per tool.** With `@session.tool`, you choose per tool whether failures trigger `"exception"` rollback via `rollback_on_error`. Read-only tools should set `rollback_on_error=False` so a query failure does not roll back unrelated filesystem changes. Outside decorated tools, `sess.run()` still rolls back on failure when configured.
- **`VerificationHaltError` must escape `with` uncaught.** Catch it outside the `with session(...)` block so `__exit__` can preserve the container in `mode="agent"`. Catching inside `with` and returning treats the exit as clean.
- **`keep_last` doesn't free disk space.** It trims label bookkeeping, not the underlying checkpoint layers.
- **Default behavior discards work.** With default arguments (`destroy_on_exit=True`, `auto_commit=False`), exiting a `with session(...)` block destroys the container and writes nothing back to the host. Pass `auto_commit=True` explicitly if you want results persisted. Halt in `mode="agent"` is the exception: the container is preserved even with `destroy_on_exit=True`.
- **Untested against multi-agent/complex tool calls.** The dangling-tool-call cleanup handles the single-message case (one assistant tool-call message immediately before the checkpoint). Behavior under deeper crash scenarios hasn't been verified.

---

## API Reference

```python
session(name="rewind_sandbox", workspace=".", *, container_name=None,
        engine=None, memory=None, destroy_on_exit=True, auto_commit=False,
        mode="interactive", escalation_handler=None)

sess.write_file(path, content)
sess.read_file(path) -> str
sess.run(cmd) -> str                  # raises RuntimeError on non-zero exit
sess.run_tests(cmd=None) -> str       # uses verifier.command when cmd omitted;
                                      # returns formatted summary on PASS

sess.sync_memory(messages, message_format="auto")
sess.get_messages(message_format="auto") -> list

sess.checkpoint(label, messages=None) -> str
sess.rollback(label="latest", patch_notes=None, message_format="auto") -> list

sess.auto_checkpoint(trigger="before_tool_call", keep_last=None)
sess.auto_rollback(*events, to=None, verifier=None)

sess.tool(fn=None, *, name=None, rollback_on_error=True)  # decorator

sess.on_tool_call(messages=None, tool_name=None)
sess.on_tool_result(messages=None, error=None)

sess.start(workspace=None, force=False)
sess.attach()
sess.destroy()
sess.status() -> dict
sess.commit()                         # manual host export; auto_commit calls this on clean exit

sess.ledger                          # VerificationLedger (survives rollback)
sess.get_ledger() -> VerificationLedger
sess.last_auto_rollback              # dict after most recent auto-rollback, or None
```

### Verification exports

```python
from rewind_sdk import (
    Verifier,
    VerificationHaltError,
    VerificationStatus,
    VerificationResult,
    VerificationLedger,
    LedgerEntry,
    EscalationContext,
    EscalationResolution,
    format_verification_result,
    parse_verifier_output,
    stdin_escalation_handler,
    stop_escalation_handler,
    wrap_langgraph,
)
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Docker not running | `docker version` should return cleanly, or start Docker Desktop |
| `RuntimeError: Session not started` | Use `with session(...)` or call `.start()` first |
| Work disappeared after the `with` block | Default `auto_commit=False`, pass `auto_commit=True` |
| `"Checkpoint X already exists"` | Checkpoint labels must be unique per session; pick a new label |
| Container destroyed after UNKNOWN halt | Catch `VerificationHaltError` **outside** `with session(...)`, not inside; use `mode="agent"` |
| `run_tests()` raises but agent keeps going | FAIL inside `@session.tool` becomes an error string (by design); UNKNOWN raises `VerificationHaltError` at the session boundary |
| Verifier always returns unknown | Ensure stdout is a single JSON object with a `"status"` field; use stderr for debug logs |

---

## Contact

Built by a solo developer. Feedback and bug reports welcome.

**Email:** rewind.sdk.dev@protonmail.com
**GitHub Issues:** https://github.com/rahulb0802/rewind-sdk/issues

## License

MIT: see [LICENSE](https://github.com/rahulb0802/rewind-sdk/blob/master/LICENSE).